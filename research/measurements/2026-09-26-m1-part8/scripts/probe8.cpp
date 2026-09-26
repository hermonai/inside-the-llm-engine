// Part VIII probes for 'Inside the LLM Engine' against llama.cpp's C API (build 8660).
//
//   probe8 batch <model.gguf> <target.txt> <fillers.txt> <n_gen> <out.json> B1 [B2 ...]
//     Greedy-decodes one target prompt alone (B = 1) and inside batches of B sequences.
//     The target's prompt is always prefilled on its own, so its cache is computed
//     identically every time; only the decode steps, which carry one token per
//     sequence, see a different batch. Records the target's tokens and, while its
//     tokens still agree with the first B = 1 run, the largest logit difference from
//     that run at every step. The first B is run twice to check run-to-run repeat.
//
//   probe8 route <model.gguf> <prompts.txt> <n_gen> <out.json>
//     For each prompt (one per line, "\n" escapes allowed), prefills and greedily
//     decodes n_gen tokens alone, recording every MoE layer's selected experts
//     (tensor "ffn_moe_topk-<layer>") for every token, via the eval callback.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const char * path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<std::string> read_lines(const char * path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::string s;
        for (size_t i = 0; i < line.size(); i++) {
            if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') { s += '\n'; i++; }
            else s += line[i];
        }
        out.push_back(s);
    }
    return out;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_bos) {
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, add_bos, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, add_bos, true);
    return toks;
}

static int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;   // lowest id wins ties
    return best;
}

// ---------------------------------------------------------------- batch mode

struct BatchRun {
    int B;
    std::vector<int> tokens;          // target's generated tokens
    std::vector<double> max_dlogit;   // vs baseline, while histories agree
    std::vector<double> margin;       // top-1 minus top-2 logit, this run
    int first_divergence = -1;        // index of first differing token vs baseline
    double seconds = 0;
};

static BatchRun run_batch(llama_model * model, const std::vector<llama_token> & target,
                          const std::vector<std::vector<llama_token>> & fillers, int B, int n_gen,
                          const std::vector<std::vector<float>> * base_logits,
                          const std::vector<int> * base_tokens, std::vector<std::vector<float>> * keep) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    size_t longest = target.size();
    for (int s = 1; s < B; s++) longest = std::max(longest, fillers[(s - 1) % fillers.size()].size());
    const int per_seq = (int) longest + n_gen + 16;

    llama_context_params cp = llama_context_default_params();
    cp.n_seq_max = B;
    cp.n_ctx = per_seq * B;
    cp.n_batch = std::max<uint32_t>(512, (uint32_t) (per_seq * B));
    cp.n_ubatch = 512;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed for B=%d\n", B); exit(1); }

    BatchRun run;
    run.B = B;
    const auto t0 = ggml_time_us();

    // 1. the target's prompt, alone
    llama_batch b = llama_batch_init((int32_t) std::max<size_t>(target.size(), 1) + per_seq * B, 0, 1);
    b.n_tokens = 0;
    for (size_t i = 0; i < target.size(); i++) {
        b.token[b.n_tokens] = target[i]; b.pos[b.n_tokens] = (llama_pos) i;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = i + 1 == target.size();
        b.n_tokens++;
    }
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed (target prefill)\n"); exit(1); }
    std::vector<int> last_idx(B, -1);
    std::vector<llama_token> cur(B);
    std::vector<int> pos(B, 0);
    cur[0] = argmax(llama_get_logits_ith(ctx, b.n_tokens - 1), n_vocab);
    pos[0] = (int) target.size();
    // record the target's first generated token as step 0
    {
        const float * lg = llama_get_logits_ith(ctx, b.n_tokens - 1);
        std::vector<float> v(lg, lg + n_vocab);
        std::vector<float> s = v; std::nth_element(s.begin(), s.begin() + 1, s.end(), std::greater<float>());
        run.margin.push_back((double) s[0] - (double) s[1]);
        if (keep) keep->push_back(v);
        run.tokens.push_back(cur[0]);
        if (base_logits && base_tokens) {
            double m = 0;
            for (int i = 0; i < n_vocab; i++) m = std::max(m, (double) std::fabs(v[i] - (*base_logits)[0][i]));
            run.max_dlogit.push_back(m);
            if (cur[0] != (*base_tokens)[0]) run.first_divergence = 0;
        }
    }

    // 2. the fillers' prompts, together
    if (B > 1) {
        b.n_tokens = 0;
        for (int s = 1; s < B; s++) {
            const auto & f = fillers[(s - 1) % fillers.size()];
            for (size_t i = 0; i < f.size(); i++) {
                b.token[b.n_tokens] = f[i]; b.pos[b.n_tokens] = (llama_pos) i;
                b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = s;
                b.logits[b.n_tokens] = i + 1 == f.size();
                if (i + 1 == f.size()) last_idx[s] = b.n_tokens;
                b.n_tokens++;
            }
            pos[s] = (int) f.size();
        }
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed (filler prefill)\n"); exit(1); }
        for (int s = 1; s < B; s++) cur[s] = argmax(llama_get_logits_ith(ctx, last_idx[s]), n_vocab);
    }

    // 3. decode all B sequences together, one token each per step
    for (int step = 1; step < n_gen; step++) {
        b.n_tokens = 0;
        for (int s = 0; s < B; s++) {
            b.token[b.n_tokens] = cur[s]; b.pos[b.n_tokens] = pos[s]++;
            b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = s;
            b.logits[b.n_tokens] = 1;
            b.n_tokens++;
        }
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed (step %d)\n", step); exit(1); }
        for (int s = 0; s < B; s++) {
            const float * lg = llama_get_logits_ith(ctx, s);
            cur[s] = argmax(lg, n_vocab);
            if (s == 0) {
                std::vector<float> v(lg, lg + n_vocab);
                std::vector<float> srt = v;
                std::nth_element(srt.begin(), srt.begin() + 1, srt.end(), std::greater<float>());
                run.margin.push_back((double) srt[0] - (double) srt[1]);
                if (keep) keep->push_back(v);
                run.tokens.push_back(cur[0]);
                if (base_logits && base_tokens && run.first_divergence < 0) {
                    double m = 0;
                    for (int i = 0; i < n_vocab; i++) m = std::max(m, (double) std::fabs(v[i] - (*base_logits)[step][i]));
                    run.max_dlogit.push_back(m);
                    if (cur[0] != (*base_tokens)[step]) run.first_divergence = step;
                }
            }
        }
    }
    run.seconds = (ggml_time_us() - t0) / 1e6;
    llama_batch_free(b);
    llama_free(ctx);
    return run;
}

static void write_runs(const char * path, const std::vector<BatchRun> & runs, const std::string & model_path) {
    FILE * f = fopen(path, "w");
    fprintf(f, "{\"model\": \"%s\", \"runs\": [\n", model_path.c_str());
    for (size_t r = 0; r < runs.size(); r++) {
        const auto & x = runs[r];
        fprintf(f, " {\"B\": %d, \"seconds\": %.3f, \"first_divergence\": %d, \"tokens\": [", x.B, x.seconds, x.first_divergence);
        for (size_t i = 0; i < x.tokens.size(); i++) fprintf(f, "%s%d", i ? "," : "", x.tokens[i]);
        fprintf(f, "], \"max_dlogit\": [");
        for (size_t i = 0; i < x.max_dlogit.size(); i++) fprintf(f, "%s%.6g", i ? "," : "", x.max_dlogit[i]);
        fprintf(f, "], \"margin\": [");
        for (size_t i = 0; i < x.margin.size(); i++) fprintf(f, "%s%.6g", i ? "," : "", x.margin[i]);
        fprintf(f, "]}%s\n", r + 1 < runs.size() ? "," : "");
    }
    fprintf(f, "]}\n");
    fclose(f);
}

// ---------------------------------------------------------------- route mode

struct RouteCapture {
    int n_layer = 0;
    int k = 0;
    // per decode call: per layer: flattened [n_tokens][k]
    std::vector<std::vector<std::vector<int32_t>>> calls;
    std::vector<uint8_t> buf;
    int pending_call = -1;
};

static bool route_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * rc = (RouteCapture *) ud;
    const bool want = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) return want;
    if (!want) return true;
    const int il = atoi(t->name + 13);
    const int k = (int) t->ne[0], n_tok = (int) t->ne[1];
    rc->k = k;
    std::vector<int32_t> ids((size_t) k * n_tok);
    // rows may be strided (a view); copy row by row
    for (int j = 0; j < n_tok; j++) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(ids.data() + (size_t) j * k, (const char *) t->data + (size_t) j * t->nb[1], (size_t) k * sizeof(int32_t));
        } else {
            ggml_backend_tensor_get(t, ids.data() + (size_t) j * k, (size_t) j * t->nb[1], (size_t) k * sizeof(int32_t));
        }
    }
    auto & call = rc->calls.back();
    if ((int) call.size() <= il) call.resize(il + 1);
    call[il] = std::move(ids);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: see source\n"); return 2; }
    const std::string mode = argv[1];
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (mode == "batch") {
        const auto target = tokenize(vocab, read_file(argv[3]), true);
        std::vector<std::vector<llama_token>> fillers;
        for (const auto & line : read_lines(argv[4])) fillers.push_back(tokenize(vocab, line, true));
        const int n_gen = atoi(argv[5]);
        std::vector<int> Bs;
        for (int i = 7; i < argc; i++) Bs.push_back(atoi(argv[i]));
        std::vector<std::vector<float>> base_logits;
        std::vector<BatchRun> runs;
        BatchRun base = run_batch(model, target, fillers, 1, n_gen, nullptr, nullptr, &base_logits);
        base.first_divergence = -1;
        runs.push_back(base);
        fprintf(stderr, "B=1 base %.2f s\n", base.seconds);
        for (int B : Bs) {
            BatchRun r = run_batch(model, target, fillers, B, n_gen, &base_logits, &base.tokens, nullptr);
            fprintf(stderr, "B=%d %.2f s first_divergence=%d max_dlogit(step0..)=%g\n", B, r.seconds,
                    r.first_divergence, r.max_dlogit.empty() ? -1.0 : *std::max_element(r.max_dlogit.begin(), r.max_dlogit.end()));
            runs.push_back(r);
        }
        write_runs(argv[6], runs, argv[2]);
    } else if (mode == "route") {
        const auto prompts = read_lines(argv[3]);
        const int n_gen = atoi(argv[4]);
        RouteCapture rc;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 8192; cp.n_batch = 4096; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.no_perf = true;
        cp.cb_eval = route_cb; cp.cb_eval_user_data = &rc;
        llama_context * ctx = llama_init_from_model(model, cp);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        FILE * f = fopen(argv[5], "w");
        fprintf(f, "{\"model\": \"%s\", \"prompts\": [\n", argv[2]);
        for (size_t p = 0; p < prompts.size(); p++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            rc.calls.clear();
            auto toks = tokenize(vocab, prompts[p], true);
            llama_batch b = llama_batch_init((int32_t) toks.size() + 1, 0, 1);
            b.n_tokens = 0;
            for (size_t i = 0; i < toks.size(); i++) {
                b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = (llama_pos) i;
                b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
                b.logits[b.n_tokens] = i + 1 == toks.size(); b.n_tokens++;
            }
            rc.calls.emplace_back();
            if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed\n"); return 1; }
            llama_token cur = argmax(llama_get_logits_ith(ctx, b.n_tokens - 1), n_vocab);
            int pos = (int) toks.size();
            std::vector<int> gen;
            for (int step = 0; step < n_gen; step++) {
                gen.push_back(cur);
                if (llama_vocab_is_eog(vocab, cur)) break;
                b.n_tokens = 1; b.token[0] = cur; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
                rc.calls.emplace_back();
                if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
                cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
            }
            llama_batch_free(b);
            // write: prefill routing [layer][token*k], then decode routing [step][layer][k]
            fprintf(f, " {\"n_prompt\": %zu, \"n_gen\": %zu, \"k\": %d, \"prefill\": [", toks.size(), gen.size(), rc.k);
            const auto & pre = rc.calls[0];
            for (size_t il = 0; il < pre.size(); il++) {
                fprintf(f, "%s[", il ? "," : "");
                for (size_t i = 0; i < pre[il].size(); i++) fprintf(f, "%s%d", i ? "," : "", pre[il][i]);
                fprintf(f, "]");
            }
            fprintf(f, "], \"decode\": [");
            for (size_t c = 1; c < rc.calls.size(); c++) {
                fprintf(f, "%s[", c > 1 ? "," : "");
                for (size_t il = 0; il < rc.calls[c].size(); il++) {
                    fprintf(f, "%s[", il ? "," : "");
                    for (size_t i = 0; i < rc.calls[c][il].size(); i++) fprintf(f, "%s%d", i ? "," : "", rc.calls[c][il][i]);
                    fprintf(f, "]");
                }
                fprintf(f, "]");
            }
            fprintf(f, "], \"generated\": [");
            for (size_t i = 0; i < gen.size(); i++) fprintf(f, "%s%d", i ? "," : "", gen[i]);
            fprintf(f, "]}%s\n", p + 1 < prompts.size() ? "," : "");
            fprintf(stderr, "prompt %zu: %zu tokens, %zu generated, %zu layers captured\n", p, toks.size(), gen.size(), pre.size());
        }
        fprintf(f, "]}\n");
        fclose(f);
        llama_free(ctx);
    }
    llama_model_free(model);
    return 0;
}
