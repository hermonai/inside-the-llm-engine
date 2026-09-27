# Chapter 6 lab: a cost calculator you can trust

`napkin.py` evaluates Chapter 6's step-time model for any model shape and any
machine:

    t_step(B, S) = max((W + B*S*m_kv) / (eta_b*beta), B*F(S) / (eta_p*pi)) + t0

and prints time per output token, tokens per second per user and in aggregate,
the largest batch that fits, the context above which decode stays
bandwidth-bound, the batch a TPOT objective allows, and dollars per million
output tokens. Models come from a Hugging Face `config.json` (`--config`) or
three presets; machines from datasheet presets (M1 GPU, H100, H200, B200).
A mixture of experts reads the union of its batch's experts (uniform routing).

**Oracle.** `test_napkin.py` reproduces every table in Chapter 6; a change that
moves a number in the book fails a test.

**Predict.** Calibrate `--eta-b` and `--eta-p` with the labs of Chapters 3 and
4, then predict your machine's `llama-batched-bench` step times at two
contexts before running it.

**Run.** From the repository root:

```bash
python3 code/labs/ch06-napkin/napkin.py --model qwen3-8b --machine h100 --tpot 15
python3 code/labs/ch06-napkin/napkin.py --model qwen3-30b-a3b --machine h100 --batches 1,8,32
python3 code/labs/ch06-napkin/napkin.py --model llama-3.2-3b --machine m1-gpu \
    --bytes 0.626 --kv-bytes 2 --context 2064 --eta-b 0.72 --eta-p 0.7 --batches 1,2,4,8
python3 -m unittest discover -s code/labs/ch06-napkin
```

**Explain.** Attribute each prediction error to a term: an error that does not
change with context belongs to the weight term (a kernel that reads weights
more than once, a fixed overhead); an error that grows with context belongs to
the cache term (an attention kernel that does not stream its cache).

**Break.** Compare the uniform-routing prediction for a mixture of experts with
a measurement, or measure with flash attention off and watch the cache term's
error grow.
