// Independent container and graph check against the pinned ggml C API.
// Only the book's tiny trusted fixture belongs on this optional path.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void check_projection(ggml_context * work, ggml_tensor * weight,
                             const float * input, int width,
                             const float * expected, int outputs) {
    auto * x = ggml_new_tensor_2d(work, GGML_TYPE_F32, width, 1);
    std::memcpy(x->data, input, width * sizeof(float));
    auto * y = ggml_mul_mat(work, weight, x);
    require(y->ne[0] == outputs && y->ne[1] == 1, "output shape mismatch");
    auto * graph = ggml_new_graph_custom(work, 32, false);
    ggml_build_forward_expand(graph, y);
    require(ggml_graph_compute_with_ctx(work, graph, 1) == GGML_STATUS_SUCCESS,
            "graph execution failed");
    const auto * actual = static_cast<const float *>(y->data);
    for (int i = 0; i < outputs; ++i) {
        std::printf(" %.9f (expected %.9f)", actual[i], expected[i]);
        std::fflush(stdout);
        require(std::isfinite(actual[i]) && std::fabs(actual[i] - expected[i]) < 1e-5f,
                "projection differs from hand calculation");
    }
    std::puts("");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: gguf-reference <book-fixture.gguf>");
        ggml_context * raw_weights = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(
            gguf_init_from_file(argv[1], {false, &raw_weights}), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(raw_weights, ggml_free);
        require(file && weights, "ggml rejected fixture");
        require(gguf_get_version(file.get()) == 3, "unexpected GGUF version");
        require(gguf_get_data_offset(file.get()) == 192, "wrong data origin");
        require(gguf_get_n_tensors(file.get()) == 2, "wrong tensor count");
        auto * dense = ggml_get_tensor(weights.get(), "dense.weight");
        auto * packed = ggml_get_tensor(weights.get(), "packed.weight");
        require(dense && packed, "tensor names missing");
        require(dense->type == GGML_TYPE_F32 && dense->ne[0] == 4 && dense->ne[1] == 2,
                "wrong dense descriptor");
        require(packed->type == GGML_TYPE_Q4_0 && packed->ne[0] == 32 && packed->ne[1] == 1,
                "wrong packed descriptor");
        require(ggml_nbytes(dense) == 32 && ggml_nbytes(packed) == 18,
                "wrong tensor byte count");
        float decoded[32];
        const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
        require(traits && traits->to_float, "Q4_0 decoder unavailable");
        traits->to_float(packed->data, decoded, 32);
        for (int i = 0; i < 32; ++i) {
            const float expected = i < 16 ? (i - 8) * .5f : (23 - i) * .5f;
            require(decoded[i] == expected, "weight decode or nibble order differs");
        }
        std::puts("Weight decoder: all 32 values match exactly");
        // Explicit teaching arena, not a measured minimum memory footprint.
        std::unique_ptr<ggml_context, decltype(&ggml_free)> work(
            ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
        require(bool(work), "work context allocation failed");
        const float input[] = {1, 2, 4, 8}, dense_expected[] = {49, 19};
        std::printf("F32 projection:");
        check_projection(work.get(), dense, input, 4, dense_expected, 2);
        float ones[32];
        for (float & x : ones) x = 1;
        // The pinned CPU Q4_0 dot path converts these inputs to Q8_0.
        // round_binary16(1/127) * 127 = 0.99993896484375 exactly.
        require(ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot_type == GGML_TYPE_Q8_0,
                "pinned Q4_0 activation type changed");
        const float activation_factor = 0.99993896484375f;
        const float packed_expected[] = {-8 * activation_factor};
        std::printf("Q4_0 projection:");
        check_projection(work.get(), packed, ones, 32, packed_expected, 1);
        // Basis inputs expose each decoded coordinate, not only their sum.
        for (int i = 0; i < 32; ++i) {
            float basis[32] = {};
            basis[i] = 1;
            const float expected[] = {decoded[i] * activation_factor};
            std::printf("basis[%02d]:", i);
            check_projection(work.get(), packed, basis, 32, expected, 1);
        }
        std::puts("PASS: ggml container, orientation and all 32 packed coordinates");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
