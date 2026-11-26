// Test harness for the C++ port. Session 1: load the weights and the test vectors, print every tensor's
// name, shape and FNV-1a hash (python/scripts/m3_check_loader.py compares these with NumPy, bitwise).
// Later sessions add the forward-pass checks against vectors.bin (see the TODO list below).
//
// Usage: test_vectors [weights.bin] [vectors.bin]   (defaults: the files in this repo)
#include <cstdio>
#include <exception>
#include <string>

#include "flowpolicy/fpbin.hpp"
#include "flowpolicy/tensor.hpp"

#ifndef FP_WEIGHTS
#define FP_WEIGHTS "weights/film_g.bin"
#endif
#ifndef FP_VECTORS
#define FP_VECTORS "tests/vectors/vectors.bin"
#endif

// Compares got with the NumPy vector `name`; prints "check <name> <shape> max_abs_err <e> PASS|FAIL".
static bool check(const fp::TensorFile& vec, const char* name, const fp::Tensor& got, float tol) {
    const fp::Tensor want = fp::Tensor::from(vec.get(name));
    const float err = fp::max_abs_diff(got, want);
    const bool ok = err < tol;
    std::printf("check %s %s max_abs_err %.3e tol %.0e %s\n", name, got.shape_str().c_str(), static_cast<double>(err),
                static_cast<double>(tol), ok ? "PASS" : "FAIL");
    return ok;
}

static void dump(const char* label, const fp::TensorFile& tf) {
    std::size_t total = 0;
    for (const auto& t : tf.tensors()) {
        std::printf("%s %s %s %016llx\n", label, t.name.c_str(), t.shape_str().c_str(),
                    static_cast<unsigned long long>(fp::fnv1a64(t.data.data(), t.size())));
        total += t.size();
    }
    std::printf("# %s: %zu tensors, %zu floats\n", label, tf.tensors().size(), total);
}

int main(int argc, char** argv) {
    const std::string wpath = argc > 1 ? argv[1] : FP_WEIGHTS;
    const std::string vpath = argc > 2 ? argv[2] : FP_VECTORS;
    try {
        const auto weights = fp::TensorFile::load(wpath);
        const auto vectors = fp::TensorFile::load(vpath);
        dump("weights", weights);
        dump("vectors", vectors);

        bool ok = true;
        const fp::Tensor x_t = fp::Tensor::from(vectors.get("x_t"));
        const fp::Tensor cond = fp::Tensor::from(vectors.get("cond"));
        // 1. Tensor + matmul + Linear: input projection (no activation)
        ok &= check(vectors, "h0", fp::Linear::from(weights, "Win", "bin").forward(x_t), 1e-5f);
        // 2. Linear + SiLU: condition encoder
        ok &= check(vectors, "cond_e", fp::silu(fp::Linear::from(weights, "Wc", "bc").forward(cond)), 1e-5f);
        if (!ok) return 2;
        // TODO, in order:
        //   3. time embedding:            temb(t, time_freqs)   vs  temb
        //   4. FiLM block 0:              vs  film0_scale, block0_out
        //   5. full network:              vs  velocity
        //   6. Euler sampler from noise0: vs  sample_chunk_n, sample_actions_px
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
