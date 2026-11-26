// Test harness for the C++ port. Session 1: load the weights and the test vectors, print every tensor's
// name, shape and FNV-1a hash (python/scripts/m3_check_loader.py compares these with NumPy, bitwise).
// Later sessions add the forward-pass checks against vectors.bin (see the TODO list below).
//
// Usage: test_vectors [weights.bin] [vectors.bin]   (defaults: the files in this repo)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

#include "flowpolicy/fpbin.hpp"
#include "flowpolicy/film.hpp"
#include "flowpolicy/policy.hpp"
#include "flowpolicy/tensor.hpp"

#ifndef FP_WEIGHTS
#define FP_WEIGHTS "weights/film_g.bin"
#endif
#ifndef FP_VECTORS
#define FP_VECTORS "tests/vectors/vectors.bin"
#endif

// allclose-style check against the NumPy vector `name`: |got - want| <= atol + rtol * |want| for every element.
// Prints max abs error, max rel error (|got - want| / max(|want|, 1e-12)) and PASS/FAIL.
static bool check(const fp::TensorFile& vec, const char* name, const fp::Tensor& got, float atol = 1e-5f,
                  float rtol = 1e-5f) {
    const fp::Tensor want = fp::Tensor::from(vec.get(name));
    if (got.shape() != want.shape()) {
        std::printf("check %s shape %s vs %s FAIL\n", name, got.shape_str().c_str(), want.shape_str().c_str());
        return false;
    }
    double max_abs = 0, max_rel = 0;
    bool ok = true;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - want[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(std::fabs(static_cast<double>(want[i])), 1e-12));
        ok = ok && d <= atol + rtol * std::fabs(static_cast<double>(want[i]));
    }
    std::printf("check %-17s %-11s max_abs %.2e  max_rel %.2e  (atol %.0e rtol %.0e)  %s\n", name,
                got.shape_str().c_str(), max_abs, max_rel, static_cast<double>(atol), static_cast<double>(rtol),
                ok ? "PASS" : "FAIL");
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
        ok &= check(vectors, "h0", fp::Linear::from(weights, "Win", "bin").forward(x_t));
        // 2. Linear + SiLU: condition encoder
        ok &= check(vectors, "cond_e", fp::silu(fp::Linear::from(weights, "Wc", "bc").forward(cond)));
        // 3. time embedding, and the condition [obs_n, temb]
        const fp::Tensor temb = fp::time_embedding(fp::Tensor::from(vectors.get("t")),
                                                   fp::Tensor::from(weights.get("time_freqs")));
        ok &= check(vectors, "temb", temb);
        ok &= check(vectors, "cond", fp::concat_cols(fp::Tensor::from(vectors.get("obs_n")), temb));
        // 4. FiLM block 0, from the NumPy inputs h0 and cond_e (isolates the block from earlier errors)
        fp::Tensor scale0;
        const fp::Tensor b0 = fp::FilmBlock::from(weights, 0).forward(
            fp::Tensor::from(vectors.get("h0")), fp::Tensor::from(vectors.get("cond_e")), &scale0);
        ok &= check(vectors, "film0_scale", scale0);
        ok &= check(vectors, "block0_out", b0);
        // 5. full network from the NumPy inputs (x_t, obs_n, t)
        const fp::FlowPolicy policy(weights, 10);
        const fp::Tensor obs_n = fp::Tensor::from(vectors.get("obs_n"));
        ok &= check(vectors, "velocity",
                    policy.net().velocity(x_t, obs_n, fp::Tensor::from(vectors.get("t"))));
        // 6. whole pipeline from raw pixels: normalise obs, Euler from NumPy's noise0, unnormalise to pixels
        const fp::Tensor obs_n_cpp = policy.normalize_obs(fp::Tensor::from(vectors.get("obs_raw")));
        ok &= check(vectors, "obs_n", obs_n_cpp);
        const fp::Tensor chunk_n = policy.sample_normalized(obs_n_cpp, fp::Tensor::from(vectors.get("noise0")));
        ok &= check(vectors, "sample_chunk_n", fp::Tensor({chunk_n.dim(0), 16, 2}, std::vector<float>(
                                                   chunk_n.data(), chunk_n.data() + chunk_n.size())));
        ok &= check(vectors, "sample_actions_px", policy.unnormalize_actions(chunk_n));
        if (!ok) return 2;
        std::printf("# all checks PASS\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
