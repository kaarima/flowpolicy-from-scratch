// Test harness for the C++ port. Session 1: load the weights and the test vectors, print every tensor's
// name, shape and FNV-1a hash (python/scripts/m3_check_loader.py compares these with NumPy, bitwise).
// Later sessions add the forward-pass checks against vectors.bin (see the TODO list below).
//
// Usage: test_vectors [weights.bin] [vectors.bin]   (defaults: the files in this repo)
#include <cstdio>
#include <exception>
#include <string>

#include "flowpolicy/fpbin.hpp"

#ifndef FP_WEIGHTS
#define FP_WEIGHTS "weights/film_g.bin"
#endif
#ifndef FP_VECTORS
#define FP_VECTORS "tests/vectors/vectors.bin"
#endif

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
        // TODO (your code, in order):
        //   1. Tensor + matmul + Linear:  x_t @ Win + bin  vs  h0           (max abs err < 1e-5)
        //   2. SiLU + condition encoder:  SiLU(cond @ Wc + bc)  vs  cond_e
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
