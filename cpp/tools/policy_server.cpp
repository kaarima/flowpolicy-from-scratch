// Runs the C++ policy behind a pipe, for the closed-loop eval in Python (python/scripts/m3_eval_cpp.py).
// Protocol (binary, little-endian), repeated until EOF on stdin:
//   in : uint32 B, then B x 36 float32  (raw obs in pixels: [frame t-1 (18), frame t (18)])
//   out: B x 16 x 2 float32             (action chunk in pixels; the caller executes the first 8)
// The initial noise comes from std::mt19937 (seed = argv[2], default 0) + std::normal_distribution.
//
// Usage: policy_server [weights.bin] [seed]
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "flowpolicy/policy.hpp"

#ifndef FP_WEIGHTS
#define FP_WEIGHTS "weights/film_g.bin"
#endif

int main(int argc, char** argv) {
    try {
        const auto weights = fp::TensorFile::load(argc > 1 ? argv[1] : FP_WEIGHTS);
        const fp::FlowPolicy policy(weights, 10);
        std::mt19937 gen(argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 0u);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uint32_t B = 0;
        while (std::fread(&B, sizeof(B), 1, stdin) == 1) {
            fp::Tensor obs({B, 36});
            if (std::fread(obs.data(), sizeof(float), obs.size(), stdin) != obs.size()) return 3;
            fp::Tensor noise({B, 32});
            for (std::size_t i = 0; i < noise.size(); ++i) noise[i] = normal(gen);
            const fp::Tensor act = policy.act(obs, noise);
            std::fwrite(act.data(), sizeof(float), act.size(), stdout);
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "policy_server: %s\n", e.what());
        return 1;
    }
    return 0;
}
