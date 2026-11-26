// Batch-1 latency of the C++ policy (one decision = obs normalisation + N Euler steps + action unnormalisation),
// for N = 1, 2, 5, 10. Warm-up, then the median (and p90) over --calls decisions. Also counts heap allocations
// inside the Euler loop (should be 0: everything lives in a preallocated Workspace).
//
// Usage: bench [weights.bin] [calls]   (defaults: repo weights, 1000)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "flowpolicy/policy.hpp"

#ifndef FP_WEIGHTS
#define FP_WEIGHTS "weights/film_g.bin"
#endif
#ifndef FP_VECTORS
#define FP_VECTORS "tests/vectors/vectors.bin"
#endif
#ifndef FP_FLAGS
#define FP_FLAGS "?"
#endif

static std::atomic<long> g_allocs{0};
void* operator new(std::size_t n) {
    ++g_allocs;
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main(int argc, char** argv) {
    const auto weights = fp::TensorFile::load(argc > 1 ? argv[1] : FP_WEIGHTS);
    const int calls = argc > 2 ? std::atoi(argv[2]) : 1000;
    const auto vectors = fp::TensorFile::load(FP_VECTORS);
    fp::FlowPolicy policy(weights, 10);

    const auto& raw = vectors.get("obs_raw");  // first real obs window, (1, 36)
    const fp::Tensor obs({1, 36}, std::vector<float>(raw.data.begin(), raw.data.begin() + 36));
    fp::Workspace ws(policy.net(), 1);
    std::mt19937 gen(0);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::printf("build flags: %s\n", FP_FLAGS);
    std::printf("%-12s %12s %12s %16s\n", "euler steps", "median ms", "p90 ms", "allocs in loop");
    for (int steps : {1, 2, 5, 10}) {
        policy.set_euler_steps(steps);
        std::vector<double> ms;
        long loop_allocs = 0;
        for (int c = 0; c < calls + 50; ++c) {
            const auto t0 = std::chrono::steady_clock::now();
            const fp::Tensor obs_n = policy.normalize_obs(obs);
            for (std::size_t i = 0; i < ws.a.size(); ++i) ws.a[i] = normal(gen);
            const long before = g_allocs.load();
            policy.sample_normalized_inplace(obs_n, ws);
            loop_allocs += g_allocs.load() - before;
            const fp::Tensor act = policy.unnormalize_actions(ws.a);
            const auto t1 = std::chrono::steady_clock::now();
            if (c >= 50) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (act[0] != act[0]) return 1;  // keep the result alive
        }
        std::sort(ms.begin(), ms.end());
        std::printf("%-12d %12.3f %12.3f %16ld\n", steps, ms[ms.size() / 2], ms[ms.size() * 9 / 10], loop_allocs);
    }
    return 0;
}
