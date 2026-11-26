// Full FiLM flow policy (inference): network, normalisation, Euler sampler. Mirrors python/flowmatch/policy.py.
#pragma once

#include <vector>

#include "flowpolicy/film.hpp"

namespace fp {

// Velocity network v(x_t, obs_n, t):
//   e = SiLU([obs_n, temb(t)] @ Wc + bc);  h = x_t @ Win + bin;  h = block_k(h, e) for k = 0..;  v = SiLU(h) @ Wout + bout
struct PolicyNet {
    Linear cond_enc, in_proj, out;
    std::vector<FilmBlock> blocks;
    Tensor time_freqs;

    static PolicyNet from(const TensorFile& f);  // uses every W1_k ... present
    // x_t (B, 32), obs_n (B, 36), t (B,) -> (B, 32)
    Tensor velocity(const Tensor& x_t, const Tensor& obs_n, const Tensor& t) const;
};

class FlowPolicy {
public:
    static constexpr std::size_t kFrameDim = 18, kObsSteps = 2, kChunk = 16, kActDim = 2;

    explicit FlowPolicy(const TensorFile& f, int euler_steps = 10);

    // obs_raw (B, 36) pixels = [frame t-1 (18), frame t (18)] -> (B, 36) in [-1, 1] with the training min/max
    Tensor normalize_obs(const Tensor& obs_raw) const;
    // Euler from t = 0 (noise) to t = 1 (data): a <- a + dt * v(a, obs_n, k * dt), k = 0..N-1, dt = 1/N.
    // noise (B, 32) is the initial a (taken as input so tests can use NumPy's noise). Returns normalised (B, 32).
    Tensor sample_normalized(const Tensor& obs_n, const Tensor& noise) const;
    // normalised chunk (B, 32) -> pixels (B, 16, 2)
    Tensor unnormalize_actions(const Tensor& chunk_n) const;
    // Everything: raw obs + noise -> action chunk in pixels (B, 16, 2). Execute the first 8.
    Tensor act(const Tensor& obs_raw, const Tensor& noise) const;

    const PolicyNet& net() const { return net_; }

private:
    PolicyNet net_;
    Tensor obs_min_, obs_max_, act_min_, act_max_;
    int euler_steps_;
};

}  // namespace fp
