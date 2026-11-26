#include "flowpolicy/policy.hpp"

#include <stdexcept>
#include <string>

namespace fp {

Workspace::Workspace(const PolicyNet& net, std::size_t batch)
    : B(batch),
      t({batch}),
      temb({batch, 2 * net.time_freqs.size()}),
      cond({batch, net.cond_enc.in_features()}),
      e({batch, net.cond_enc.out_features()}),
      h({batch, net.in_proj.out_features()}),
      hs({batch, net.in_proj.out_features()}),
      v({batch, net.out.out_features()}),
      a({batch, net.in_proj.in_features()}),
      film(batch, net.in_proj.out_features()) {}

PolicyNet PolicyNet::from(const TensorFile& f) {
    PolicyNet n{Linear::from(f, "Wc", "bc"), Linear::from(f, "Win", "bin"), Linear::from(f, "Wout", "bout"), {},
                Tensor::from(f.get("time_freqs"))};
    for (int k = 0; f.has("W1_" + std::to_string(k)); ++k) n.blocks.push_back(FilmBlock::from(f, k));
    if (n.blocks.empty()) throw std::invalid_argument("PolicyNet: no FiLM blocks in file");
    return n;
}

Tensor PolicyNet::velocity(const Tensor& x_t, const Tensor& obs_n, const Tensor& t) const {
    Workspace ws(*this, x_t.dim(0));
    ws.t = t;
    velocity_into(x_t, obs_n, ws);
    return ws.v;
}

void PolicyNet::velocity_into(const Tensor& x_t, const Tensor& obs_n, Workspace& ws) const {
    time_embedding_into(ws.t, time_freqs, ws.temb);
    concat_cols_into(obs_n, ws.temb, ws.cond);
    cond_enc.forward_into(ws.cond, ws.e);
    silu_inplace(ws.e);
    in_proj.forward_into(x_t, ws.h);
    for (const auto& b : blocks) b.forward_inplace(ws.h, ws.e, ws.film);
    silu_into(ws.h, ws.hs);
    out.forward_into(ws.hs, ws.v);
}

FlowPolicy::FlowPolicy(const TensorFile& f, int euler_steps)
    : net_(PolicyNet::from(f)),
      obs_min_(Tensor::from(f.get("obs_min"))),
      obs_max_(Tensor::from(f.get("obs_max"))),
      act_min_(Tensor::from(f.get("act_min"))),
      act_max_(Tensor::from(f.get("act_max"))),
      euler_steps_(euler_steps) {
    if (obs_min_.size() != kFrameDim || act_min_.size() != kActDim) throw std::invalid_argument("FlowPolicy: bad stats");
}

Tensor FlowPolicy::normalize_obs(const Tensor& obs_raw) const {
    const std::size_t B = obs_raw.dim(0), D = kFrameDim * kObsSteps;
    if (obs_raw.ndim() != 2 || obs_raw.dim(1) != D) throw std::invalid_argument("normalize_obs: " + obs_raw.shape_str());
    Tensor y({B, D});
    for (std::size_t i = 0; i < B * D; ++i) {
        const std::size_t j = i % kFrameDim;  // same stats for both frames
        const float range = obs_max_[j] - obs_min_[j];
        y[i] = 2.0f * (obs_raw[i] - obs_min_[j]) / range - 1.0f;
    }
    return y;
}

Tensor FlowPolicy::sample_normalized(const Tensor& obs_n, const Tensor& noise) const {
    Workspace ws(net_, obs_n.dim(0));
    ws.a = noise;
    sample_normalized_inplace(obs_n, ws);
    return ws.a;
}

void FlowPolicy::sample_normalized_inplace(const Tensor& obs_n, Workspace& ws) const {
    const float dt = 1.0f / static_cast<float>(euler_steps_);
    for (int k = 0; k < euler_steps_; ++k) {
        const float tk = static_cast<float>(k * (1.0 / euler_steps_));  // same t grid as NumPy (k * dt)
        for (std::size_t i = 0; i < ws.B; ++i) ws.t[i] = tk;
        net_.velocity_into(ws.a, obs_n, ws);
        for (std::size_t i = 0; i < ws.a.size(); ++i) ws.a[i] += dt * ws.v[i];
    }
}

Tensor FlowPolicy::unnormalize_actions(const Tensor& chunk_n) const {
    const std::size_t B = chunk_n.dim(0);
    Tensor px({B, kChunk, kActDim});
    for (std::size_t i = 0; i < px.size(); ++i) {
        const std::size_t j = i % kActDim;
        px[i] = (chunk_n[i] + 1.0f) * 0.5f * (act_max_[j] - act_min_[j]) + act_min_[j];
    }
    return px;
}

Tensor FlowPolicy::act(const Tensor& obs_raw, const Tensor& noise) const {
    return unnormalize_actions(sample_normalized(normalize_obs(obs_raw), noise));
}

}  // namespace fp
