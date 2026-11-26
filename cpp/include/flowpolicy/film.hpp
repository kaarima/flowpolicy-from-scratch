// Time embedding and FiLM residual block of the policy network (python/flowmatch/flow.py, film.py).
#pragma once

#include "flowpolicy/tensor.hpp"

namespace fp {

// Into a preallocated (B, 2F) tensor.
void time_embedding_into(const Tensor& t, const Tensor& freqs, Tensor& out);
// t: (B,), freqs: (F,) -> (B, 2F) = [sin(t * f_0..F-1), cos(t * f_0..F-1)]  (concatenated, not interleaved).
Tensor time_embedding(const Tensor& t, const Tensor& freqs);

// [A | B] along columns: A (M, P), B (M, Q) -> (M, P + Q).
Tensor concat_cols(const Tensor& A, const Tensor& B);
void concat_cols_into(const Tensor& A, const Tensor& B, Tensor& C);

// One FiLM residual block (hidden H, condition C):
//   u1 = SiLU(h) @ W1 + b1                 (pre-activation: SiLU before W1)
//   scale, shift = split(e @ Wg + bg)      (first H columns, last H columns)
//   u2 = u1 * (1 + scale) + shift
//   h' = h + SiLU(u2) @ W2 + b2
struct FilmBlock {
    Linear l1, lg, l2;

    static FilmBlock from(const TensorFile& f, int k);  // tensors W1_k b1_k Wg_k bg_k W2_k b2_k
    // h: (B, H), e: (B, C) -> (B, H). If scale_out is given, it receives the (B, H) FiLM scale.
    Tensor forward(const Tensor& h, const Tensor& e, Tensor* scale_out = nullptr) const;

    // Scratch buffers for forward_inplace, all (B, H) except g (B, 2H). Allocate once, reuse every call.
    struct Scratch {
        Tensor a, u, g, r;
        Scratch(std::size_t B, std::size_t H) : a({B, H}), u({B, H}), g({B, 2 * H}), r({B, H}) {}
    };
    // h <- block(h, e) without allocating. Same arithmetic, same order as forward().
    void forward_inplace(Tensor& h, const Tensor& e, Scratch& s, Tensor* scale_out = nullptr) const;
};

}  // namespace fp
