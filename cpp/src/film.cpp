#include "flowpolicy/film.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace fp {

Tensor time_embedding(const Tensor& t, const Tensor& freqs) {
    if (t.ndim() != 1 || freqs.ndim() != 1) throw std::invalid_argument("time_embedding: t and freqs must be 1-D");
    const std::size_t B = t.dim(0), F = freqs.dim(0);
    Tensor out({B, 2 * F});
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t j = 0; j < F; ++j) {
            const float a = t[i] * freqs[j];  // float32 product, as in NumPy
            out[i * 2 * F + j] = std::sin(a);
            out[i * 2 * F + F + j] = std::cos(a);
        }
    return out;
}

Tensor concat_cols(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2 || A.dim(0) != B.dim(0))
        throw std::invalid_argument("concat_cols: " + A.shape_str() + " | " + B.shape_str());
    const std::size_t M = A.dim(0), P = A.dim(1), Q = B.dim(1);
    Tensor C({M, P + Q});
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < P; ++j) C[i * (P + Q) + j] = A[i * P + j];
        for (std::size_t j = 0; j < Q; ++j) C[i * (P + Q) + P + j] = B[i * Q + j];
    }
    return C;
}

FilmBlock FilmBlock::from(const TensorFile& f, int k) {
    const std::string s = std::to_string(k);
    FilmBlock b{Linear::from(f, "W1_" + s, "b1_" + s), Linear::from(f, "Wg_" + s, "bg_" + s),
                Linear::from(f, "W2_" + s, "b2_" + s)};
    const std::size_t H = b.l1.out_features();
    if (b.lg.out_features() != 2 * H || b.l2.in_features() != H || b.l2.out_features() != H)
        throw std::invalid_argument("FilmBlock " + s + ": inconsistent shapes");
    return b;
}

Tensor FilmBlock::forward(const Tensor& h, const Tensor& e, Tensor* scale_out) const {
    const std::size_t B = h.dim(0), H = l1.out_features();
    Tensor u = l1.forward(silu(h));  // u1
    const Tensor g = lg.forward(e);  // (B, 2H): [scale | shift]
    if (scale_out) *scale_out = Tensor({B, H});
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t j = 0; j < H; ++j) {
            const float scale = g[i * 2 * H + j];
            const float shift = g[i * 2 * H + H + j];
            if (scale_out) (*scale_out)[i * H + j] = scale;
            u[i * H + j] = u[i * H + j] * (1.0f + scale) + shift;  // u2
        }
    silu_inplace(u);
    Tensor out = l2.forward(u);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += h[i];  // residual
    return out;
}

}  // namespace fp
