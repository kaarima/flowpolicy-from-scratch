#include "flowpolicy/film.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace fp {

Tensor time_embedding(const Tensor& t, const Tensor& freqs) {
    Tensor out({t.dim(0), 2 * freqs.dim(0)});
    time_embedding_into(t, freqs, out);
    return out;
}

void time_embedding_into(const Tensor& t, const Tensor& freqs, Tensor& out) {
    if (t.ndim() != 1 || freqs.ndim() != 1) throw std::invalid_argument("time_embedding: t and freqs must be 1-D");
    const std::size_t B = t.dim(0), F = freqs.dim(0);
    if (out.size() != B * 2 * F) throw std::invalid_argument("time_embedding_into: bad out size");
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t j = 0; j < F; ++j) {
            const float a = t[i] * freqs[j];  // float32 product, as in NumPy
            out[i * 2 * F + j] = std::sin(a);
            out[i * 2 * F + F + j] = std::cos(a);
        }
}

Tensor concat_cols(const Tensor& A, const Tensor& B) {
    Tensor C({A.dim(0), A.dim(1) + B.dim(1)});
    concat_cols_into(A, B, C);
    return C;
}

void concat_cols_into(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.ndim() != 2 || B.ndim() != 2 || A.dim(0) != B.dim(0))
        throw std::invalid_argument("concat_cols: " + A.shape_str() + " | " + B.shape_str());
    const std::size_t M = A.dim(0), P = A.dim(1), Q = B.dim(1);
    if (C.size() != M * (P + Q)) throw std::invalid_argument("concat_cols_into: bad out size");
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < P; ++j) C[i * (P + Q) + j] = A[i * P + j];
        for (std::size_t j = 0; j < Q; ++j) C[i * (P + Q) + P + j] = B[i * Q + j];
    }
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
    Tensor out = h;
    Scratch s(h.dim(0), l1.out_features());
    forward_inplace(out, e, s, scale_out);
    return out;
}

void FilmBlock::forward_inplace(Tensor& h, const Tensor& e, Scratch& s, Tensor* scale_out) const {
    const std::size_t B = h.dim(0), H = l1.out_features();
    silu_into(h, s.a);
    l1.forward_into(s.a, s.u);  // u1
    lg.forward_into(e, s.g);    // (B, 2H): [scale | shift]
    if (scale_out && scale_out->size() != B * H) *scale_out = Tensor({B, H});
    for (std::size_t i = 0; i < B; ++i)
        for (std::size_t j = 0; j < H; ++j) {
            const float scale = s.g[i * 2 * H + j];
            const float shift = s.g[i * 2 * H + H + j];
            if (scale_out) (*scale_out)[i * H + j] = scale;
            s.u[i * H + j] = s.u[i * H + j] * (1.0f + scale) + shift;  // u2
        }
    silu_inplace(s.u);
    l2.forward_into(s.u, s.r);
    for (std::size_t i = 0; i < h.size(); ++i) h[i] += s.r[i];  // residual
}

}  // namespace fp
