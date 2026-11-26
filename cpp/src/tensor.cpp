#include "flowpolicy/tensor.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fp {

namespace {
std::size_t numel(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}
}  // namespace

Tensor::Tensor(std::vector<std::size_t> shape, float fill) : shape_(std::move(shape)), data_(numel(shape_), fill) {}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    if (data_.size() != numel(shape_)) throw std::invalid_argument("Tensor: data size does not match shape");
}

std::string Tensor::shape_str() const {
    std::ostringstream os;
    os << "(";
    for (std::size_t i = 0; i < shape_.size(); ++i) os << (i ? ", " : "") << shape_[i];
    if (shape_.size() == 1) os << ",";
    os << ")";
    return os.str();
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) throw std::invalid_argument("matmul: inputs must be 2-D");
    Tensor C({A.dim(0), B.dim(1)});
    matmul(A, B, C);
    return C;
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.ndim() != 2 || B.ndim() != 2 || A.dim(1) != B.dim(0))
        throw std::invalid_argument("matmul: shapes " + A.shape_str() + " @ " + B.shape_str());
    const std::size_t M = A.dim(0), K = A.dim(1), N = B.dim(1);
    if (C.ndim() != 2 || C.dim(0) != M || C.dim(1) != N)
        throw std::invalid_argument("matmul: out shape " + C.shape_str());
    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();
    for (std::size_t i = 0; i < M * N; ++i) c[i] = 0.0f;
    for (std::size_t i = 0; i < M; ++i) {
        float* ci = c + i * N;
        for (std::size_t k = 0; k < K; ++k) {
            const float aik = a[i * K + k];
            const float* bk = b + k * N;
            for (std::size_t j = 0; j < N; ++j) ci[j] += aik * bk[j];
        }
    }
}

Linear Linear::from(const TensorFile& f, const std::string& w_name, const std::string& b_name) {
    Linear l{Tensor::from(f.get(w_name)), Tensor::from(f.get(b_name))};
    if (l.W.ndim() != 2 || l.b.ndim() != 1 || l.b.dim(0) != l.W.dim(1))
        throw std::invalid_argument("Linear: bad shapes " + l.W.shape_str() + ", " + l.b.shape_str());
    return l;
}

Tensor Linear::forward(const Tensor& x) const {
    Tensor y({x.dim(0), out_features()});
    forward_into(x, y);
    return y;
}

void Linear::forward_into(const Tensor& x, Tensor& y) const {
    matmul(x, W, y);
    const std::size_t rows = y.dim(0), n = y.dim(1);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < n; ++j) y[i * n + j] += b[j];
}

void silu_inplace(Tensor& x) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

Tensor silu(Tensor x) {
    silu_inplace(x);
    return x;
}

void silu_into(const Tensor& src, Tensor& dst) {
    if (src.size() != dst.size()) throw std::invalid_argument("silu_into: size mismatch");
    for (std::size_t i = 0; i < src.size(); ++i) dst[i] = src[i] / (1.0f + std::exp(-src[i]));
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("max_abs_diff: " + a.shape_str() + " vs " + b.shape_str());
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace fp
