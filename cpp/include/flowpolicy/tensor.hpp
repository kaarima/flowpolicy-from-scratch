// Minimal dense float32 tensor + the first layers of the FiLM policy: matmul, Linear, SiLU.
// Row-major (C order), same as NumPy and the .bin files. Weights are (in, out): y = x @ W + b.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "flowpolicy/fpbin.hpp"

namespace fp {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape, float fill = 0.0f);
    Tensor(std::vector<std::size_t> shape, std::vector<float> data);
    static Tensor from(const NamedTensor& t) { return Tensor(t.shape, t.data); }

    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t ndim() const { return shape_.size(); }
    std::size_t dim(std::size_t i) const { return shape_.at(i); }
    std::size_t size() const { return data_.size(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    float& operator[](std::size_t i) { return data_[i]; }
    float operator[](std::size_t i) const { return data_[i]; }

    std::string shape_str() const;

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

// C = A @ B. A: (M, K), B: (K, N) -> (M, N). Loop order i-k-j: the inner loop walks rows of B and C,
// which are contiguous in memory. Accumulates in float (like NumPy float32).
Tensor matmul(const Tensor& A, const Tensor& B);

// Fully connected layer, y = x @ W + b with W (in, out), b (out).
struct Linear {
    Tensor W, b;

    static Linear from(const TensorFile& f, const std::string& w_name, const std::string& b_name);
    std::size_t in_features() const { return W.dim(0); }
    std::size_t out_features() const { return W.dim(1); }
    Tensor forward(const Tensor& x) const;  // x: (B, in) -> (B, out)
};

// SiLU(z) = z / (1 + exp(-z)), element-wise.
void silu_inplace(Tensor& x);
Tensor silu(Tensor x);

// max |a - b| over all elements; shapes must match.
float max_abs_diff(const Tensor& a, const Tensor& b);

}  // namespace fp
