// Reader for the FPOL v1 tensor container (layout: cpp/weights/FORMAT.md).
// Plumbing only: it returns raw float buffers with names and shapes. Your Tensor type can wrap these.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fp {

struct NamedTensor {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> data;  // row-major (C order)

    std::size_t size() const { return data.size(); }
    std::string shape_str() const;  // e.g. "(52, 256)"
};

class TensorFile {
public:
    // Throws std::runtime_error on a bad magic/version, truncated or oversized file, or non-little-endian host.
    static TensorFile load(const std::string& path);

    const std::vector<NamedTensor>& tensors() const { return tensors_; }
    bool has(const std::string& name) const;
    const NamedTensor& get(const std::string& name) const;  // throws if missing

private:
    std::vector<NamedTensor> tensors_;
};

// 64-bit FNV-1a hash of the raw float bytes; used to check bitwise equality against NumPy.
std::uint64_t fnv1a64(const float* data, std::size_t n);

}  // namespace fp
