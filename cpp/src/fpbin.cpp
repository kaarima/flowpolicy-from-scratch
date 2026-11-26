#include "flowpolicy/fpbin.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace fp {

namespace {

bool host_is_little_endian() {
    const std::uint32_t one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

class Reader {
public:
    explicit Reader(std::vector<char> buf) : buf_(std::move(buf)) {}

    void need(std::size_t n) const {
        if (pos_ + n > buf_.size()) throw std::runtime_error("fpbin: truncated file");
    }
    std::uint32_t u32() {
        need(4);
        std::uint32_t v;
        std::memcpy(&v, buf_.data() + pos_, 4);
        pos_ += 4;
        return v;
    }
    std::string str(std::size_t n) {
        need(n);
        std::string s(buf_.data() + pos_, n);
        pos_ += n;
        return s;
    }
    void floats(float* out, std::size_t n) {
        need(4 * n);
        std::memcpy(out, buf_.data() + pos_, 4 * n);
        pos_ += 4 * n;
    }
    std::size_t pos() const { return pos_; }
    std::size_t size() const { return buf_.size(); }

private:
    std::vector<char> buf_;
    std::size_t pos_ = 0;
};

}  // namespace

std::string NamedTensor::shape_str() const {
    std::ostringstream os;
    os << "(";
    for (std::size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
    if (shape.size() == 1) os << ",";
    os << ")";
    return os.str();
}

TensorFile TensorFile::load(const std::string& path) {
    if (!host_is_little_endian()) throw std::runtime_error("fpbin: big-endian hosts are not supported");
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("fpbin: cannot open " + path);
    Reader r(std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));

    if (r.str(4) != "FPOL") throw std::runtime_error("fpbin: bad magic in " + path);
    const std::uint32_t version = r.u32();
    if (version != 1) throw std::runtime_error("fpbin: unsupported version " + std::to_string(version));
    const std::uint32_t count = r.u32();

    TensorFile tf;
    tf.tensors_.resize(count);
    for (auto& t : tf.tensors_) {
        t.name = r.str(r.u32());
        const std::uint32_t ndim = r.u32();
        std::size_t n = 1;
        for (std::uint32_t d = 0; d < ndim; ++d) {
            t.shape.push_back(r.u32());
            n *= t.shape.back();
        }
        t.data.resize(n);
    }
    for (auto& t : tf.tensors_) r.floats(t.data.data(), t.data.size());
    if (r.pos() != r.size()) throw std::runtime_error("fpbin: trailing bytes in " + path);
    return tf;
}

bool TensorFile::has(const std::string& name) const {
    for (const auto& t : tensors_)
        if (t.name == name) return true;
    return false;
}

const NamedTensor& TensorFile::get(const std::string& name) const {
    for (const auto& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("fpbin: no tensor named " + name);
}

std::uint64_t fnv1a64(const float* data, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p = reinterpret_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < 4 * n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

}  // namespace fp
