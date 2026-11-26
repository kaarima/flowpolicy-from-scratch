"""Read/write the .fpbin tensor container used by the C++ port (layout in cpp/weights/FORMAT.md).

    magic   4 bytes  b"FPOL"
    version uint32   1
    count   uint32   number of tensors
    count x { name_len uint32, name (UTF-8, no NUL), ndim uint32, dims uint32[ndim] }
    data: every tensor in header order, float32, row-major (C order), contiguous, no padding
All integers and floats little-endian.
"""
import struct

import numpy as np

MAGIC, VERSION = b"FPOL", 1


def write(path, tensors):
    """tensors: list of (name, array). Arrays are converted to little-endian float32, C order."""
    header = [MAGIC, struct.pack("<II", VERSION, len(tensors))]
    data = []
    for name, a in tensors:
        a = np.ascontiguousarray(a, dtype="<f4")
        nb = name.encode()
        header.append(struct.pack("<I", len(nb)) + nb + struct.pack(f"<I{a.ndim}I", a.ndim, *a.shape))
        data.append(a.tobytes())
    with open(path, "wb") as f:
        f.write(b"".join(header) + b"".join(data))


def read(path):
    """Returns a dict name -> float32 array (in file order)."""
    buf = open(path, "rb").read()
    assert buf[:4] == MAGIC, "bad magic"
    version, count = struct.unpack_from("<II", buf, 4)
    assert version == VERSION, version
    off, specs = 12, []
    for _ in range(count):
        (n,) = struct.unpack_from("<I", buf, off)
        name = buf[off + 4:off + 4 + n].decode()
        off += 4 + n
        (ndim,) = struct.unpack_from("<I", buf, off)
        dims = struct.unpack_from(f"<{ndim}I", buf, off + 4)
        off += 4 + 4 * ndim
        specs.append((name, dims))
    out = {}
    for name, dims in specs:
        size = int(np.prod(dims)) if dims else 1
        out[name] = np.frombuffer(buf, dtype="<f4", count=size, offset=off).reshape(dims).copy()
        off += 4 * size
    assert off == len(buf), "trailing bytes"
    return out
