# `.bin` tensor container (FPOL v1)

Used for `cpp/weights/film_g.bin` (model) and `cpp/tests/vectors/vectors.bin` (test vectors).
Writer/reader in Python: `python/flowmatch/fpbin.py`. Reader in C++: `cpp/src/fpbin.cpp`.

## Layout (all little-endian)
| Offset | Type | Content |
|---|---|---|
| 0 | 4 bytes | magic `FPOL` |
| 4 | uint32 | version = 1 |
| 8 | uint32 | `count` = number of tensors |
| 12 | `count` entries | `name_len` uint32, `name` (UTF-8, no NUL), `ndim` uint32, `dims` uint32[`ndim`] |
| after header | data | every tensor in header order: float32, row-major (C order), contiguous, no padding |

The file ends exactly after the last tensor (the readers check this).

## Weight orientation: **(in, out)**
Every Linear is stored as `W` with shape `(in, out)` and a bias `b` with shape `(out,)`:

    y = x @ W + b          x: (B, in), y: (B, out)
    y[i][j] = b[j] + sum_k x[i][k] * W[k][j]      W[k][j] is at data[k * out + j]

This is the transpose of PyTorch's `nn.Linear` (which stores `(out, in)`).

## `film_g.bin`: tensors in file order (29)
Network (FiLM residual MLP, `python/flowmatch/film.py`), hidden H = 512, cond hidden C = 256, 3 blocks:

| Name | Shape | Role |
|---|---|---|
| `Wc`, `bc` | (52, 256), (256) | condition encoder: e = SiLU(cond @ Wc + bc), cond = [obs_n (36), temb (16)] |
| `Win`, `bin` | (32, 512), (512) | input projection: h = x @ Win + bin (no activation), x = noisy chunk (32) |
| `W1_k`, `b1_k` | (512, 512), (512) | block k: u1 = SiLU(h) @ W1_k + b1_k |
| `Wg_k`, `bg_k` | (256, 1024), (1024) | block k FiLM: g = e @ Wg_k + bg_k; scale = g[:512], shift = g[512:] |
| `W2_k`, `b2_k` | (512, 512), (512) | block k: h = h + SiLU(u1 * (1 + scale) + shift) @ W2_k + b2_k |
| `Wout`, `bout` | (512, 32), (32) | output: v = SiLU(h) @ Wout + bout |

(k = 0, 1, 2, in the order `W1_k b1_k Wg_k bg_k W2_k b2_k`.)

Normalisation and time embedding:

| Name | Shape | Role |
|---|---|---|
| `obs_min`, `obs_max` | (18), (18) | per-frame obs [16 keypoints x0,y0..x7,y7, agent x, y], pixels. obs_n = 2 (x - min) / (max - min) - 1, applied to each of the 2 frames |
| `act_min`, `act_max` | (2), (2) | action x, y in pixels. action_px = (a_n + 1) / 2 * (max - min) + min |
| `time_freqs` | (8) | temb = [sin(t * f_0..7), cos(t * f_0..7)] (16), all float32. f = geomspace(1, 100, 8) |

Sampler (`FlowPolicy.sample`): a = noise (32); for s in 0..9: a = a + 0.1 * v(a, obs_n, t = 0.1 s); reshape to
(16, 2), unnormalise, execute the first 8. Convention: t = 0 is noise, t = 1 is data.
SiLU(z) = z / (1 + exp(-z)). Readable config: `film_g.json`.

## `vectors.bin` (K = 4 test inputs, from `python/scripts/m3_export.py`)
| Name | Shape | What |
|---|---|---|
| `obs_raw` | (4, 36) | 2 frames x 18, pixels (4 real val windows) |
| `obs_n` | (4, 36) | normalised obs |
| `x_t` | (4, 32) | noisy action chunk input |
| `t` | (4) | 0, 0.3, 0.7, 0.95 |
| `temb` | (4, 16) | time embedding |
| `cond` | (4, 52) | [obs_n, temb] |
| `cond_e` | (4, 256) | e = SiLU(cond @ Wc + bc) |
| `h0` | (4, 512) | **first Linear**: x_t @ Win + bin |
| `film0_scale` | (4, 512) | block 0 FiLM scale |
| `block0_out`, `block1_out`, `block2_out` | (4, 512) | h after each block |
| `velocity` | (4, 32) | network output v(x_t, obs_n, t) |
| `noise0` | (4, 32) | fixed initial noise for the sampler |
| `sample_chunk_n` | (4, 16, 2) | 10 Euler steps from `noise0`, normalised |
| `sample_actions_px` | (4, 16, 2) | same, in pixels |
