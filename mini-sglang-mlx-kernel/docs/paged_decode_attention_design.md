# Paged Decode Attention Design

## 1. Goal

`paged_decode_attention` is the Metal decode kernel used by `mini-sglang-mlx` for the
single-query-token attention step during autoregressive generation.

The design targets Apple Silicon GPUs and is optimized for:

- paged KV cache with indirection through `kv_indices`
- GQA / MQA, where multiple Q heads share one KV head
- long-context decode via split-KV parallelism
- Apple GPU tensor core style execution through
  `mpp::tensor_ops::matmul2d`

The current implementation supports:

- dtypes: `float16`, `bfloat16`
- head dims: `128`, `256`
- `q_heads / kv_heads <= 16`


## 2. Inputs And Outputs

### Inputs

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| `q` | `(batch, num_q_heads, head_dim)` | f16/bf16 | one decode query per head |
| `k_cache` | `(num_pages, num_kv_heads, head_dim)` | f16/bf16 | paged key cache |
| `v_cache` | `(num_pages, num_kv_heads, head_dim)` | f16/bf16 | paged value cache |
| `kv_indptr` | `(batch + 1,)` | int32 | CSR pointers into `kv_indices` |
| `kv_indices` | `(total_kv_tokens,)` | int32 | page id for each KV token |
| `num_kv_splits` | `(batch,)` | int32 | per-request split count |
| `sm_scale` | scalar | float | attention scale |

### Output

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| `o` | `(batch, num_q_heads, head_dim)` | f16/bf16 | final decode attention output |

### Intermediate Buffers

| Tensor | Shape | Type | Meaning |
|---|---|---|---|
| `att_out` | `(batch, num_q_heads, max_kv_splits, head_dim)` | float32 | partial outputs per split |
| `att_lse` | `(batch, num_q_heads, max_kv_splits)` | float32 | log-sum-exp per split |


## 3. High-Level Structure

The kernel follows the standard flash-decoding pattern:

1. Split a long KV sequence into multiple chunks.
2. Compute partial attention independently for each chunk.
3. Reduce the chunk-local results into the final output.

This is implemented as two stages:

- `stage1`: per-split partial attention
- `stage2`: cross-split reduction


## 4. Stage 1 Overview

Each `stage1` threadgroup computes:

- one batch item
- one KV head
- one KV split

So the logical grid is:

```text
grid = (batch, num_kv_heads, max_kv_splits)
```

This is different from the older design that tiled over Q-head groups. The new
mapping is important for GQA:

- one threadgroup owns one KV head
- all Q heads that share that KV head are processed together
- the same streamed K/V tile is reused by all those Q heads

That avoids the old problem where `q_heads / kv_heads > 8` had to be split
across multiple threadgroups.


## 5. GQA Packing Strategy

Let:

```text
kv_group_num = num_q_heads / num_kv_heads
```

The implementation chooses the number of packed Q heads at runtime:

- if `kv_group_num <= 8`, dispatch `BLOCK_H = 8`
- else dispatch `BLOCK_H = 16`

This is a runtime dispatch done in C++, not a macro switch.

Why:

- Apple tensor ops work naturally with `M = 8/16`
- most practical GQA ratios are `<= 16`
- using one threadgroup per KV head preserves K/V reuse

Inactive lanes inside the chosen `BLOCK_H` are zero-padded and masked out.


## 6. Split-KV Scheduling

For each request:

- `num_kv_splits[b]` decides how many chunks the sequence is partitioned into
- each split gets a contiguous logical token range
- the range is rounded up to a multiple of `BLOCK_N = 32`

Within a split, the kernel loops over KV blocks:

```text
for block_start in [split_start, split_end) step BLOCK_N
```

This gives two levels of tiling:

- coarse split tiling across threadgroups
- fine `BLOCK_N=32` tiling inside each threadgroup


## 7. Tensor-Core Matmul Strategy

The old decode kernel used `simdgroup_multiply_accumulate` and loaded a full
`BLOCK_N x head_dim` K/V tile into threadgroup memory.

The current kernel uses Apple GPU tensor ops:

- `QK`: `Q[BLOCK_H, DK] @ K^T[DK, 32] -> S[BLOCK_H, 32]`
- `PV`: `P[BLOCK_H, 32] @ V[32, DV] -> O[BLOCK_H, DV]`

The matmul primitive is:

```metal
mpp::tensor_ops::matmul2d
```

with:

- `execution_simdgroups<4>`
- `K = 32` for streamed reduction tiles
- float accumulators


## 8. Streamed Sub-Tile Loading

This is the key design change.

### What stays resident

`Q` is loaded once per threadgroup into threadgroup memory because it is reused
across all KV blocks in the split.

### What is streamed

`K` and `V` are not stored as full `BLOCK_N x head_dim` tiles anymore.
Instead:

- for `QK`, a `BLOCK_H x 32` slice of `Q` and a `32 x 32` slice of `K` are loaded
- for `PV`, a `BLOCK_H x 32` probability tile and a `32 x 32` slice of `V` are loaded

The kernel iterates over the head dimension in chunks of 32:

```text
dk_base = 0, 32, 64, ...
dv_base = 0, 32, 64, ...
```

Benefits:

- much smaller threadgroup memory footprint
- better occupancy
- natural fit for tensor core tile shapes


## 9. KV Index Reuse

Each KV block first caches its 32 page indices into threadgroup memory:

```text
s_idx[0:BLOCK_N]
```

Then both phases reuse them:

- `QK` gathers K through `s_idx`
- `PV` gathers V through `s_idx`

This removes one round of repeated `kv_indices` reads from device memory for
every block.


## 10. Softmax Formulation

`stage1` uses online softmax inside each split.

For each packed head it tracks:

- `s_emax`: running max
- `s_esum`: running exp-sum
- `so`: running output accumulator

For each KV block:

1. compute scores `S`
2. apply scale
3. update `(emax, esum)` using online softmax
4. rescale the current output accumulator
5. accumulate `P @ V`

At the end of the split:

- `att_out` stores the normalized partial output
- `att_lse` stores `emax + log(esum)`


## 11. Stage 2 Reduction

`stage2` reduces all split-local outputs for one `(batch, q_head)`.

Logical grid:

```text
grid = (batch, num_q_heads, 1)
```

Each threadgroup is one SIMD group (`32` threads).

For each split:

1. load `att_lse`
2. combine split contributions with online log-sum-exp
3. accumulate the corresponding `att_out`

Finally:

```text
o = acc / e_sum
```

This stage is intentionally simple. In practice, `stage1` dominates the work,
so most optimization effort is concentrated there.


## 12. Threadgroup Memory Layout

`stage1` threadgroup memory contains:

- `sq`: resident Q block
- `sqk`: streamed Q sub-tile for `QK`
- `st`: streamed K/V sub-tile
- `s_idx`: cached page indices for the current KV block
- `ss`: score tile
- `so_tile`: temporary output tile for `PV`
- `so`: full output accumulator
- `s_emax`, `s_esum`: online softmax state
- `sp`: probability tile in half precision

This layout is chosen so the same temporary buffers can be reused across both
matmul phases.


## 13. Vectorized Loads

The current implementation also uses 2-lane vectorized loads/stores for the
hot contiguous paths:

- loading Q into `sq`
- copying Q into `sqk`
- gathering K and V sub-tiles
- moving float accumulators between `so` and `so_tile`
- writing `att_out`

This reduces scalar memory ops and helps both prefill and decode kernels on
Apple GPUs.


## 14. Runtime Dispatch In C++

`paged_decode_attention.cpp` does the runtime selection for `BLOCK_H`:

```text
kv_group_num <= 8  -> kernel with BLOCK_H = 8
kv_group_num > 8   -> kernel with BLOCK_H = 16
```

The same file also computes the threadgroup memory size based on the chosen
`BLOCK_H`.

The implementation currently rejects:

```text
q_heads / kv_heads > 16
```

because that case is outside the intended deployment range and is not tiled by
the current kernel.


## 15. Why This Design Works Well

This design tends to work especially well when:

- batch size is larger than 1
- KV lengths are long enough to benefit from split-KV parallelism
- multiple Q heads share one KV head

The main reasons are:

- split-KV exposes more threadgroup-level parallelism
- one streamed K/V tile is reused by all Q heads in the same GQA group
- tensor-core matmuls reduce math cost versus the old simdgroup path
- streamed sub-tiles keep threadgroup memory small enough to maintain good
  occupancy


## 16. Source Files

Relevant files:

```text
mini-sglang-mlx-kernel/
├── csrc/paged_decode_attention.h
├── csrc/paged_decode_attention.cpp
├── csrc/paged_decode_attention.metal
└── test_and_bench/paged_decode_attention.py
```
