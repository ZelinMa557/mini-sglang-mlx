# Paged Decode Attention Kernel Design

## 1. Problem Statement

Implement a high-performance **paged decode attention** kernel targeting Apple Silicon GPUs via Metal, for use in the mlx-serve LLM inference engine.

**Key requirements:**
- **Paged KV cache** (page_size = 1): KV tokens are scattered in memory, accessed via indirection
- **GQA-centric**: Multiple Q heads share one KV head (e.g., 4:1 or 8:1). MHA is treated as GQA with group_num=1
- **Flash Decoding**: Split long KV sequences across multiple threadgroups for parallelism
- **Query head packing**: Pack multiple Q heads into one threadgroup for KV reuse + simdgroup matmul
- Data types: `float16` and `bfloat16`
- Head dimensions: 128

## 2. Data Layout & KV Cache Format

### Inputs

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `q` | `(batch, num_q_heads, head_dim)` | f16/bf16 | Query vectors (seq_len=1) |
| `k_cache` | `(num_pages, num_kv_heads, head_dim)` | f16/bf16 | Paged key cache |
| `v_cache` | `(num_pages, num_kv_heads, head_dim)` | f16/bf16 | Paged value cache |
| `kv_indptr` | `(batch + 1,)` | int32 | CSR row pointers into `kv_indices` |
| `kv_indices` | `(total_kv_tokens,)` | int32 | Page indices for each KV token |
| `num_kv_splits` | `(batch,)` | int32 | Per-request number of KV splits |
| `sm_scale` | scalar | float | Typically `1/sqrt(head_dim)` |

### Outputs

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `o` | `(batch, num_q_heads, head_dim)` | f16/bf16 | Attention output |

### Intermediate Buffers

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `att_out` | `(batch, num_q_heads, max_kv_splits, head_dim)` | float32 | Partial outputs |
| `att_lse` | `(batch, num_q_heads, max_kv_splits)` | float32 | Log-sum-exp per split |

## 3. Algorithm Overview

### Stage 1: Partial Attention (per KV split)

Each threadgroup processes one batch item, `BLOCK_H` query heads (same KV head), one KV chunk.

1. Determine KV range `[split_start, split_end)` for this split
2. Load `BLOCK_H` query vectors into shared memory
3. For each block of `BLOCK_N` KV tokens:
   - **Sliding window**: skip tokens outside `[kv_len - window_size, kv_len)` (but always include token 0 if attention sink is enabled)
   - **Gather K** via page indices → shared memory
   - **QK^T**: `Q[BLOCK_H, DK] @ K^T[DK, BLOCK_N] → S[BLOCK_H, BLOCK_N]` using simdgroup matmul
   - **Scale**: `S *= sm_scale`
   - **Mask**: set out-of-range positions to `-inf` (for sliding window + sink)
   - **Online softmax**: update `e_max`, `e_sum`, rescale `acc`
   - **Gather V** → shared memory
   - **Accumulate**: `acc += P @ V` using simdgroup matmul
4. Store `att_out` and `att_lse`

### Stage 2: Cross-Split Reduction

Each threadgroup reduces one `(batch, head)` across all KV splits:

1. Load partial outputs and LSE from each split
2. Online softmax combination across splits
3. **Attention sink**: if `sink_ptr` is provided, add `exp(sink_value - e_max)` to `e_sum` before final division
4. Write final output `o = acc / e_sum`

### Sliding Window + Attention Sink

When `window_size > 0`:
- Only attend to KV positions in `[max(1, kv_len - window_size), kv_len)` plus position 0 (sink)
- The sink token (position 0) is always attended to regardless of the window
- Positions outside the window (except sink) get mask value `-inf`

When `window_size < 0` (disabled):
- Attend to all positions, no masking needed

## 4. GQA Query Head Packing

Pack `BLOCK_H=8` query heads sharing the same KV head into one threadgroup:
- KV data loaded once, reused for all heads → bandwidth savings
- QK^T becomes `(8, DK) @ (DK, BLOCK_N)` → leverages simdgroup 8x8 matmul
- For `kv_group_num < 8`: pad to 8, mask invalid heads
- For `kv_group_num > 8`: multiple threadgroup tiles along head dimension

### Grid Dimensions (Stage 1)

```
grid = (batch, ceil(num_q_heads / BLOCK_H), max_kv_splits)
```

## 5. Metal Implementation Details

### Template Parameters

```metal
template <typename T, short DK, short DV, short BLOCK_H, short BLOCK_N, short NSG>
```

- `T`: `half` or `bfloat16_t`
- `DK`, `DV`: head key/value dimensions (64, 128, or 512)
- `BLOCK_H`: query heads per threadgroup (8)
- `BLOCK_N`: KV tokens per inner loop (32)
- `NSG`: SIMD groups per threadgroup (4)

### simdgroup Matrix Multiply

Using `simdgroup_multiply_accumulate` for 8x8 half-precision matmul on both QK^T and S@V.

### Thread Organization

- 4 SIMD groups × 32 threads = 128 threads per threadgroup
- All SIMD groups cooperate on loading Q/K/V
- SIMD groups split output tile columns for matmul

## 6. C++ / Python Interface

```cpp
mx::array paged_decode_attention(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& kv_indptr,
    const mx::array& kv_indices,
    const mx::array& num_kv_splits,
    float sm_scale,
    int max_kv_splits,
    int window_size,       // -1 = no sliding window
    mx::StreamOrDevice s = {});
```

## 7. Template Instantiation

```
Types: {float16, bfloat16}
Head dims (DK=DV): {64, 128, 512}
BLOCK_H: {8}
BLOCK_N: {32}
NSG: {4}
```

## 8. File Structure

```
csrc/
├── paged_decode_attention.h
├── paged_decode_attention.cpp
├── paged_decode_attention.metal
└── bindings.cpp  (updated)
```
