#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <assert.h>

namespace mx = mlx::core;

namespace mlx_serve {

/// Paged prefill (extend) attention.
///
/// Q layout:  (total_q_tokens, num_q_heads, head_dim)
/// O layout:  (total_q_tokens, num_q_heads, head_dim)
/// K/V cache: (num_pages, num_kv_heads, head_dim)
///
/// qo_indptr:    (batch + 1,)  CSR pointers for Q/O tokens per sequence
/// kv_indptr:    (batch + 1,)  CSR pointers for unified KV (prefix + extend)
/// kv_indices:   (total_kv,)   page indices for all KV tokens
/// prefix_lens:  (batch,)      prefix length per sequence
///
/// Causal masking is always applied for the extend region.
/// Sliding window + attention sink are optional.
mx::array paged_prefill_attention(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& qo_indptr,
    const mx::array& kv_indptr,
    const mx::array& kv_indices,
    const mx::array& prefix_lens,
    float sm_scale,
    int max_len_extend,
    int window_size,
    mx::StreamOrDevice s = {});

class PagedPrefillAttention : public mx::Primitive {
 public:
  explicit PagedPrefillAttention(
      mx::Stream stream,
      float sm_scale,
      int max_len_extend,
      int window_size,
      int num_q_heads,
      int num_kv_heads,
      int head_dim,
      int total_q_tokens)
      : mx::Primitive(stream),
        sm_scale_(sm_scale),
        max_len_extend_(max_len_extend),
        window_size_(window_size),
        num_q_heads_(num_q_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        total_q_tokens_(total_q_tokens) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    assert(false);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "PagedPrefillAttention";
  }

 private:
  float sm_scale_;
  int max_len_extend_;
  int window_size_;
  int num_q_heads_;
  int num_kv_heads_;
  int head_dim_;
  int total_q_tokens_;
};

}  // namespace mlx_serve
