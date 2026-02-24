#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <assert.h>

namespace mx = mlx::core;

namespace mlx_serve {

mx::array paged_decode_attention(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& kv_indptr,
    const mx::array& kv_indices,
    const mx::array& num_kv_splits,
    float sm_scale,
    int max_kv_splits,
    mx::StreamOrDevice s = {});

class PagedDecodeAttention : public mx::Primitive {
 public:
  explicit PagedDecodeAttention(
      mx::Stream stream,
      float sm_scale,
      int max_kv_splits,
      int num_q_heads,
      int num_kv_heads,
      int head_dim)
      : mx::Primitive(stream),
        sm_scale_(sm_scale),
        max_kv_splits_(max_kv_splits),
        num_q_heads_(num_q_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    assert(false);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "PagedDecodeAttention";
  }

 private:
  float sm_scale_;
  int max_kv_splits_;
  int num_q_heads_;
  int num_kv_heads_;
  int head_dim_;
};

}  // namespace mlx_serve
