#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <assert.h>

namespace mx = mlx::core;

namespace mini_sglang_mlx {

// Fused GatedDeltaNet recurrence with slot-indexed in-place state update.
//
// Decode mode: omit qo_indptr and assume one token per request.
// Prefill mode: provide qo_indptr so each request can own a variable-length span.
//
// q, k:       (tokens, hk, dk) bfloat16
// v:          (tokens, hv, dv) bfloat16
// g, beta:    (tokens, hv) bfloat16
// state:      (num_slots, hv, dv, dk) float32, updated in-place at slot_ids[b]
// slot_ids:   (batch,) int32
// qo_indptr:  (batch + 1,) int32 cumulative token offsets, optional
// returns y:  (tokens, hv, dv) bfloat16
mx::array gdn_state_inplace(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    const mx::array& qo_indptr,
    bool single_token_mode = false,
    mx::StreamOrDevice s = {});

class GDNStateInplace : public mx::Primitive {
 public:
  explicit GDNStateInplace(
      mx::Stream stream,
      int hk,
      int hv,
      int dk,
      int dv,
      bool single_token_mode)
      : mx::Primitive(stream),
        hk_(hk),
        hv_(hv),
        dk_(dk),
        dv_(dv),
        single_token_mode_(single_token_mode) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    assert(false);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "GDNStateInplace";
  }

 private:
  int hk_;
  int hv_;
  int dk_;
  int dv_;
  bool single_token_mode_;
};

}  // namespace mini_sglang_mlx
