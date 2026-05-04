#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <assert.h>

namespace mx = mlx::core;

namespace mlx_serve {

// GatedDeltaNet recurrence for target-verify (MTP) phase.
//
// Each sequence in the batch processes `num_draft` tokens sequentially.
// The initial recurrent state is read from `slot_ids[b, 0]`.
// After processing token j, the updated state is written to
// `slot_ids[b, j]` so the caller can rollback to any accepted
// token position after verification.
//
// q, k:       [batch * num_draft, hk, dk] bfloat16
// v:          [batch * num_draft, hv, dv] bfloat16
// g, beta:    [batch * num_draft, hv]     bfloat16
// state:      [num_slots, hv, dv, dk]     float32, read+write
// slot_ids:   [batch, num_draft]          int32 — slot_ids[b,0] is the base state
// returns y:  [batch * num_draft, hv, dv] bfloat16
mx::array gdn_state_verify(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    mx::StreamOrDevice s = {});

class GDNStateVerify : public mx::Primitive {
 public:
  explicit GDNStateVerify(
      mx::Stream stream,
      int hk,
      int hv,
      int dk,
      int dv)
      : mx::Primitive(stream),
        hk_(hk),
        hv_(hv),
        dk_(dk),
        dv_(dv) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    assert(false);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "GDNStateVerify";
  }

 private:
  int hk_;
  int hv_;
  int dk_;
  int dv_;
};

}  // namespace mlx_serve
