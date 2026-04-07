#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <assert.h>

namespace mx = mlx::core;

namespace mlx_serve {

// Fused single-step GatedDeltaNet recurrence.
//
// q, k:      (batch, hk, dk)
// v:         (batch, hv, dv)
// g, beta:   (batch, hv)
// state:     (num_slots, hv, dv, dk), updated in-place at slot_ids[b]
// slot_ids:  (batch,) int32
// returns y: (batch, hv, dv)
mx::array gdn_decode_inplace(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    mx::StreamOrDevice s = {});

// Fused variable-length prefill GatedDeltaNet recurrence.
//
// q, k:       (total_tokens, hk, dk)
// v:          (total_tokens, hv, dv)
// g, beta:    (total_tokens, hv)
// state:      (num_slots, hv, dv, dk), updated in-place at slot_ids[b]
// slot_ids:   (batch,) int32
// qo_indptr:  (batch + 1,) int32 cumulative token offsets
// returns y:  (total_tokens, hv, dv)
mx::array gdn_prefill_inplace(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const mx::array& slot_ids,
    const mx::array& qo_indptr,
    mx::StreamOrDevice s = {});

class GDNDecodeInplace : public mx::Primitive {
 public:
  explicit GDNDecodeInplace(
      mx::Stream stream,
      int hk,
      int hv,
      int dk,
      int dv)
      : mx::Primitive(stream), hk_(hk), hv_(hv), dk_(dk), dv_(dv) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    assert(false);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "GDNDecodeInplace";
  }

 private:
  int hk_;
  int hv_;
  int dk_;
  int dv_;
};

class GDNPrefillInplace : public mx::Primitive {
 public:
  explicit GDNPrefillInplace(
      mx::Stream stream,
      int hk,
      int hv,
      int dk,
      int dv)
      : mx::Primitive(stream), hk_(hk), hv_(hv), dk_(dk), dv_(dv) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    assert(false);
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "GDNPrefillInplace";
  }

 private:
  int hk_;
  int hv_;
  int dk_;
  int dv_;
};

}  // namespace mlx_serve
