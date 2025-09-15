
#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <assert.h>
namespace mx = mlx::core;

namespace mlx_serve {

mx::array fused_add_rmsnorm(
    const mx::array& x,
    const mx::array& y,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice s = {});

class FusedAddRmsnorm : public mx::Primitive {
 public:
  explicit FusedAddRmsnorm(mx::Stream stream, float eps)
      : mx::Primitive(stream), eps_(eps){};

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override { assert(false);}
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  /** The name of primitive. */
  const char* name() const override {
    return "FusedAddRmsnorm";
  }

 private:
  float eps_;
};

} // namespace mlx_serve