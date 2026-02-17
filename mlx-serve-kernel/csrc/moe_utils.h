#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <assert.h>
namespace mx = mlx::core;

namespace mlx_serve {

mx::array moe_sum_reduce(
    const mx::array& y,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

mx::array moe_sum_reduce_with_reorder(
    const mx::array& y,
    const mx::array& scores,
    const mx::array& inv_order,
    mx::StreamOrDevice s = {});

class MoeSumReduce : public mx::Primitive {
 public:
  explicit MoeSumReduce(mx::Stream stream)
      : mx::Primitive(stream){};

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override { assert(false);}
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  /** The name of primitive. */
  const char* name() const override {
    return "MoeSumReduce";
  }
};

class MoeSumReduceWithReorder : public mx::Primitive {
 public:
  explicit MoeSumReduceWithReorder(mx::Stream stream)
      : mx::Primitive(stream){};

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override { assert(false);}
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  /** The name of primitive. */
  const char* name() const override {
    return "MoeSumReduceWithReorder";
  }
};

mx::array moe_scatter_broadcast(
    const mx::array& x,
    const mx::array& inv_order,
    int topk_num,
    mx::StreamOrDevice s = {});

class MoeScatterBroadcast : public mx::Primitive {
 public:
  explicit MoeScatterBroadcast(mx::Stream stream, int topk_num)
      : mx::Primitive(stream), topk_num_(topk_num) {};

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override { assert(false);}
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "MoeScatterBroadcast";
  }

 private:
  int topk_num_;
};

} // namespace mlx_serve

