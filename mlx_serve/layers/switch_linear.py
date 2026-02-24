# Adapted from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py
import math
import mlx.core as mx
import mlx.nn as nn

from mlx_serve.layers.activations import swiglu
from mlx_serve_kernel import moe_scatter_broadcast, moe_sum_reduce, moe_sum_reduce_with_reorder


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


def _gather_sort(x, indices):
    """Sort tokens by expert index for efficient gather_mm.

    Args:
        x: [L, 1, 1, D] expanded token features
        indices: [L, K] expert indices per token

    Returns:
        x_sorted: [L*K, 1, D] tokens reordered by expert
        idx_sorted: [L*K] sorted expert indices
        inv_order: [L*K] inverse permutation (uint32, for fused kernels)
    """
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x_flat = x.flatten(0, -3)  # [L, 1, D]
    x_sorted = moe_scatter_broadcast(
        x_flat.squeeze(-2),    # [L, D]
        inv_order.astype(mx.uint32),
        M
    )                          # [L*K, D]
    x_sorted = mx.expand_dims(x_sorted, -2)  # [L*K, 1, D]
    return x_sorted, indices[order], inv_order


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices, scores) -> mx.array:
        """
        Args:
            x: [L, D] token features
            indices: [L, K] expert indices per token
            scores: [L, K] expert weights per token

        Returns:
            out: [L, D] weighted expert outputs
        """
        x = mx.expand_dims(x, (-2, -3))  # [L, 1, 1, D]

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        # x: [L*K, 1, D] (sorted) or [L, K, 1, D] (unsorted)
        x = x.squeeze(-2)  # [L*K, D] or [L, K, D]

        if do_sort:
            # Fused unsort + weighted sum via custom kernel
            # x is [L*K, D], scores is [L, K], inv_order is [L*K]
            return moe_sum_reduce_with_reorder(
                x, scores, inv_order.astype(mx.uint32)
            )
        else:
            # No sorting was done, x is [L, K, D], scores is [L, K]
            return moe_sum_reduce(
                x.reshape(-1, x.shape[-1]), scores
            )
