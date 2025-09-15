import mlx.core as mx
import mlx.nn as nn
import time
from mlx_serve_kernel import varlen_rope

# --- 1. 数据生成函数 ---
def generate_rope_inputs(
    seq_len: int,
    n_heads: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    offset: int,
    dtype=mx.bfloat16,
):
    """
    生成用于 RoPE 算子基准测试的输入张量和缓存。

    参数:
        seq_len (int): 序列长度。
        n_heads (int): 注意力头数。
        rotary_dim (int): 旋转嵌入的维度。
        max_position_embeddings (int): 模型支持的最大位置嵌入。
        base (float): 旋转频率的基数。
        dtype (mlx.dtype): 张量的数据类型。
        offset (int): 序列起始位置

    返回:
        tuple: (hidden_states, position_ids, cos_cache, sin_cache, rotary_dim, base)
    """
    shape = [n_heads, seq_len, rotary_dim]
    hidden_states = mx.random.normal(shape, dtype=dtype)
    position_ids = mx.arange(seq_len, dtype=mx.int32)
    position_ids = position_ids + offset

    # 预热以确保 GPU 处于就绪状态
    mx.eval(hidden_states, position_ids)
    return hidden_states, position_ids


# --- 2. 核心函数 ---
def use_varlen_rope(
    x: mx.array,
    position_ids: mx.array,
    rotary_dim: int,
    base: float,
    offset: int,
) -> mx.array:
    """
    使用自定义的缓存 RoPE 实现。
    """
    return varlen_rope(
        x,
        position_ids,
        rotary_dim,
        base
    )

native_rope: nn.RoPE = None

def use_mlx_native(
    x: mx.array,
    position_ids: mx.array,
    rotary_dim: int,
    base: float,
    offset: int,
) -> mx.array:
    """
    使用 MLX 的原生 RoPE 实现。
    """
    return native_rope(x, offset=offset)



def run_functional_test(fused_func, naive_func, inputs):
    """
    验证两个函数的输出是否一致。

    参数:
        fused_func (callable): 缓存 RoPE 函数。
        naive_func (callable): MLX 原生 RoPE 函数。
        inputs (tuple): 函数所需的输入张量和参数。

    返回:
        bool: 如果输出近似相等则返回 True，否则返回 False。
    """
    a = fused_func(*inputs)
    b = naive_func(*inputs)
    return mx.allclose(a, b, rtol=1e-2, atol=1e-3)


# --- 3. 主程序 ---
if __name__ == "__main__":
    mx.random.seed(1025)

    # 定义要测试的不同参数组合
    test_configs = [
        # (seq_len, n_heads, rotary_dim, max_position_embeddings, base, offset)
        (1, 8, 128, 40960, 100000, 3),
        (1, 32, 128, 40960, 100000, 4),
        (2, 8, 128, 40960, 100000, 5),
        (2, 4, 128, 40960, 100000, 127),
        (13, 32, 128, 40960, 100000, 6),
        (32, 64, 128, 40960, 1000000, 7),
        (2048, 32, 128, 40960, 100000, 163),
        (4096, 16, 128, 40960, 1000000, 0),
    ]

    print("🚀 启动 RoPE 算子测试...\n")

    for seq_len, n_heads, rotary_dim, max_pos, base, offset in test_configs:
        print(
            f"--- 测试参数: seq_len={seq_len}, n_heads={n_heads}, rotary_dim={rotary_dim}, base={base}, offset={offset} ---"
        )
        (
            hidden_states,
            position_ids,
        ) = generate_rope_inputs(seq_len, n_heads, rotary_dim, max_pos, base, offset)
        
        # 将所有参数打包成一个元组
        inputs = (
            hidden_states,
            position_ids,
            rotary_dim,
            base,
            offset,
        )

        native_rope = nn.RoPE(dims=rotary_dim, traditional=False, base=base)

        # 运行功能测试
        is_correct = run_functional_test(use_varlen_rope, use_mlx_native, inputs)
        if is_correct:
            print("✅ 功能测试: 变长 RoPE 输出与 MLX 原生实现匹配。")
        else:
            print("❌ 功能测试: 输出不匹配！请检查变长 RoPE 内核。")