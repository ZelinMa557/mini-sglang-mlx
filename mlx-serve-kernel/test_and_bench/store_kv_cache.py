import numpy as np
import mlx.core as mx
from mlx_serve_kernel import store_kv_cache


# --- 1. 数据生成函数 ---
def generate_store_kv_inputs(
    length: int,
    num_pages: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=mx.float16,
):
    """
    生成用于 store_kv_cache 测试的输入张量。

    参数:
        length (int): 要写入的序列长度 L，须满足 length <= num_pages。
        num_pages (int): KV cache 的页数。
        num_kv_heads (int): KV 头数。
        head_dim (int): 头维度，须为 4 的倍数。
        dtype (mlx.dtype): 张量的数据类型。

    out_loc 为 [0, num_pages) 的随机排列的前 length 个，保证无重复。

    返回:
        tuple: (k, v, k_cache_empty, v_cache_empty, out_loc)
    """
    assert head_dim % 4 == 0, "head_dim must be a multiple of 4"
    assert length <= num_pages, "length must be <= num_pages for unique out_loc"
    k = mx.random.normal((length, num_kv_heads, head_dim), dtype=dtype)
    v = mx.random.normal((length, num_kv_heads, head_dim), dtype=dtype)
    k_cache_empty = mx.zeros((num_pages, num_kv_heads, head_dim), dtype=dtype)
    v_cache_empty = mx.zeros((num_pages, num_kv_heads, head_dim), dtype=dtype)
    # out_loc: 无重复的页索引，对 [0, num_pages) 做 shuffle 后取前 length 个
    perm = np.random.permutation(num_pages)
    out_loc = mx.array(perm[:length].astype(np.int32))
    mx.eval(k, v, k_cache_empty, v_cache_empty, out_loc)
    return k, v, k_cache_empty, v_cache_empty, out_loc


# --- 2. 核心函数 ---
def store_kv_cache_naive(
    k_cache: mx.array,
    v_cache: mx.array,
    indices: mx.array,
    k: mx.array,
    v: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Naive 实现：按位置逐行 scatter，k_cache[indices[r]] = k[r], v_cache[indices[r]] = v[r]。
    """
    length = indices.size
    for r in range(length):
        pos = int(indices[r].item())
        k_cache[pos] = k[r]
        v_cache[pos] = v[r]
    return k_cache, v_cache


def run_functional_test(
    length: int,
    num_pages: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=mx.float16,
):
    """
    验证 store_kv_cache 与 naive 实现的输出是否一致。

    参数:
        length, num_pages, num_kv_heads, head_dim (int): 形状参数。
        dtype (mlx.dtype): 数据类型。

    返回:
        bool: 若 k_cache / v_cache 结果一致则返回 True，否则返回 False。
    """
    k, v, k_cache_empty, v_cache_empty, out_loc = generate_store_kv_inputs(
        length, num_pages, num_kv_heads, head_dim, dtype
    )

    # Kernel 路径：拷贝空 cache 后调用 store_kv_cache（in-place）
    k_cache_kernel = mx.zeros_like(k_cache_empty)
    v_cache_kernel = mx.zeros_like(v_cache_empty)
    mx.eval(k_cache_kernel, v_cache_kernel)
    store_kv_cache(
        k_cache=k_cache_kernel,
        v_cache=v_cache_kernel,
        indices=out_loc,
        k=k,
        v=v,
    )

    # Naive 路径：拷贝空 cache 后逐行 scatter
    k_cache_naive = mx.zeros_like(k_cache_empty)
    v_cache_naive = mx.zeros_like(v_cache_empty)
    print(v_cache_naive)
    k_cache_naive, v_cache_naive = store_kv_cache_naive(
        k_cache_naive, v_cache_naive, out_loc, k, v
    )
    mx.eval(k_cache_naive, v_cache_naive)
    print(v_cache_naive)

    k_ok = mx.allclose(k_cache_kernel, k_cache_naive, rtol=1e-5, atol=1e-5)
    v_ok = mx.allclose(v_cache_kernel, v_cache_naive, rtol=1e-5, atol=1e-5)
    return bool(k_ok.item()) and bool(v_ok.item())


# --- 3. 主程序 ---
if __name__ == "__main__":
    mx.random.seed(1025)

    # head_dim 须为 4 的倍数
    test_configs = [
        # (length, num_pages, num_kv_heads, head_dim)
        (1, 128, 4, 128),
        (3, 128, 4, 128),
        (32, 128, 4, 128),
        (67, 256, 8, 128),
        (129, 512, 16, 128),
    ]

    print("🚀 启动 store_kv_cache 算子测试...\n")

    for length, num_pages, num_kv_heads, head_dim in test_configs:
        print(
            f"--- 测试参数: length={length}, num_pages={num_pages}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim} ---"
        )
        is_correct = run_functional_test(
            length, num_pages, num_kv_heads, head_dim
        )
        if is_correct:
            print("✅ 功能测试: store_kv_cache 与 naive 实现结果一致。")
        else:
            print("❌ 功能测试: 结果不一致，请检查 store_kv_cache 内核。")
