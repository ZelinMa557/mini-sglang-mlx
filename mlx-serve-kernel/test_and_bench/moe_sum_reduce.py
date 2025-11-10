import mlx.core as mx
import time
from mlx_serve_kernel import moe_sum_reduce, moe_sum_reduce_with_reorder

# --- 1. 数据生成函数 ---
def generate_inputs(token_num, topk_num, hidden_dim, dtype=mx.bfloat16):
    """
    生成具有指定形状和数据类型的随机张量。
    
    参数:
        token_num (int): token数量
        topk_num (int): topk专家数量
        hidden_dim (int): 隐藏维度
        dtype (mlx.dtype): 张量的数据类型。
        
    返回:
        tuple: (y, scores)
    """
    # y shape: [token_num, topk_num, hidden_dim]
    y = mx.random.normal((token_num, topk_num, hidden_dim), dtype=dtype)
    # scores shape: [token_num, topk_num]
    scores = mx.random.uniform((token_num, topk_num), dtype=mx.float32)
    # Normalize scores to sum to 1 for each token
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    # 在评估前预热，以获得更准确的计时
    mx.eval(y, scores)
    return y, scores

def generate_inputs_with_reorder(reordered_token_num, token_num, topk_num, hidden_dim, dtype=mx.bfloat16):
    """
    生成包含重排序索引的输入。
    
    参数:
        reordered_token_num (int): 重排序后的token数量
        token_num (int): 原始token数量
        topk_num (int): topk专家数量
        hidden_dim (int): 隐藏维度
        dtype (mlx.dtype): 张量的数据类型。
        
    返回:
        tuple: (y, scores, inv_order)
    """
    # y shape: [reordered_token_num, topk_num, hidden_dim]
    y = mx.random.normal((reordered_token_num, topk_num, hidden_dim), dtype=dtype)
    # scores shape: [token_num, topk_num]
    scores = mx.random.uniform((token_num, topk_num), dtype=mx.float32)
    # Normalize scores to sum to 1 for each token
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    # inv_order: 对于输出位置i（原始token位置），应该从重排序后的y的哪个位置读取
    # 例如，如果原始顺序是 [0, 1, 2, 3]，重排序后y中存储的是 [token2, token0, token3, token1]
    # 那么 inv_order 应该是 [1, 3, 0, 2]，表示：
    #   - 输出位置0（原始token0）应该从重排序后的位置1读取
    #   - 输出位置1（原始token1）应该从重排序后的位置3读取
    #   - 输出位置2（原始token2）应该从重排序后的位置0读取
    #   - 输出位置3（原始token3）应该从重排序后的位置2读取
    # 生成一个随机排列
    inv_order = mx.arange(token_num, dtype=mx.int32)
    mx.random.shuffle(inv_order)
    # 在评估前预热
    mx.eval(y, scores, inv_order)
    return y, scores, inv_order

# --- 2. 核心函数 ---
def moe_sum_reduce_naive(y, scores):
    """
    原始的 MoE sum reduce 实现。
    y: [token_num, topk_num, hidden_dim]
    scores: [token_num, topk_num]
    返回: [token_num, hidden_dim]
    """
    return (y * scores[..., None]).sum(axis=-2)

def moe_sum_reduce_with_reorder_naive(y, scores, inv_order):
    """
    原始的 MoE sum reduce with reorder 实现。
    y: [reordered_token_num, topk_num, hidden_dim]
    scores: [token_num, topk_num]
    inv_order: [token_num]
    返回: [token_num, hidden_dim]
    """
    y_reordered = y[inv_order]
    return (y_reordered * scores[..., None]).sum(axis=-2)

def run_benchmark(func, *args, warmup_iters=5, test_iters=100):
    """
    对指定函数执行基准测试。
    
    参数:
        func (callable): 要测试的函数
        *args: 函数参数
        warmup_iters (int): 预热迭代次数
        test_iters (int): 基准测试迭代次数
        
    返回:
        float: 以毫秒为单位的平均执行时间
    """
    # 预热循环，确保 GPU 处于就绪状态
    for _ in range(warmup_iters):
        z = func(*args)
        mx.eval(z)

    start_time = time.time()
    for _ in range(test_iters):
        z = func(*args)
        mx.eval(z)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / test_iters
    return avg_time * 1000  # 转换为毫秒

def run_functional_test(fused_func, naive_func, *args):
    """
    验证两个函数的输出是否一致。
    
    参数:
        fused_func (callable): 融合函数
        naive_func (callable): 原始函数
        *args: 函数参数
        
    返回:
        bool: 如果输出近似相等则返回 True，否则返回 False
    """
    z_fused = fused_func(*args)
    z_naive = naive_func(*args)
    mx.eval(z_fused, z_naive)
    
    # 使用 mx.allclose 检查浮点数的近似相等性
    return mx.allclose(z_fused, z_naive, rtol=1e-4, atol=1e-5)

# --- 3. 主程序 ---
if __name__ == "__main__":
    mx.random.seed(1024)
    # 定义要测试的不同输入形状
    test_cases = [
        (32, 2, 4096),      # small batch, topk=2
        (128, 4, 4096),     # medium batch, topk=4
        (512, 8, 4096),     # large batch, topk=8
        (1024, 2, 8192),    # large batch, large hidden
        (256, 4, 2048),     # medium batch, medium hidden
    ]
    
    print("🚀 启动 moe_sum_reduce 内核测试...\n")
    
    # Test moe_sum_reduce (without reorder)
    print("=" * 60)
    print("测试 moe_sum_reduce (无重排序版本)")
    print("=" * 60)
    for token_num, topk_num, hidden_dim in test_cases:
        print(f"\n--- 测试配置: token_num={token_num}, topk_num={topk_num}, hidden_dim={hidden_dim} ---")
        y, scores = generate_inputs(token_num, topk_num, hidden_dim)
        
        # 运行功能测试
        is_correct = run_functional_test(
            moe_sum_reduce, moe_sum_reduce_naive, y, scores
        )
        if is_correct:
            print("✅ 功能测试: 融合内核输出与原始实现匹配。")
        else:
            print("❌ 功能测试: 输出不匹配！请检查融合内核。")
            # 打印一些调试信息
            z_fused = moe_sum_reduce(y, scores)
            z_naive = moe_sum_reduce_naive(y, scores)
            mx.eval(z_fused, z_naive)
            max_diff = mx.max(mx.abs(z_fused - z_naive))
            print(f"   最大差异: {max_diff.item()}")

        # 运行基准测试
        naive_time = run_benchmark(moe_sum_reduce_naive, y, scores)
        fused_time = run_benchmark(moe_sum_reduce, y, scores)
        
        speedup = naive_time / fused_time if fused_time > 0 else float('inf')
        
        print(f"⏱️  性能: Naive Time: {naive_time:.3f}ms | Fused Time: {fused_time:.3f}ms")
        print(f"🚀 加速比: {speedup:.2f}x")
    
    # Test moe_sum_reduce_with_reorder
    print("\n" + "=" * 60)
    print("测试 moe_sum_reduce_with_reorder (带重排序版本)")
    print("=" * 60)
    for token_num, topk_num, hidden_dim in test_cases:
        print(f"\n--- 测试配置: token_num={token_num}, topk_num={topk_num}, hidden_dim={hidden_dim} ---")
        # 对于重排序版本，reordered_token_num 应该等于 token_num
        reordered_token_num = token_num
        y, scores, inv_order = generate_inputs_with_reorder(
            reordered_token_num, token_num, topk_num, hidden_dim
        )
        
        # 运行功能测试
        is_correct = run_functional_test(
            moe_sum_reduce_with_reorder, moe_sum_reduce_with_reorder_naive,
            y, scores, inv_order
        )
        if is_correct:
            print("✅ 功能测试: 融合内核输出与原始实现匹配。")
        else:
            print("❌ 功能测试: 输出不匹配！请检查融合内核。")
            # 打印一些调试信息
            z_fused = moe_sum_reduce_with_reorder(y, scores, inv_order)
            z_naive = moe_sum_reduce_with_reorder_naive(y, scores, inv_order)
            mx.eval(z_fused, z_naive)
            max_diff = mx.max(mx.abs(z_fused - z_naive))
            print(f"   最大差异: {max_diff.item()}")

        # 运行基准测试
        naive_time = run_benchmark(moe_sum_reduce_with_reorder_naive, y, scores, inv_order)
        fused_time = run_benchmark(moe_sum_reduce_with_reorder, y, scores, inv_order)
        
        speedup = naive_time / fused_time if fused_time > 0 else float('inf')
        
        print(f"⏱️  性能: Naive Time: {naive_time:.3f}ms | Fused Time: {fused_time:.3f}ms")
        print(f"🚀 加速比: {speedup:.2f}x")

