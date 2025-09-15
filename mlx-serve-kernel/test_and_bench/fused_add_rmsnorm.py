import mlx.core as mx
import time
from mlx_serve_kernel import fused_add_rmsnorm

# --- 1. 数据生成函数 ---
def generate_inputs(shape, dtype=mx.bfloat16):
    """
    生成具有指定形状和数据类型的随机张量。
    
    参数:
        shape (tuple): 张量的形状。
        dtype (mlx.dtype): 张量的数据类型。
        
    返回:
        tuple: (a, b, weights)
    """
    # 确保张量在 GPU 上
    a = mx.random.normal(shape, dtype=dtype)
    b = mx.random.normal(shape, dtype=dtype)
    # 权重张量为 (shape[-1],)，与最后一个维度匹配
    weights = mx.ones((shape[-1],), dtype=dtype)
    # 在评估前预热，以获得更准确的计时
    mx.eval(a, b, weights)
    return a, b, weights

# --- 2. 核心函数 ---
def add_rmsnorm_naive(x, y, weights, eps=1e-6):
    """
    原始的 RMSNorm 实现。
    """
    return mx.fast.rms_norm(x + y, weight=weights, eps=eps)

def run_benchmark(func, a, b, weights, eps=1e-6, warmup_iters=5, test_iters=100):
    """
    对指定函数执行基准测试。
    
    参数:
        func (callable): 要测试的函数 (例如: fused_add_rmsnorm)。
        a, b, weights (mx.array): 输入张量。
        eps (float): epsilon 值。
        warmup_iters (int): 预热迭代次数。
        test_iters (int): 基准测试迭代次数。
        
    返回:
        float: 以毫秒为单位的平均执行时间。
    """
    # 预热循环，确保 GPU 处于就绪状态
    for _ in range(warmup_iters):
        z = func(a, b, weights, eps)
        mx.eval(z)

    start_time = time.time()
    for _ in range(test_iters):
        z = func(a, b, weights, eps)
        mx.eval(z)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / test_iters
    return avg_time * 1000  # 转换为毫秒

def run_functional_test(fused_func, naive_func, a, b, weights, eps=1e-6):
    """
    验证两个函数的输出是否一致。
    
    参数:
        fused_func (callable): 融合函数。
        naive_func (callable): 原始函数。
        a, b, weights (mx.array): 输入张量。
        eps (float): epsilon 值。
        
    返回:
        bool: 如果输出近似相等则返回 True，否则返回 False。
    """
    z_fused = fused_func(a, b, weights, eps)
    z_naive = naive_func(a, b, weights, eps)
    mx.eval(z_fused, z_naive)
    
    # 使用 mx.allclose 检查浮点数的近似相等性
    # atol 和 rtol 是容忍度参数
    return mx.allclose(z_fused, z_naive, rtol=1e-5, atol=1e-6)

# --- 3. 主程序 ---
if __name__ == "__main__":
    mx.random.seed(1024)
    # 定义要测试的不同输入形状
    test_shapes = [
        (1, 4096),
        (1566, 4096),
        (8192, 4096),
        (1, 8192),
        (8192, 8192),
        (4, 128, 32),
        (1566, 128, 32),
        (1566, 128, 64),
        (4096, 128, 64),
    ]
    
    eps_val = 1e-6
    
    print("🚀 启动 add_rmsnorm 内核测试...\n")
    
    for shape in test_shapes:
        print(f"--- 测试张量形状: {shape} ---")
        a, b, weights = generate_inputs(shape)
        
        # 运行功能测试
        is_correct = run_functional_test(fused_add_rmsnorm, add_rmsnorm_naive, a, b, weights, eps_val)
        if is_correct:
            print("✅ 功能测试: 融合内核输出与原始实现匹配。")
        else:
            print("❌ 功能测试: 输出不匹配！请检查融合内核。")

        # 运行基准测试
        naive_time = run_benchmark(add_rmsnorm_naive, a, b, weights, eps_val)
        fused_time = run_benchmark(fused_add_rmsnorm, a, b, weights, eps_val)
        
        speedup = naive_time / fused_time if fused_time > 0 else float('inf')
        
        print(f"⏱️ 性能: Naive Time: {naive_time:.3f}ms | Fused Time: {fused_time:.3f}ms")
        print(f"🚀 加速比: {speedup:.2f}x\n")