import jax
import jax.numpy as jnp
import numpy as np
from envs.mnist_env import MnistEnv
from utils.mnist_loader import load_mnist_data

def main():
    print("=== 1. 检查输入源 (MnistEnv) ===")
    
    # 1. 加载数据
    try:
        images, labels = load_mnist_data('train')
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 初始化环境 (模拟 train_mnist.sh 的参数)
    # 假设: 100Hz, 100ms (200 steps), dt=0.5
    dt_ms = 0.5
    snn_steps = 200 
    input_hz = 100.0 # 默认值
    
    env = MnistEnv(images, labels, presentation_steps=snn_steps, input_hz=input_hz, dt_ms=dt_ms)
    
    # 3. 生成一个样本
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    obs = state.obs # Shape: (Time, 196)
    
    # 4. 统计分析
    print(f"观测数据形状 (Time, Features): {obs.shape}")
    
    total_elements = obs.size
    total_spikes = jnp.sum(obs)
    sparsity = total_spikes / total_elements
    
    print(f"总像素点数 (Time * Pixels): {total_elements}")
    print(f"总脉冲数 (Spikes): {total_spikes}")
    print(f"全局平均发放率 (Sparsity): {sparsity:.4f} (每步每个像素发放的概率)")
    
    # 计算实际 Hz
    # Sparsity = Hz * dt / 1000
    # Hz = Sparsity * 1000 / dt
    actual_hz = sparsity * 1000 / dt_ms
    print(f"等效平均频率: {actual_hz:.2f} Hz (包含黑色背景)")
    
    # 检查非零像素的频率 (排除黑色背景)
    # 假设 MNIST 只有 ~20% 是笔画
    active_pixels_mask = jnp.sum(obs, axis=0) > 0
    active_spikes = jnp.sum(obs[:, active_pixels_mask])
    active_count = jnp.sum(active_pixels_mask) * snn_steps
    active_sparsity = active_spikes / active_count if active_count > 0 else 0
    print(f"笔画区域平均频率: {active_sparsity * 1000 / dt_ms:.2f} Hz")

    print("\n--- 诊断结论 ---")
    if sparsity < 0.001:
        print("❌ 信号极其微弱！网络几乎什么都看不到。")
        print("建议: 提高 MnistEnv 的 input_hz (例如到 200Hz 或 500Hz)")
    elif sparsity < 0.05:
        print("⚠️ 信号较稀疏。需要很大的 K_in 才能驱动网络。")
    else:
        print("✅ 信号强度适中。")

if __name__ == "__main__":
    main()