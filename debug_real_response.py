import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from networks.conn_snn import ConnSNN
from envs.mnist_env import MnistEnv
from utils.mnist_loader import load_mnist_data

def main(args):
    print(f"--- 真实响应测试: K_in={args.K_in}, K_h={args.K_h} ---")
    
    # 1. 准备数据和环境
    images, labels = load_mnist_data('train')
    dt_ms = 0.5
    snn_steps = 200
    env = MnistEnv(images, labels, presentation_steps=snn_steps, input_hz=100.0, dt_ms=dt_ms)
    
    # 生成真实输入
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    real_input = state.obs # (200, 196)
    
    # 2. 准备网络
    try:
        physics_data = np.load('neuron_physics.npz')
        num_neurons = int(physics_data['num_neurons'])
        exc_ratio = float(physics_data['excitatory_ratio'])
        tau_vm_vec = tuple(physics_data['tau_Vm'].tolist())
    except:
        print("无生物数据，使用默认设置")
        num_neurons = 509
        exc_ratio = 0.76
        tau_vm_vec = tuple([10.0] * 509)

    model = ConnSNN(
        out_dims=10,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        tau_Vm_vector=tau_vm_vec,
        K_in=args.K_in,
        K_h=args.K_h,
        K_out=args.K_out,
        dt=dt_ms
    )
    
    # 3. 初始化参数 (生物拓扑 + 随机)
    # 我们用稍微真实的连接，而不是全连接
    rng_net = jax.random.PRNGKey(0)
    # 输入层稀疏度 0.1
    k_in = jax.random.bernoulli(rng_net, 0.1, (2*196, num_neurons))
    # 递归层稀疏度 0.05
    k_h = jax.random.bernoulli(rng_net, 0.05, (num_neurons, num_neurons))
    k_out = jax.random.bernoulli(rng_net, 0.5, (num_neurons, 10))
    
    params = freeze({
        'kernel_in': k_in, 'kernel_h': k_h, 'kernel_out': k_out
    })
    
    # 4. 运行一次前向传播 (模拟 vmap 内部)
    # 重新初始化 carry (无 batch 维)
    carry_batch = model.initial_carry(rng, 1)
    carry = jax.tree_map(lambda x: x[0], carry_batch)
    
    variables = {'params': params, 'fixed_weights': freeze({'dummy': None})}
    
    # 运行！
    final_carry, output = model.apply(variables, carry, real_input)
    v_m_final, i_syn_final, rate_final, spike_final = final_carry
    
    # 5. 核心诊断指标
    print("\n=== 网络状态诊断 ===")
    
    # A. 膜电位分析
    print(f"平均膜电位 (V_m): {jnp.mean(v_m_final):.4f}")
    print(f"最大膜电位 (V_m): {jnp.max(v_m_final):.4f} (阈值是 1.0)")
    
    if jnp.max(v_m_final) < 0.5:
        print("❌ 电压过低！从未接近阈值。输入电流严重不足。")
        print("💡 必须大幅增加 K_in。")
    elif jnp.max(v_m_final) > 5.0:
        print("⚠️ 电压过高，可能饱和。")
    else:
        print("✅ 电压范围看起来健康。")
        
    # B. 发放率分析
    # rate_final 存储的是最后时刻的平滑发放率，或者我们可以看 spike_final
    # 但由于我们无法获取中间过程的 spike，这里只能看 rate
    mean_rate = jnp.mean(rate_final)
    print(f"平均发放率指标 (Rate): {mean_rate:.6f}")
    
    if mean_rate < 1e-5:
        print("❌ 网络完全静默 (Silence)。")
    
    # C. 输出分析
    print(f"输出 Logits: {output}")
    print(f"输出范围: {jnp.max(output) - jnp.min(output):.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K_in", type=float, default=10.0)
    parser.add_argument("--K_h", type=float, default=1.0)
    parser.add_argument("--K_out", type=float, default=100.0)
    args = parser.parse_args()
    main(args)