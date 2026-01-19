import argparse
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze

# 假设 conn_snn 在 networks 文件夹中
from networks.conn_snn import ConnSNN

def run_simulation(model_def, params, fixed_weights, num_steps, in_dims):
    @partial(jax.jit, static_argnums=(0,))
    def jit_step_fn(model_def_static, current_carry, constant_input):
        variables = {'params': params, 'fixed_weights': fixed_weights}
        new_carry, output = model_def_static.apply(variables, current_carry, constant_input)
        return new_carry, output

    key = jax.random.PRNGKey(0)
    batch_size = 1
    carry = model_def.initial_carry(key, batch_size)
    constant_input = jnp.ones((batch_size, in_dims), dtype=jnp.float32)

    for _ in range(num_steps):
        carry, _ = jit_step_fn(model_def, carry, constant_input)
        
    return carry

def main(args):
    print(f"--- 测试参数: K_in = {args.K_in}, K_h = {args.K_h} ---")
    
    in_dims = 784

    try:
        physics_data = np.load('neuron_physics.npz')
        num_neurons = int(physics_data['num_neurons'])
        exc_ratio = float(physics_data['excitatory_ratio'])
        tau_vm_vec = tuple(physics_data['tau_Vm'].tolist())
    except FileNotFoundError:
        print("错误: 找不到 neuron_physics.npz。")
        return

    model = ConnSNN(
        out_dims=10,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        tau_Vm_vector=tau_vm_vec,
        K_in=args.K_in,
        K_h=args.K_h,
        dt=0.5
    )

    # [修改] 创建更真实的稀疏连接矩阵
    print("正在创建稀疏连接用于测试...")
    key = jax.random.PRNGKey(42)
    
    # 1. 递归连接 (kernel_h): 使用生物概率矩阵
    try:
        bio_prob_matrix = np.load('init_probability.npy')
        # 根据混合因子调整 (与 ec.py 保持一致)
        mix_factor = 0.2
        mixed_prob = mix_factor * bio_prob_matrix + (1.0 - mix_factor) * 0.5
        
        key, subkey = jax.random.split(key)
        kernel_h_bool = jax.random.uniform(subkey, bio_prob_matrix.shape) < mixed_prob
        print(f"递归连接稀疏度: {1 - kernel_h_bool.mean():.2f}")
    except FileNotFoundError:
        print("警告: 找不到 init_probability.npy, kernel_h 将使用 50% 随机稀疏。")
        key, subkey = jax.random.split(key)
        kernel_h_bool = jax.random.uniform(subkey, (num_neurons, num_neurons)) < 0.5

    # 2. 输入连接 (kernel_in): 随机稀疏 (例如 10%)
    input_sparsity = 0.1
    key, subkey = jax.random.split(key)
    kernel_in_bool = jax.random.uniform(subkey, (2 * in_dims, num_neurons)) < input_sparsity
    print(f"输入连接稀疏度: {1 - input_sparsity:.2f}")

    dummy_params = freeze({
        'kernel_in': kernel_in_bool,
        'kernel_h': kernel_h_bool,
        'kernel_out': jnp.ones((num_neurons, 10), dtype=bool) # 输出层不重要
    })
    
    dummy_fixed_weights = freeze({'dummy': None})

    num_steps = 1000
    final_state = run_simulation(model, dummy_params, dummy_fixed_weights, num_steps, in_dims)

    metrics = model.carry_metrics(final_state)
    
    spikes_per_ms = float(metrics['spikes_per_ms'])
    avg_i_syn = float(metrics['avg_i_syn'])

    print(f"结果: spikes_per_ms = {spikes_per_ms:.4f}  |  avg_i_syn = {avg_i_syn:.2f}")
    
    if 0.01 <= spikes_per_ms <= 0.05:
        print("✅ 这个参数组合看起来很有希望！")
    elif spikes_per_ms > 0.1:
        print("⚠️ 过度兴奋，建议减小 K 值。")
    else:
        print("ℹ️ 网络过于抑制或未被充分激活。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="调试 ConnSNN 的发放率")
    parser.add_argument("--K_in", type=float, required=True, help="输入增益")
    parser.add_argument("--K_h", type=float, required=True, help="循环增益")
    
    cli_args = parser.parse_args()
    main(cli_args)