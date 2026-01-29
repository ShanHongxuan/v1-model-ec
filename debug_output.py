import argparse
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from networks.conn_snn import ConnSNN

def run_simulation(model_def, params, fixed_weights, num_steps, x_seq):
    # x_seq: (Time, In_Dims)
    
    @partial(jax.jit, static_argnums=(0,))
    def jit_apply(model_def_static, carry, x):
        variables = {'params': params, 'fixed_weights': fixed_weights}
        # 使用修改后的 ConnSNN，它接受时序输入 (Time, In)
        # 注意：这里我们不需要 scan，因为 ConnSNN.__call__ 内部已经包含了 scan
        final_carry, output = model_def_static.apply(variables, carry, x)
        return output

    key = jax.random.PRNGKey(0)
    batch_size = 1 # 我们只测一个样本，但 ConnSNN 需要输入形状匹配
    # 注意：vmap 模式下 ConnSNN 期望 (Time, In)，Batch 模式下期望 (Batch, Time, In)
    # 为了模拟 ec.py 的行为，我们这里不使用 Batch 维度，直接模拟 vmap 内部的一次调用
    
    # 重新初始化一个不带 Batch 的 carry
    # ConnSNN.initial_carry 通常返回 (Batch, N)，我们需要 (N,)
    carry_batch = model_def.initial_carry(key, 1)
    carry = jax.tree_map(lambda x: x[0], carry_batch)

    output = jit_apply(model_def, carry, x_seq)
    return output

def main(args):
    print(f"--- 诊断输出: K_in={args.K_in}, K_h={args.K_h}, K_out={args.K_out} ---")
    
    # 模拟 14x14 = 196 输入
    in_dims = 196
    time_steps = 200 # 100ms

    try:
        physics_data = np.load('neuron_physics.npz')
        num_neurons = int(physics_data['num_neurons'])
        exc_ratio = float(physics_data['excitatory_ratio'])
        tau_vm_vec = tuple(physics_data['tau_Vm'].tolist())
    except FileNotFoundError:
        print("错误: 找不到 neuron_physics.npz")
        return

    # 实例化模型
    model = ConnSNN(
        out_dims=10,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        tau_Vm_vector=tau_vm_vec,
        K_in=args.K_in,
        K_h=args.K_h,
        K_out=args.K_out, # [关键] 传入 K_out
        dt=0.5
    )

    # 生成模拟数据
    key = jax.random.PRNGKey(42)
    
    # 1. 模拟泊松输入 (Time, 196)
    # 假设输入强度 0.5
    input_probs = jnp.full((time_steps, in_dims), 0.1) 
    x_seq = jax.random.bernoulli(key, input_probs).astype(jnp.float32)
    
    # 2. 模拟参数 (稀疏)
    # 这里我们只关心输出值的量级，随机稀疏即可
    dummy_params = freeze({
        'kernel_in': jax.random.bernoulli(key, 0.1, (2*in_dims, num_neurons)),
        'kernel_h': jax.random.bernoulli(key, 0.1, (num_neurons, num_neurons)),
        'kernel_out': jax.random.bernoulli(key, 0.5, (num_neurons, 10))
    })
    dummy_fixed_weights = freeze({'dummy': None})

    # 运行
    try:
        logits = run_simulation(model, dummy_params, dummy_fixed_weights, time_steps, x_seq)
        
        print("\n=== 诊断结果 ===")
        print(f"Logits (原始输出): {logits}")
        print(f"Logits Mean: {jnp.mean(logits):.6f}, Max: {jnp.max(logits):.6f}, Min: {jnp.min(logits):.6f}")
        print(f"Logits Range (Max-Min): {jnp.max(logits) - jnp.min(logits):.6f}")
        
        probs = jax.nn.softmax(logits)
        print(f"Softmax Probs: {probs}")
        print(f"Max Prob: {jnp.max(probs):.4f} (随机猜测是 0.1)")
        
        if jnp.max(probs) < 0.15:
            print("❌ 输出区分度太低！Softmax 后几乎是均匀分布。")
            print("💡 建议: 大幅增加 K_out")
        else:
            print("✅ 输出区分度尚可。")
            
    except Exception as e:
        print(f"运行出错: {e}")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K_in", type=float, default=0.1)
    parser.add_argument("--K_h", type=float, default=0.08)
    parser.add_argument("--K_out", type=float, default=5.0) # 默认值
    args = parser.parse_args()
    main(args)