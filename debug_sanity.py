import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from networks.conn_snn import ConnSNN

def run_single_pass(model, params, x_seq):
    """运行一次前向传播，不使用 scan (因为输入已经是序列)"""
    # 构造 dummy fixed_weights
    variables = {'params': params, 'fixed_weights': freeze({'dummy': None})}
    
    # 初始化 carry (batch_size=1)
    key = jax.random.PRNGKey(0)
    carry_batch = model.initial_carry(key, 1)
    # 去掉 batch 维度，因为我们在模拟 vmap 内部的一次调用
    carry = jax.tree_map(lambda x: x[0], carry_batch)
    
    # 运行
    final_carry, output = model.apply(variables, carry, x_seq)
    return output

def test_sanity():
    print("=== 开始 ConnSNN 健全性测试 (Sanity Check) ===")
    
    # 1. 配置极简网络
    num_neurons = 10 
    in_dims = 4
    out_dims = 2
    time_steps = 20 # 跑 20 步
    
    # 强制所有神经元为兴奋性 (简化测试)
    exc_ratio = 1.0 
    # 统一 Tau
    tau_vec = tuple([10.0] * num_neurons)

    model = ConnSNN(
        out_dims=out_dims,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        tau_Vm_vector=tau_vec,
        K_in=10.0,  # 强输入
        K_h=0.0,    # 禁用循环 (便于观察前馈逻辑)
        K_out=10.0, # 强输出
        dt=0.5
    )
    
    # 2. 构造“手术刀”式的手动权重
    # 我们只用前两个神经元：
    # Neuron 0: 负责传递 Input[0] -> Output[0]
    # Neuron 1: 负责传递 Input[1] -> Output[1]
    
    k_in = np.zeros((2 * in_dims, num_neurons), dtype=bool)
    k_in[0, 0] = True # Input 0 (正) 连接到 Neuron 0
    k_in[1, 1] = True # Input 1 (正) 连接到 Neuron 1
    
    k_h = np.zeros((num_neurons, num_neurons), dtype=bool) # 无递归
    
    k_out = np.zeros((num_neurons, out_dims), dtype=bool)
    k_out[0, 0] = True # Neuron 0 连接到 Output 0
    k_out[1, 1] = True # Neuron 1 连接到 Output 1
    
    params = freeze({
        'kernel_in': jnp.array(k_in),
        'kernel_h': jnp.array(k_h),
        'kernel_out': jnp.array(k_out)
    })

    # --- 测试 A: 静默测试 ---
    print("\n[测试 A] 静默测试 (全 0 输入)")
    x_silent = jnp.zeros((time_steps, in_dims), dtype=jnp.float32)
    out_silent = run_single_pass(model, params, x_silent)
    print(f"输出: {out_silent}")
    if jnp.allclose(out_silent, 0.0):
        print("✅ 通过: 无输入则无输出")
    else:
        print("❌ 失败: 存在幻像噪音")

    # --- 测试 B: 通路 0 激活 ---
    print("\n[测试 B] 通路 0 测试 (激活 Input 0 -> 应该 Output 0 高)")
    # 模拟 Input 0 持续发放 (概率 1.0)
    x_ch0 = np.zeros((time_steps, in_dims), dtype=np.float32)
    x_ch0[:, 0] = 1.0 
    out_ch0 = run_single_pass(model, params, jnp.array(x_ch0))
    print(f"输出: {out_ch0}")
    
    if out_ch0[0] > 1.0 and out_ch0[1] == 0.0:
        print("✅ 通过: 信号正确地从 Input 0 传到了 Output 0，且没有串线")
    elif out_ch0[0] == 0.0:
        print("❌ 失败: 信号中断，未能激活 Output 0 (检查 K_in/K_out 是否过小)")
    else:
        print(f"❌ 失败: 信号串线或异常 (Out1={out_ch0[1]})")

    # --- 测试 C: 通路 1 激活 ---
    print("\n[测试 C] 通路 1 测试 (激活 Input 1 -> 应该 Output 1 高)")
    x_ch1 = np.zeros((time_steps, in_dims), dtype=np.float32)
    x_ch1[:, 1] = 1.0 
    out_ch1 = run_single_pass(model, params, jnp.array(x_ch1))
    print(f"输出: {out_ch1}")
    
    if out_ch1[1] > 1.0 and out_ch1[0] == 0.0:
        print("✅ 通过: 信号正确地从 Input 1 传到了 Output 1")
    else:
        print("❌ 失败")

if __name__ == "__main__":
    test_sanity()