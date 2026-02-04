import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from networks.conn_snn import ConnSNN

def test_inhibition_logic():
    print("=== 抑制逻辑物理验证 (Inhibition Logic Check) ===")
    
    # 1. 构造一个只有 2 个神经元的微型网络
    # Neuron 0: 兴奋性 (Exc)
    # Neuron 1: 抑制性 (Inh)
    num_neurons = 2
    exc_ratio = 0.5 # 1个E, 1个I
    
    model = ConnSNN(
        out_dims=2,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        K_in=0.0,   # 关闭外部输入
        K_h=10.0,   # 开启极强的递归，方便观察
        dt=0.5,
        tau_Vm=10.0
    )

    # 2. 手动构造权重矩阵 (kernel_h)
    # 我们让 Neuron 1 (Inh) 连接到 Neuron 0 (Exc)
    # 矩阵形状 (pre, post) -> (2, 2)
    k_h = np.zeros((num_neurons, num_neurons), dtype=bool)
    k_h[1, 0] = True  # 1 -> 0 的抑制性连接
    
    params = freeze({
        'kernel_in': jnp.zeros((196, num_neurons), dtype=bool),
        'kernel_h': jnp.array(k_h),
        'kernel_out': jnp.zeros((num_neurons, 2), dtype=bool)
    })
    
    # 3. 构造初始状态：让 Neuron 0 的电位处于 0.5 (中立)
    key = jax.random.PRNGKey(0)
    carry_batch = model.initial_carry(key, 1)
    v_m, i_syn, rate, spike = jax.tree_map(lambda x: x[0], carry_batch)
    
    v_m = jnp.array([0.5, 0.0]) # Neuron 0 电位 0.5
    
    # 4. 手动设置 Spike：让 Neuron 1 (Inh) 刚刚发放了一个脉冲
    spike = jnp.array([0, 1], dtype=jnp.int8) 
    
    carry = (v_m, i_syn, rate, spike)
    
    # 5. 运行一步仿真 (0.5ms)
    # 注意：我们的模型接收序列输入，这里给一个全 0 序列跑 1 步
    x_input = jnp.zeros((1, 196))
    
    print(f"初始状态: Neuron 0 Vm = {v_m[0]:.4f}, Neuron 1 Spike = {spike[1]}")
    
    variables = {'params': params, 'fixed_weights': freeze({'dummy': None})}
    new_carry, _ = model.apply(variables, carry, x_input)
    
    new_v_m = new_carry[0]
    new_i_syn = new_carry[1]
    
    print(f"仿真一步后:")
    print(f"  Neuron 0 突触电流 (i_syn): {new_i_syn[0]:.4f}")
    print(f"  Neuron 0 膜电位 (v_m): {new_v_m[0]:.4f}")

    # 6. 判定
    if new_i_syn[0] < 0:
        print("\n✅ 物理逻辑正确：抑制性神经元产生了负电流。")
    else:
        print("\n❌ 逻辑错误：抑制性脉冲没有产生负电流！")
        
    if new_v_m[0] < 0.5:
        print("✅ 物理逻辑正确：膜电位因抑制而下降。")
    else:
        print("❌ 逻辑错误：膜电位未下降，抑制失效。")

if __name__ == "__main__":
    test_inhibition_logic()