import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from networks.conn_snn import ConnSNN
from utils.mnist_loader import load_mnist_data

def main():
    print("=== 抑制作用深度诊断 (Inhibition Diagnosis) ===")
    
    # 1. 准备数据 (只看一张图)
    images, _ = load_mnist_data('train')
    # 找一张典型的数字 0 (像素多) 和 1 (像素少)
    img_0 = images[1] # 通常 index 1 是 '0'
    img_1 = images[3] # 通常 index 3 是 '1'
    
    # 2. 配置 (使用 ec_2class 失败时的参数)
    K_IN = 100.0
    K_H = 0.5     # 这里我们怀疑抑制不够
    K_OUT = 100.0
    
    # 3. 初始化网络
    num_neurons = 509
    exc_ratio = 0.76
    num_exc = int(round(num_neurons * exc_ratio))
    
    # 模拟 Tau
    tau_vec = tuple([10.0] * num_neurons)
    
    model = ConnSNN(
        out_dims=2,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        K_in=K_IN, K_h=K_H, K_out=K_OUT, dt=0.5,
        tau_Vm_vector=tau_vec
    )
    
    # 4. 初始化参数 (使用生物混合概率，模拟真实情况)
    rng = jax.random.PRNGKey(42)
    # 模拟一个 dense 连接 (概率 0.5) 来测试最大潜力
    # 或者加载您的 init_probability.npy
    try:
        prob = np.load('init_probability.npy')
        prob = 0.5 * prob + 0.5 * 0.5 # Mix 0.5
        k_h_mask = jax.random.bernoulli(rng, prob).astype(bool)
        print(">>> 已加载生物连接矩阵")
    except:
        print(">>> 使用随机连接矩阵")
        k_h_mask = jax.random.bernoulli(rng, 0.5, (num_neurons, num_neurons))
        
    k_in_mask = jax.random.bernoulli(rng, 0.1, (196, num_neurons))
    k_out_mask = jax.random.bernoulli(rng, 0.5, (num_neurons, 2))
    
    params = freeze({
        'kernel_in': k_in_mask,
        'kernel_h': k_h_mask,
        'kernel_out': k_out_mask
    })
    
    # 5. 定义诊断函数 (提取内部状态)
    @jax.jit
    def run_diagnosis(x_in):
        # 扩展时间维 (200步)
        # x_in: (196,)
        # 归一化模拟 (ConnSNN 内部会再次做，但这里我们需要先生成脉冲)
        
        # 为了精确控制，我们直接生成概率
        prob_per_step = x_in * (100.0 * 0.5 / 1000.0)
        probs = jnp.expand_dims(prob_per_step, 0).repeat(200, axis=0)
        spikes_in = jax.random.bernoulli(jax.random.PRNGKey(0), probs).astype(jnp.float32)
        
        # 运行
        carry = model.initial_carry(jax.random.PRNGKey(0), 1) # Batch=1
        carry = jax.tree_util.tree_map(lambda x: x[0], carry)
        
        # 我们需要改写 apply 来 hook 内部变量？
        # 不，ConnSNN 返回 final_carry，其中包含 rate
        # 但我们看不到 i_spike (电流)。
        # 没关系，我们可以通过 rate 推算电流。
        
        final_carry, output = model.apply(
            {'params': params, 'fixed_weights': freeze({'dummy': None})}, 
            carry, spikes_in
        )
        
        # final_carry: (v_m, i_syn, rate, spike)
        final_rate = final_carry[2]
        
        return final_rate, output

    # 6. 运行并分析
    print("\n--- 运行诊断 (Input: Image 0) ---")
    rate, logits = run_diagnosis(img_0)
    
    # 7. 统计分析 (核心部分)
    rate_E = rate[:num_exc]
    rate_I = rate[num_exc:]
    
    mean_rate_E = jnp.mean(rate_E)
    mean_rate_I = jnp.mean(rate_I)
    
    # 估算总电流贡献
    # E 给全网贡献的正电流 ~ Sum(Rate_E) * K_h
    # I 给全网贡献的负电流 ~ Sum(Rate_I) * K_h
    # 这里忽略稀疏度差异，假设连接概率均匀
    total_exc_drive = jnp.sum(rate_E)
    total_inh_drive = jnp.sum(rate_I)
    
    EI_Ratio_Activity = total_exc_drive / (total_inh_drive + 1e-6)
    
    print(f"Logits: {logits}")
    print(f"平均发放率 (Exc): {mean_rate_E:.4f}")
    print(f"平均发放率 (Inh): {mean_rate_I:.4f}")
    print(f"总兴奋驱动 (Total E-Drive): {total_exc_drive:.2f}")
    print(f"总抑制驱动 (Total I-Drive): {total_inh_drive:.2f}")
    print(f"E/I 驱动比 (Activity Ratio): {EI_Ratio_Activity:.2f}")
    
    print("\n--- 诊断结论 ---")
    if mean_rate_I < 1e-4:
        print("❌ 抑制性神经元完全沉默！网络处于失控状态。")
        print("💡 原因：I 神经元没有接收到足够的输入。")
    elif EI_Ratio_Activity > 3.0:
        print(f"❌ 抑制太弱！兴奋是抑制的 {EI_Ratio_Activity:.1f} 倍。")
        print("   虽然 I 在发放，但它们人少力微，压不住 E。")
        print("💡 建议：需要增强抑制性权重 (Inhibitory Weight Scaling)。")
    elif EI_Ratio_Activity < 0.5:
        print("⚠️ 抑制过强，网络可能被冻结。")
    else:
        print("✅ E/I 平衡良好 (0.5 - 3.0)。")

if __name__ == "__main__":
    main()