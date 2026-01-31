import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from omegaconf import OmegaConf
from tqdm import tqdm

# 引入你的模块
from networks.conn_snn import ConnSNN
# 假设这些都在当前目录
from networks import NETWORKS 

# ================= 1. 定义一个“作弊”环境 =================
# 这个环境永远只输出同一张图，标签永远是 0
class OverfitEnv:
    def __init__(self, n_features=196, n_steps=200, input_hz=100.0, dt=0.5):
        self.n_features = n_features
        self.n_steps = n_steps
        self.prob_per_step = input_hz * (dt / 1000.0)
        
        # 固定生成一张“假图”：前半部分是亮的，后半部分是暗的
        # 这是一个非常强的特征
        img = np.zeros(n_features)
        img[:n_features//2] = 1.0 
        self.fixed_image = jnp.array(img)
        self.fixed_label = 1 # 永远是类别 0

    def reset(self, rng):
        # 忽略 rng，生成固定的泊松序列
        # 形状: (Batch=1, Time, Features) -> 我们在外部 vmap，所以这里返回 (Time, Features)
        
        # 这里为了简单，我们甚至不使用随机泊松，直接使用确定性输入测试
        # 或者是固定的随机种子
        rng_fixed = jax.random.PRNGKey(999)
        probs = self.fixed_image * self.prob_per_step
        # 扩展时间维
        probs = jnp.expand_dims(probs, 0).repeat(self.n_steps, axis=0)
        obs = jax.random.bernoulli(rng_fixed, probs).astype(jnp.float32)
        
        return {
            'obs': obs,
            'label': self.fixed_label
        }

    def step(self, state, action):
        # action 是 logits
        pred = jnp.argmax(action)
        reward = jnp.where(pred == state['label'], 1.0, 0.0)
        # 同时也计算 Softmax 奖励以便观察
        probs = jax.nn.softmax(action)
        soft_reward = probs[state['label']]
        
        return reward, soft_reward

# ================= 2. 极简训练循环 =================
def main():
    print("=== 开始过拟合测试 (Overfit Test) ===")
    
    # --- 配置 ---
    POP_SIZE = 128 # 小种群
    LR = 0.2
    GENS = 200
    
    # 使用你认为“应该工作”的参数
    K_IN = 12
    K_H = 1
    K_OUT = 100.0
    
    # 模拟 ec.py 的配置结构
    network_config = {
        "out_dims": 10,
        "num_neurons": 509,
        "excitatory_ratio": 0.76,
        "K_in": K_IN,
        "K_h": K_H,
        "K_out": K_OUT,
        "dt": 0.5,
        "tau_Vm_vector": None # 简化，先不用生物 Tau
    }
    
    model = ConnSNN(**network_config)
    
    # 初始化参数 (全 0.5)
    key = jax.random.PRNGKey(0)
    init_obs = jnp.zeros((200, 196)) # Dummy input for init
    
    # [修正] 正确处理 initial_carry
    # 1. 获取带 Batch 的初始状态: (Batch, N)
    init_carry_batch = model.initial_carry(key, 1) 
    
    # 2. 去除 Batch 维度，但保留 (v_m, i_syn, rate, spike) 的元组结构
    # 这里的 lambda x: x[0] 是对元组里的每个数组操作，取第0个样本
    init_carry = jax.tree_map(lambda x: x[0], init_carry_batch)
    
    # 3. 传入完整的 tuple 结构进行初始化
    variables = model.init(key, init_carry, init_obs)
    
    # 进化参数 (Probabilities)
    params = jax.tree_map(lambda x: jnp.full_like(x, 0.5), variables['params'])
    fixed_weights = variables['fixed_weights']
    
    # 优化器
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)
    
    # 环境
    env = OverfitEnv()
    
    # --- 核心函数 (JIT) ---
    @jax.jit
    def train_step(rng, current_params, opt_state):
        # 1. 采样参数 (Binary Masks)
        # noise: 0/1 mask
        noise_keys = jax.random.split(rng, POP_SIZE)
        
        def sample_mask(p, k):
            return jax.random.uniform(k, p.shape) < p
            
        # vmap 采样
        batch_sample = jax.vmap(lambda k: jax.tree_map(lambda p: sample_mask(p, k), current_params))
        binary_params = batch_sample(noise_keys)
        
        # 2. 评估 (Evaluation)
        def evaluate_one(bin_param):
            # Reset
            state = env.reset(jax.random.PRNGKey(0)) # 固定环境种子
            # Forward
            # 注意: ConnSNN 需要 ('params', 'fixed_weights')
            vars_in = {'params': bin_param, 'fixed_weights': fixed_weights}
            carry = model.initial_carry(jax.random.PRNGKey(0), 1)
            # 去掉 batch 维
            carry = jax.tree_map(lambda x: x[0], carry)
            
            _, output = model.apply(vars_in, carry, state['obs'])
            
            # Reward
            acc, soft_rew = env.step(state, output)
            return acc, soft_rew, output

        # Vmap over population
        rewards, soft_rewards, outputs = jax.vmap(evaluate_one)(binary_params)
        
        # 3. NES 更新 (Natural Evolution Strategies)
        # 使用 soft_reward 作为梯度信号通常更好
        fitness = soft_rewards 
        
        # 秩变换 (Centered Rank)
        ranks = jnp.argsort(jnp.argsort(fitness))
        w = (ranks / (POP_SIZE - 1)) - 0.5
        
        # 梯度 = E[w * (theta - p)]
        # theta 是 0/1, p 是概率
        # 注意: 我们需要把 w 广播到参数形状
        def compute_grad(p, theta):
            # theta: (Pop, ...)
            # w: (Pop,)
            w_expanded = w.reshape((-1,) + (1,) * (p.ndim))
            
            # 关键修改：theta.astype(jnp.float32)
            # 必须先转成 float 才能进行减法运算
            return -jnp.mean(w_expanded * (theta.astype(jnp.float32) - p), axis=0)
            
        grads = jax.tree_map(lambda p, theta: compute_grad(p, theta), current_params, binary_params)
        
        # 4. 优化器更新
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(current_params, updates)
        
        # Clip
        new_params = jax.tree_map(lambda x: jnp.clip(x, 0.001, 0.999), new_params)
        
        return new_params, new_opt_state, jnp.mean(rewards), jnp.mean(soft_rewards), outputs[0]

    # --- 循环 ---
    rng = jax.random.PRNGKey(42)
    pbar = tqdm(range(GENS))
        # 用于记录是否已经达标过
    has_reached_target = False
    
    # 用于存储最后一步的输出
    final_sample_out = None
    final_acc = 0.0

    for i in pbar:
        rng, step_key = jax.random.split(rng)
        params, opt_state, mean_acc, mean_soft, sample_out = train_step(step_key, params, opt_state)
        
        # 更新最后的状态
        final_sample_out = sample_out
        final_acc = mean_acc
        
        probs = jax.nn.softmax(sample_out)
        target_prob = probs[1] # 目标是类别 1 (根据之前的修改)
        
        pbar.set_description(f"Acc: {mean_acc:.2f} | Soft: {mean_soft:.3f} | TargetProb: {target_prob:.3f}")
        
        # [修改] 达到目标后不退出，仅打印提示
        if mean_acc > 0.95 and not has_reached_target:
            has_reached_target = True
            # 使用 tqdm.write 防止打乱进度条
            tqdm.write(f"\n✨ 第 {i} 代首次达到目标精度！继续训练至结束...")
            # return  <-- [关键修改] 注释掉这就不会提前退出了

    print("\n" + "="*30)
    print("🏁 训练指定步数完成")
    
    if has_reached_target:
        print("✅ 状态: 成功 (训练过程中曾达到目标)")
    else:
        print("❌ 状态: 失败 (从未达到目标)")

    print(f"最终精度 (Final Acc): {final_acc:.4f}")
    print(f"最终 Logits:\n{final_sample_out}")
    
    # 额外分析一下信号强度
    target_logit = final_sample_out[1]
    other_logits = jnp.delete(final_sample_out, 1)
    avg_noise = jnp.mean(other_logits)
    margin = target_logit - avg_noise
    
    print(f"\n信号强度分析:")
    print(f"目标得分 (Target): {target_logit:.4f}")
    print(f"背景噪音 (Avg Noise): {avg_noise:.4f}")
    print(f"信噪比差值 (Margin): {margin:.4f} (越大越好)")
    print("="*30)

if __name__ == "__main__":
    main()