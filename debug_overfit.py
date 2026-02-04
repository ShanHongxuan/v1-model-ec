import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax import linen as nn
from flax.core import freeze

# 导入网络
from networks.conn_snn import ConnSNN

# ================= 1. 模拟环境 (固定泊松输入) =================
class RealisticOverfitEnv:
    def __init__(self, n_features=196, n_steps=200, input_hz=50.0, dt=0.5):
        self.n_steps = n_steps
        # 生成一个固定的泊松输入序列，作为“过拟合目标”
        # 前半部分亮，后半部分暗 -> 标签 1
        img = np.zeros(n_features)
        img[:n_features//2] = 1.0 
        
        prob = input_hz * (dt/1000.0)
        rng = jax.random.PRNGKey(42)
        
        # 预先生成脉冲，形状 (Time, Features)
        probs = jnp.array(img) * prob
        probs = jnp.expand_dims(probs, 0).repeat(n_steps, axis=0)
        self.fixed_obs = jax.random.bernoulli(rng, probs).astype(jnp.float32)
        self.fixed_label = 1

    def get_obs(self):
        return self.fixed_obs, self.fixed_label

# ================= 2. 辅助函数 =================
def centered_rank(x):
    ranks = jnp.argsort(jnp.argsort(x))
    return (ranks / (len(x) - 1)) - 0.5

# ================= 3. 主程序 =================
def main():
    print("=== 深度诊断版 Overfit Test (全程运行) ===")
    
    # --- 配置 ---
    POP_SIZE = 128
    LR = 0.1
    GENS = 300 # 运行完整的 300 代
    
    # 参数
    K_IN = 15.0
    K_H = 1.0
    K_OUT = 100.0
    
    in_dims = 196
    num_neurons = 509
    exc_ratio = 0.76
    
    print(f"Params: K_in={K_IN}, K_h={K_H}, K_out={K_OUT}")

    # --- 模型与优化器 ---
    tau_vec = tuple([10.0] * num_neurons)
    
    model = ConnSNN(
        out_dims=10,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        K_in=K_IN, K_h=K_H, K_out=K_OUT,
        dt=0.5,
        tau_Vm_vector=tau_vec
    )
    
    optimizer = optax.adam(LR)
    
    # 初始化
    key = jax.random.PRNGKey(0)
    init_obs = jnp.zeros((200, 196))
    
    # [修复] 正确调用 initial_carry 并解包
    init_carry_batch = model.initial_carry(key, 1)
    # 使用 tree_map 去除 batch 维度
    init_carry = jax.tree_util.tree_map(lambda x: x[0], init_carry_batch)
    
    variables = model.init(key, init_carry, init_obs)
    
    # 概率参数初始化为 0.5
    params = jax.tree_map(lambda x: jnp.full_like(x, 0.5), variables['params'])
    fixed_weights = variables['fixed_weights']
    opt_state = optimizer.init(params)
    
    env = RealisticOverfitEnv()
    target_obs, target_label = env.get_obs()
    
    # 检查输入
    print(f"输入脉冲总数: {jnp.sum(target_obs)}")

    # --- 训练步 (JIT) ---
    @jax.jit
    def train_step(rng, current_params, opt_state):
        # 1. 采样
        noise_keys = jax.random.split(rng, POP_SIZE)
        
        def sample_mask(p, k):
            return jax.random.uniform(k, p.shape) < p
            
        binary_params = jax.vmap(lambda k: jax.tree_util.tree_map(lambda p: sample_mask(p, k), current_params))(noise_keys)
        
        # 2. 评估
        def evaluate_one(bin_param):
            vars_in = {'params': bin_param, 'fixed_weights': fixed_weights}
            # 每次评估重新初始化状态
            carry_batch = model.initial_carry(jax.random.PRNGKey(0), 1)
            carry = jax.tree_util.tree_map(lambda x: x[0], carry_batch)
            
            final_carry, output = model.apply(vars_in, carry, target_obs)
            
            probs = jax.nn.softmax(output)
            acc = (jnp.argmax(output) == target_label).astype(jnp.float32)
            score = probs[target_label]
            
            # 监控发放率
            final_rate_mean = jnp.mean(final_carry[2])
            
            return acc, score, output, final_rate_mean

        accs, scores, outputs, rates = jax.vmap(evaluate_one)(binary_params)
        
        # 3. 梯度
        fitness = scores
        fitness_centered = centered_rank(fitness)
        
        def compute_grad(p, theta):
            w_expanded = fitness_centered.reshape((-1,) + (1,) * (p.ndim))
            # [修复] 显式转换 bool -> float32
            return -jnp.mean(w_expanded * (theta.astype(jnp.float32) - p), axis=0)
            
        grads = jax.tree_util.tree_map(lambda p, theta: compute_grad(p, theta), current_params, binary_params)
        
        # 4. 更新
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(current_params, updates)
        new_params = jax.tree_util.tree_map(lambda x: jnp.clip(x, 0.001, 0.999), new_params)
        
        grad_norm = jnp.mean(jnp.abs(grads['kernel_h']))
        
        return new_params, new_opt_state, {
            'acc': jnp.mean(accs),
            'score': jnp.mean(scores),
            'rate': jnp.mean(rates),
            'grad_norm': grad_norm,
            'logits_sample': outputs[0]
        }

    # --- 循环 ---
    rng = jax.random.PRNGKey(100)
    pbar = tqdm(range(GENS))
    
    solved_once = False
    
    for i in pbar:
        rng, key = jax.random.split(rng)
        params, opt_state, metrics = train_step(key, params, opt_state)
        
        logits = metrics['logits_sample']
        logit_range = jnp.max(logits) - jnp.min(logits)
        
        pbar.set_description(
            f"Acc:{metrics['acc']:.2f} | "
            f"Score:{metrics['score']:.3f} | "
            f"Rate:{metrics['rate']:.4f} | "
            f"GNorm:{metrics['grad_norm']:.5f} | "
            f"LRange:{logit_range:.1f}"
        )
        
        # [修改] 不退出，只打印提示
        if metrics['acc'] > 0.98 and not solved_once:
            tqdm.write(f"\n🚀 在第 {i} 代首次收敛！(Accuracy > 98%)")
            solved_once = True

    print("\n=== 训练结束 ===")
    print(f"Final Accuracy: {metrics['acc']:.4f}")
    print(f"Final Target Score: {metrics['score']:.4f}")
    print(f"Final Logits Sample: {logits}")

if __name__ == "__main__":
    main()