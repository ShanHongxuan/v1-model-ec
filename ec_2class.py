# ==================== [1] TF 预初始化 ====================
try:
    from utils.mnist_loader import load_mnist_data
    print(">>> [System] 预加载 MNIST 数据...")
    _ = load_mnist_data()
    _TF_PREINIT_SUCCESS = True
except:
    _TF_PREINIT_SUCCESS = False
# =========================================================

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from omegaconf import OmegaConf
from tqdm import tqdm
from functools import partial
import os
from typing import Any, Tuple, Dict

from networks import NETWORKS
from envs.mnist_env import MnistEnv
from brax.envs import wrappers
from brax.training.acme import running_statistics
from brax.training.acme import specs
from utils.functions import mean_weight_abs

# ==================== 配置类 ====================
@flax.struct.dataclass
class ESConfig:
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None
    pop_size:       int = 2048
    lr:           float = 0.1
    eps:          float = 1e-3
    weight_decay: float = 0.
    warmup_steps:   int = 0
    eval_size:      int = 128
    action_dtype: Any   = jnp.float32
    p_dtype:       Any  = jnp.float32
    network_dtype: Any  = jnp.float32
    clip_action:   bool = False
    normalize_obs: bool = False

@flax.struct.dataclass
class RunnerState:
    key: Any
    normalizer_state: running_statistics.RunningStatisticsState
    env_reset_pool: Any
    params:        Any
    fixed_weights: Any
    opt_state:     Any

@flax.struct.dataclass
class PopulationState:
    network_params: Any
    network_states: Any
    env_states:     Any
    fitness_totrew: jnp.ndarray
    fitness_sum:    jnp.ndarray
    fitness_n:      jnp.ndarray

# ==================== 辅助函数 ====================
def _centered_rank_transform(x):
    shape = x.shape
    x = x.ravel()
    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)

def _sample_bernoulli_parameter(key, params, dtype, batch_size=()):
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    return jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

def _deterministic_bernoulli_parameter(params, batch_size=()):
    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)

# ==================== 评估步骤 ====================
def _evaluate_step(pop, runner, conf):
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))
    
    new_network_states, act = vmapped_apply(
        {"params": pop.network_params, "fixed_weights": runner.fixed_weights}, 
        pop.network_states, pop.env_states.obs
    )
    
    # 禁用 Clip
    # act = jnp.clip(act, -1, 1) 
        
    new_env_states = conf.env_cls.step(pop.env_states, act)
    
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward
    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n + 1, pop.fitness_n)
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)
    
    return pop.replace(
        network_states=new_network_states,
        env_states=new_env_states,
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )

# ==================== [核心修改] 训练一代 (Run Generation) ====================
@partial(jax.jit, donate_argnums=(0,), static_argnums=(1, 2))
def _run_generation(runner, conf, eval_batch_indices):
    """
    eval_batch_indices: 本代要评估的图片索引列表 (Device Array)
    """
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # 1. 采样种群参数
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype, (conf.pop_size - conf.eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))
    network_params = jax.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)

    # 初始化累计 Fitness
    total_fitness = jnp.zeros(conf.pop_size)
    total_eval_n = jnp.zeros(conf.pop_size, dtype=jnp.int32)

    # 2. [循环评估] 遍历 eval_batch_indices 中的每一张图片
    # scan 的 carry 是 (total_fitness, total_eval_n)
    # scan 的 x 是 image_index
    def _eval_one_image(carry, img_idx):
        fit_sum, fit_n = carry
        
        # 构造本轮的环境状态 (强制所有人看同一张图 img_idx)
        # 注意: MnistEnv.reset 实际上是随机取图。
        # 我们需要一种方法指定 index。
        # HACK: 我们传入一个特殊的 seed，让 Env 内部解析出 index?
        # 不，最好的办法是直接调用 env.reset 但传入特定的 key，
        # 或者我们修改 Env 让它支持指定 Index。
        # 为了不改 Env，我们这里用一种简化的假设：
        # 我们假设外部已经准备好了 env_states (通过 hack reset)，或者
        # 我们利用 Env 的 reset 机制，通过精心构造的 key 来控制 (太难)。
        
        # 简单方案：直接在这里构造 EnvState，跳过 reset 的随机逻辑
        # 我们需要访问全局的 images 和 labels，这在 JIT 中比较麻烦。
        # 最佳方案：MnistEnv 支持直接传入 image_index
        
        # 既然我们不想改 Env 太多，我们这里用一个小技巧：
        # 我们把 images 传给 reset 的 key? 不行。
        
        # 回退一步：我们在 Python 端循环，不把循环 JIT 进去。
        # 这样我们可以每次调用 reset 时传入特定的数据。
        # 但这会慢。
        
        # 为了高性能，我们假设 MnistEnv.reset(key) 中的 key 就是 image_index!
        # 我们需要修改 MnistEnv 吗？是的，微调一下最好。
        # 但现在我们先假设我们可以通过某种方式控制。
        
        # [临时方案] 
        # 我们在 _run_generation 外部做循环！
        # 这样 _run_generation 只跑一张图。
        # 不，那样梯度更新就太频繁了。
        
        # [最终方案]
        # 这里的 env_cls 是 VmapWrapper(MnistEnv)。
        # 我们调用 conf.env_cls.reset(indices) 
        # 我们需要修改 MnistEnv 的 reset 让他接受 index。
        
        # 为了不卡在这里，我们用一个假设：
        # eval_batch_indices 是 (N_imgs, Pop_Size, 2) 的 Keys
        # 我们直接用这些 Keys reset。
        # 只要外部保证这些 Keys 对应了平衡的 0/1 样本即可。
        # 但 MnistEnv 是随机取样...
        
        return carry, None # 占位

    # --- 重新设计 ---
    # 我们不在 JIT 内部做多图循环，那太复杂了。
    # 我们在 Python 端做循环，累积梯度，然后更新一次。
    # 这就是标准的 Batch Gradient Descent。
    
    return runner, None # 占位，见下文 main 函数修改

# ==================== [6] 探针 ====================
def probe_network(network, runner, env, key):
    binary_params = jax.tree_util.tree_map(lambda p: p > 0.5, runner.params)
    variables = {'params': binary_params, 'fixed_weights': runner.fixed_weights}
    
    logit_0, logit_1 = None, None
    prob_0, prob_1 = None, None
    
    # 暴力搜索 0 和 1
    rng = key
    for _ in range(50):
        rng, subkey = jax.random.split(rng)
        state = env.reset(subkey) 
        label = int(state.current_label)
        obs = state.obs
        
        carry = network.initial_carry(subkey, 1)
        _, output = network.apply(variables, carry, obs)
        logits = output[0]
        probs = jax.nn.softmax(logits)
        
        if label == 0 and logit_0 is None:
            logit_0, prob_0 = logits, probs
        elif label == 1 and logit_1 is None:
            logit_1, prob_1 = logits, probs
            
        if logit_0 is not None and logit_1 is not None:
            break
            
    return logit_0, prob_0, logit_1, prob_1

# ==================== [7] 主程序 ====================
def main():
    print("=== 二分类平衡批次训练 (Balanced Batch Training) ===")
    
    # 1. 数据准备 (保持不变)
    print(">>> 准备平衡数据集...")
    images, labels = load_mnist_data('train')
    mask0 = labels == 0
    mask1 = labels == 1
    imgs0, imgs1 = images[mask0], images[mask1]
    
    val_imgs = np.concatenate([imgs0[:16], imgs1[:16]])
    val_labels = np.concatenate([np.zeros(16), np.ones(16)])
    
    # 2. 配置
    K_IN = 100.0
    K_H = 0.5
    K_OUT = 0.1
    
    conf = OmegaConf.create({
        "seed": 42,
        "pop_size": 1024,
        "lr": 0.1,
        "total_generations": 100,
        "batch_size": 16,
        
        # [修复] 添加缺失的参数
        "eval_size": 128,  # 评估集大小
        "eps": 0.001,      # 概率截断阈值
        
        "network_conf": {
            "num_neurons": 509, "excitatory_ratio": 0.76,
            "K_in": K_IN, "K_h": K_H, "K_out": K_OUT, "dt": 0.5
        },
        "use_bio": True, "mix": 0.5
    })
    
    # 3. 环境与网络
    # 注意：我们这里用全量数据初始化 Env，但在 reset 时我们会 hack 它
    snn_steps = 200
    # 仅使用 0/1 数据集
    all_imgs_2c = np.concatenate([imgs0, imgs1])
    all_lbls_2c = np.concatenate([np.zeros(len(imgs0)), np.ones(len(imgs1))])
    
    base_env = MnistEnv(all_imgs_2c, all_lbls_2c, presentation_steps=snn_steps, dt_ms=0.5)
    base_env.action_size = 2
    env = wrappers.VmapWrapper(base_env)
    
    network_cls = NETWORKS["ConnSNN"]
    tau_vec = None
    prob_mat = None
    if os.path.exists('neuron_physics.npz'):
        phys = np.load('neuron_physics.npz')
        tau_vec = tuple(phys['tau_Vm'].tolist())
        if conf.use_bio and os.path.exists('init_probability.npy'):
            raw = np.load('init_probability.npy')
            prob_mat = conf.mix * raw + (1.0 - conf.mix) * 0.5

    network = network_cls(out_dims=2, tau_Vm_vector=tau_vec, **conf.network_conf)
    
    # 4. ES Setup
    optim = optax.chain(optax.scale_by_adam(), optax.scale(-conf.lr))
    es_conf = ESConfig(
        network_cls=network, optim_cls=optim, env_cls=env,
        pop_size=conf.pop_size, clip_action=False, normalize_obs=False
    )
    
    # 初始化
    key_run, key_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    dummy_obs = jnp.zeros((conf.pop_size, 200, 196))
    init_carry = network.initial_carry(key_init, conf.pop_size)
    vars_init = network.init(key_init, init_carry, dummy_obs)
    
    net_params = vars_init['params']
    if prob_mat is not None:
        bio_jnp = jnp.array(prob_mat)
        def _mapper(path, p):
            return bio_jnp if path[-1] == 'kernel_h' else jnp.full_like(p, 0.5)
        net_params = jax.tree_util.tree_map_with_path(_mapper, net_params)
    else:
        net_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5), net_params)
        
    opt_state = optim.init(net_params)
    env_pool = env.reset(jax.random.split(key_init, conf.pop_size)) # Dummy pool
    
    runner = RunnerState(
        key=key_run,
        normalizer_state=running_statistics.init_state(specs.Array((196,), jnp.float32)),
        env_reset_pool=env_pool,
        params=net_params,
        fixed_weights=vars_init['fixed_weights'],
        opt_state=opt_state
    )

    # ================= [关键] 自定义单步训练函数 =================
    # 我们将 _evaluate_step 提取出来，支持指定输入
    @jax.jit
    def evaluate_batch(params, fixed_weights, batch_obs, batch_labels):
        pop_size = jax.tree_util.tree_leaves(params)[0].shape[0]
        
        # 扩展输入以匹配 PopSize
        # obs: (Pop, Time, Feat)
        obs_broadcast = jnp.repeat(jnp.expand_dims(batch_obs, 0), pop_size, axis=0)
        
        # 初始化状态
        carry = network.initial_carry(jax.random.PRNGKey(0), pop_size)
        
        # [核心修复] 使用 vmap 进行批量评估
        # in_axes: 
        #   variables: {'params': 0 (种群维), 'fixed_weights': None (不分种群)}
        #   carry: 0 (每个个体有自己的状态)
        #   x: 0 (每个个体有自己的输入)
        vmapped_apply = jax.vmap(
            network.apply, 
            in_axes=({'params': 0, 'fixed_weights': None}, 0, 0)
        )
        
        # 调用 vmap 后的函数
        _, output = vmapped_apply(
            {'params': params, 'fixed_weights': fixed_weights}, 
            carry, 
            obs_broadcast
        )
        
        # 计算奖励 (Softmax)
        logits = output - jnp.max(output, axis=-1, keepdims=True)
        probs = jax.nn.softmax(logits)
        
        rewards = probs[:, batch_labels] 
        
        return rewards

    @partial(jax.jit, donate_argnums=(0,))
    def train_step_balanced(runner, batch_imgs, batch_lbls):
        # 1. 采样参数
        new_key, run_key = jax.random.split(runner.key)
        runner = runner.replace(key=new_key)
        
        # 注意：这里使用 es_conf.network_dtype 是对的
        train_params = _sample_bernoulli_parameter(run_key, runner.params, es_conf.network_dtype, (conf.pop_size - conf.eval_size, ))
        eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))
        
        pop_params = jax.tree_util.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)
        
        # 2. 循环评估
        def _scan_body(cum_fitness, idx):
            img = batch_imgs[idx] 
            lbl = batch_lbls[idx] 
            
            rewards = evaluate_batch(pop_params, runner.fixed_weights, img, lbl)
            return cum_fitness + rewards, None

        total_fitness, _ = jax.lax.scan(_scan_body, jnp.zeros(conf.pop_size), jnp.arange(batch_imgs.shape[0]))
        
        avg_fitness = total_fitness / batch_imgs.shape[0]
        
        # 3. 分割 Train/Eval
        fit_train, fit_eval = jnp.split(avg_fitness, [conf.pop_size - conf.eval_size])
        
        # 4. 梯度更新
        weight = _centered_rank_transform(fit_train)
        
        def _nes_grad(p, theta):
            w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
            return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)

        grads = jax.tree_util.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]), runner.params, pop_params)
        
        # [修复] 使用 es_conf.optim_cls 而不是 conf.optim_cls
        updates, new_opt_state = es_conf.optim_cls.update(grads, runner.opt_state, runner.params)
        
        new_params = optax.apply_updates(runner.params, updates)
        new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)
        
        runner = runner.replace(params=new_params, opt_state=new_opt_state)
        
        # 计算梯度模长用于诊断
        grad_norm = jnp.mean(jnp.abs(grads['kernel_h']))
        
        return runner, jnp.mean(fit_train), jnp.mean(fit_eval), grad_norm
    # --- 训练循环 ---
    print(">>> 开始训练 (Batch Size = 16, Balanced)...")
    pbar = tqdm(range(1, conf.total_generations + 1))
    
    # 预先生成泊松脉冲 (为了加速，我们不每次 reset，而是重用一组数据)
    # 构造 16 个固定的泊松序列用于训练 (8个0, 8个1)
    # 注意：在真实训练中应该每代换数据，但为了过拟合测试，固定数据更好
    rng_data = jax.random.PRNGKey(999)
    
    def make_poisson_batch(key, imgs, lbls):
        # imgs: (B, 196)
        probs = imgs * (100.0 * 0.5 / 1000.0) # 100Hz
        probs = jnp.expand_dims(probs, 1) # (B, 1, 196)
        probs = jnp.repeat(probs, snn_steps, axis=1) # (B, T, 196)
        spikes = jax.random.bernoulli(key, probs).astype(jnp.float32)
        return spikes, lbls

    train_spikes, train_lbls = make_poisson_batch(rng_data, val_imgs, val_labels) # (16, 200, 196)

    for step in pbar:
        runner, fit, eval_fit, grad = train_step_balanced(runner, train_spikes, train_lbls.astype(jnp.int32))
        
        desc = f"Fit:{fit:.3f} | Eval:{eval_fit:.3f} | Grad:{grad:.5f}"
        
        # 探针
        if step % 10 == 0:
            l0, p0, l1, p1 = probe_network(network, runner, base_env, jax.random.PRNGKey(step))
            if l0 is not None:
                # 转换为 numpy 方便打印
                l0_np = np.array(l0)
                l1_np = np.array(l1)
                
                diff_0 = l0[0] - l0[1]
                diff_1 = l1[1] - l1[0]
                
                tqdm.write(f"\n[Gen {step} Probe]")
                tqdm.write(f"  Input 0 -> Logits: {l0_np} | Prob(0): {p0[0]:.4f} | Diff: {diff_0:.2f}")
                tqdm.write(f"  Input 1 -> Logits: {l1_np} | Prob(1): {p1[1]:.4f} | Diff: {diff_1:.2f}")
                
                if p0[0] > 0.9 and p1[1] > 0.9:
                    tqdm.write("🚀 2分类训练成功！")
                    return

        pbar.set_description(desc)

if __name__ == "__main__":
    main()