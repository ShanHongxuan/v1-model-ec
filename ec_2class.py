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
import pandas as pd 

from networks import NETWORKS
from networks.conn_snn import ConnSNN_Selected # [关键] 导入新网络类
from envs.mnist_env import MnistEnv
from brax.envs import wrappers
from brax.training.acme import running_statistics
from brax.training.acme import specs
from utils.functions import mean_weight_abs

# 手动注册新网络，防止 __init__.py 未更新
NETWORKS["ConnSNN_Selected"] = ConnSNN_Selected

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

# ==================== 评估步骤 (Evaluate Step) ====================
# 主要用于兼容接口，实际上主要逻辑在 train_step_balanced
def _evaluate_step(pop, runner, conf):
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))
    
    new_network_states, act = vmapped_apply(
        {"params": pop.network_params, "fixed_weights": runner.fixed_weights}, 
        pop.network_states, pop.env_states.obs
    )
    
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

# ==================== [核心] 自定义单步训练函数 ====================
@partial(jax.jit, static_argnums=(0,))
def evaluate_batch(network, params, fixed_weights, batch_obs, batch_labels):
    # params: (Pop, ...)
    # batch_obs: (Batch, Time, Feat)
    
    pop_size = jax.tree_util.tree_leaves(params)[0].shape[0]
    
    # 扩展输入以匹配 PopSize -> (Pop, Batch, Time, Feat)
    # 但 vmap 只能处理单一维度映射。
    # 策略：我们将 Batch 维度作为 vmap 的一部分吗？不，Batch 应该在内部处理或外部循环。
    # 正确策略：obs_broadcast: (Pop, Batch, Time, Feat) 太大了。
    # 更好的策略：evaluate_batch 处理的是单个 Image (Batch=1)，我们在外部 scan 循环 Batch。
    
    # 修正后的逻辑：batch_obs 是单个样本 (Time, Feat)
    # obs_broadcast: (Pop, Time, Feat)
    obs_broadcast = jnp.repeat(jnp.expand_dims(batch_obs, 0), pop_size, axis=0)
    
    carry = network.initial_carry(jax.random.PRNGKey(0), pop_size)
    
    vmapped_apply = jax.vmap(
        network.apply, 
        in_axes=({'params': 0, 'fixed_weights': None}, 0, 0)
    )
    
    # output: (Pop, 2)
    _, output = vmapped_apply(
        {'params': params, 'fixed_weights': fixed_weights}, 
        carry, 
        obs_broadcast
    )
    
    logits = output - jnp.max(output, axis=-1, keepdims=True)
    probs = jax.nn.softmax(logits)
    rewards = probs[:, batch_labels] # batch_labels 是标量
    
    return rewards

@partial(jax.jit, donate_argnums=(0,), static_argnums=(3, 4))
def train_step_balanced(runner, batch_imgs, batch_lbls, es_conf, network):
    conf_pop_size = es_conf.pop_size
    conf_eval_size = es_conf.eval_size
    
    new_key, run_key = jax.random.split(runner.key)
    runner = runner.replace(key=new_key)
    
    train_params = _sample_bernoulli_parameter(run_key, runner.params, es_conf.network_dtype, (conf_pop_size - conf_eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf_eval_size, ))
    pop_params = jax.tree_util.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)
    
    # 循环评估 Batch 中的每一张图
    def _scan_body(cum_fitness, idx):
        img = batch_imgs[idx] # (Time, Feat)
        lbl = batch_lbls[idx] # Scalar
        rewards = evaluate_batch(network, pop_params, runner.fixed_weights, img, lbl)
        return cum_fitness + rewards, None

    total_fitness, _ = jax.lax.scan(_scan_body, jnp.zeros(conf_pop_size), jnp.arange(batch_imgs.shape[0]))
    avg_fitness = total_fitness / batch_imgs.shape[0]
    
    fit_train, fit_eval = jnp.split(avg_fitness, [conf_pop_size - conf_eval_size])
    weight = _centered_rank_transform(fit_train)
    
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(es_conf.p_dtype)
        return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)

    grads = jax.tree_util.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf_pop_size - conf_eval_size)]), runner.params, pop_params)
    
    updates, new_opt_state = es_conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)
    new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, es_conf.eps, 1 - es_conf.eps), new_params)
    
    runner = runner.replace(params=new_params, opt_state=new_opt_state)
    grad_norm = jnp.mean(jnp.abs(grads['kernel_h']))
    
    return runner, jnp.mean(fit_train), jnp.mean(fit_eval), grad_norm

# ==================== [新增辅助] L5 兴奋性神经元筛选器 ====================
def get_l5_excitatory_indices(csv_path, total_neurons):
    """
    读取 CSV，复现预处理时的排序逻辑，并筛选出 L5 Excitatory 神经元的索引。
    """
    print(f">>> 正在筛选 L5 Excitatory 神经元 (From {csv_path})...")
    if not os.path.exists(csv_path):
        print(f"⚠️  警告: 找不到 {csv_path}，将回退到默认的前 2 个神经元。")
        return tuple(range(2))
        
    df = pd.read_csv(csv_path)
    
    # 1. 复现 preprocess_data.py 的排序逻辑
    df['EI_rank'] = df['EI'].map({'E': 0, 'I': 1})
    df_sorted = df.sort_values(['EI_rank', 'simple_id']).reset_index(drop=True)
    
    if len(df_sorted) != total_neurons:
        print(f"⚠️  警告: CSV 神经元数量 ({len(df_sorted)}) 与 物理参数 ({total_neurons}) 不一致！")
    
    # 2. 筛选 L5 Excitatory
    l5e_mask = (df_sorted['layer'] == 'L5') & (df_sorted['EI'] == 'E')
    l5e_indices = df_sorted[l5e_mask].index.to_numpy()
    
    print(f"    - 找到 {len(l5e_indices)} 个 L5 Excitatory 神经元。")
    
    if len(l5e_indices) < 2:
        print("❌ 错误: L5E 神经元不足 2 个。")
        return tuple(range(2))
        
    # 3. 选取 2 个代表 (均匀分布)
    selected_indices = np.linspace(0, len(l5e_indices) - 1, 2, dtype=int)
    final_indices = l5e_indices[selected_indices]
    
    print(f"    - 选定 2 个读出神经元 ID: {final_indices}")
    return tuple(final_indices.tolist())

# ==================== 探针 ====================
def probe_network(network, runner, env, key):
    binary_params = jax.tree_util.tree_map(lambda p: p > 0.5, runner.params)
    variables = {'params': binary_params, 'fixed_weights': runner.fixed_weights}
    
    results = {0: {"logit": None, "prob": None, "rate": None}, 
               1: {"logit": None, "prob": None, "rate": None}}
    
    rng = key
    for _ in range(50):
        rng, subkey = jax.random.split(rng)
        state = env.reset(subkey) 
        label = int(state.current_label)
        
        if label in results and results[label]["logit"] is None:
            carry = network.initial_carry(subkey, 1)
            final_carry, output = network.apply(variables, carry, state.obs)
            
            logits = output[0]
            probs = jax.nn.softmax(logits)
            avg_rate = jnp.mean(final_carry[2]) 
            
            results[label]["logit"] = logits
            results[label]["prob"] = probs
            results[label]["rate"] = avg_rate
            
        if results[0]["logit"] is not None and results[1]["logit"] is not None:
            break
            
    return results[0], results[1]

# ==================== [7] 主程序 ====================
def main():
    print("=== 二分类 L5E读出 平衡批次训练 (Time-Windowed) ===")
    
    # 1. 数据准备
    print(">>> 准备平衡数据集...")
    images, labels = load_mnist_data('train')
    mask0 = labels == 0
    mask1 = labels == 1
    imgs0, imgs1 = images[mask0], images[mask1]
    
    val_imgs = np.concatenate([imgs0[:16], imgs1[:16]])
    val_labels = np.concatenate([np.zeros(16), np.ones(16)])
    
    # 2. 配置 (时间参数)
    STEPS_PRE = 100   # 50ms (静默)
    STEPS_STIM = 200  # 100ms (刺激)
    STEPS_RESP = 100  # 50ms (响应/静默)
    TOTAL_STEPS = 400
    
    # 读出窗口: 刺激结束后的 50ms
    READOUT_START = STEPS_PRE + STEPS_STIM
    READOUT_END = TOTAL_STEPS
    
    # 物理参数
    K_IN = 2.0
    K_H = 0.05
    K_OUT = 20.0
    
    # 获取生物数据与读出索引
    NEURON_CSV = '../dataset/mice_unnamed/neurons.csv.gz' # 请确认路径
    
    tau_vec = None
    prob_mat = None
    num_neurons_loaded = 509
    l5e_indices = (0, 1) # 默认
    
    if os.path.exists('neuron_physics.npz'):
        phys = np.load('neuron_physics.npz')
        tau_vec = tuple(phys['tau_Vm'].tolist())
        num_neurons_loaded = int(phys['num_neurons'])
        
        # [核心] 获取 L5E 索引 (只取 2 个)
        l5e_indices = get_l5_excitatory_indices(NEURON_CSV, num_neurons_loaded)
        # 确保只取前2个用于二分类
        if len(l5e_indices) > 2:
            l5e_indices = l5e_indices[:2]
    else:
        print("⚠️ 未找到 physics 文件，使用默认索引。")

    conf = OmegaConf.create({
        "seed": 42,
        "pop_size": 1024,
        "lr": 0.1,
        "total_generations": 500,
        "batch_size": 16,
        "eval_size": 128,
        "eps": 0.001,
        
        "network_conf": {
            "num_neurons": num_neurons_loaded, 
            "excitatory_ratio": 0.76,
            "K_in": K_IN, "K_h": K_H, "K_out": K_OUT, "dt": 0.5,
            
            # [关键] 指定 2 个读出神经元 + 时间窗
            "readout_indices": l5e_indices,
            "readout_start_step": READOUT_START,
            "readout_end_step": READOUT_END
        },
        "use_bio": True, "mix": 0.5
    })
    
    # 加载概率矩阵
    if conf.use_bio and os.path.exists('init_probability.npy'):
        raw = np.load('init_probability.npy')
        prob_mat = conf.mix * raw + (1.0 - conf.mix) * 0.5
        print(f">>> 生物概率已加载 (Mix={conf.mix})")

    # 3. 环境与网络
    # [修改] 传入时间结构参数
    base_env = MnistEnv(
        np.concatenate([imgs0, imgs1]), 
        np.concatenate([np.zeros(len(imgs0)), np.ones(len(imgs1))]), 
        input_hz=200.0, dt_ms=0.5,
        steps_pre_stim=STEPS_PRE, 
        steps_stim=STEPS_STIM, 
        steps_response=STEPS_RESP
    )
    base_env.action_size = 2
    env = wrappers.VmapWrapper(base_env)
    
    network_cls = NETWORKS["ConnSNN_Selected"]
    network = network_cls(out_dims=2, tau_Vm_vector=tau_vec, **conf.network_conf)
    
    # 4. ES Setup
    optim = optax.chain(optax.scale_by_adam(), optax.scale(-conf.lr))
    es_conf = ESConfig(
        network_cls=network, optim_cls=optim, env_cls=env,
        pop_size=conf.pop_size, clip_action=False, normalize_obs=False
    )
    
    # 初始化
    key_run, key_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    dummy_obs = jnp.zeros((conf.pop_size, TOTAL_STEPS, 196))
    init_carry = network.initial_carry(key_init, conf.pop_size)
    vars_init = network.init(key_init, init_carry, dummy_obs)
    
    net_params = vars_init['params']
    if prob_mat is not None:
        bio_jnp = jnp.array(prob_mat)
        def _mapper(path, p): return bio_jnp if path[-1] == 'kernel_h' else jnp.full_like(p, 0.5)
        net_params = jax.tree_util.tree_map_with_path(_mapper, net_params)
    else:
        net_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5), net_params)
        
    opt_state = optim.init(net_params)
    env_pool = env.reset(jax.random.split(key_init, conf.pop_size))
    
    runner = RunnerState(
        key=key_run,
        normalizer_state=running_statistics.init_state(specs.Array((196,), jnp.float32)),
        env_reset_pool=env_pool,
        params=net_params,
        fixed_weights=vars_init['fixed_weights'],
        opt_state=opt_state
    )

    # --- 训练循环 ---
    print(">>> 开始训练...")
    pbar = tqdm(range(1, conf.total_generations + 1))
    
    rng_data = jax.random.PRNGKey(999)
    
    # [修改] 生成时序数据 (3阶段)
    def make_temporal_batch(key, imgs, lbls):
        B = imgs.shape[0]
        rngs = jax.random.split(key, B)
        def _gen_one(rng, img):
            base = img * (1000.0 * 0.5 / 1000.0)
            silence = jnp.zeros_like(base)
            seq = jnp.concatenate([
                jnp.repeat(jnp.expand_dims(silence, 0), STEPS_PRE, axis=0),
                jnp.repeat(jnp.expand_dims(base, 0), STEPS_STIM, axis=0),
                jnp.repeat(jnp.expand_dims(silence, 0), STEPS_RESP, axis=0)
            ], axis=0)
            return jax.random.bernoulli(rng, seq).astype(jnp.float32)
        spikes = jax.vmap(_gen_one)(rngs, imgs)
        return spikes, lbls

    train_spikes, train_lbls = make_temporal_batch(rng_data, val_imgs, val_labels)

    for step in pbar:
        # 运行一代训练
        runner, fit, eval_fit, grad = train_step_balanced(runner, train_spikes, train_lbls.astype(jnp.int32), es_conf, network)
        
        desc = f"Fit:{fit:.3f} | Eval:{eval_fit:.3f} | Grad:{grad:.5f}"
        
        # [核心修改] 增强的探针输出
        if step % 10 == 0:
            results = probe_network(network, runner, base_env, jax.random.PRNGKey(step))
            
            # 获取结果
            res0 = results[0]
            res1 = results[1]
            
            if res0["logit"] is not None and res1["logit"] is not None:
                # 转换数据以便打印
                l0 = np.array(res0["logit"])
                p0 = np.array(res0["prob"])
                r0 = float(res0["rate"])
                
                l1 = np.array(res1["logit"])
                p1 = np.array(res1["prob"])
                r1 = float(res1["rate"])
                
                # 计算差异 (Diff)
                diff0 = l0[0] - l0[1] # 输入0时，Logit[0] 比 Logit[1] 大多少 (越大越好)
                diff1 = l1[1] - l1[0] # 输入1时，Logit[1] 比 Logit[0] 大多少 (越大越好)
                
                # 计算 Logits 的绝对范围 (用于判断 K_out)
                range0 = np.max(l0) - np.min(l0)
                range1 = np.max(l1) - np.min(l1)
                avg_range = (range0 + range1) / 2.0

                tqdm.write(f"\n[Gen {step} Diagnostic] Avg Logit Range: {avg_range:.2f}")
                tqdm.write(f"  Input 0 -> Rate:{r0:.3f} | Logits:{l0} | P(0):{p0[0]:.4f} | Diff: {diff0:.2f}")
                tqdm.write(f"  Input 1 -> Rate:{r1:.3f} | Logits:{l1} | P(1):{p1[1]:.4f} | Diff: {diff1:.2f}")
                
                # [K_out 调节建议]
                if avg_range < 2.0:
                    tqdm.write("  💡 提示: Logits 差异太小，Softmax 梯度微弱。建议 -> 增大 K_out")
                elif avg_range > 50.0:
                    tqdm.write("  💡 提示: Logits 差异极大，Softmax 已饱和。建议 -> 减小 K_out (如果不收敛)")
                else:
                    tqdm.write("  ✅ 提示: Logits 范围适中 (黄金区间 5.0 ~ 30.0)")

                # [发放率监控]
                if r0 > 0.5 or r1 > 0.5:
                    tqdm.write("  ⚠️ 警告: 发放率过高 (饱和风险) -> 建议减小 K_in 或增大抑制")
                elif r0 < 0.01 and r1 < 0.01:
                    tqdm.write("  ⚠️ 警告: 发放率过低 (静默风险) -> 建议增大 K_in")

        pbar.set_description(desc)

if __name__ == "__main__":
    main()