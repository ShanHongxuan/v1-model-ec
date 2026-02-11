# ==================== [1] TF 预初始化 ====================
try:
    from utils.mnist_loader import load_mnist_data
    import tensorflow as tf
    # 强制 TF 内存按需增长，防止与 JAX 抢显存
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(">>> [System] 预加载 MNIST 数据以初始化 GPU 环境...")
    # 预加载一次以触发 TF 的初始化
    _ = load_mnist_data()
    print(">>> [System] 预加载完成。")
    _TF_PREINIT_SUCCESS = True
except (ImportError, ModuleNotFoundError):
    print(">>> [System] 未找到 MNIST 环境，将仅支持 Brax 任务。")
    _TF_PREINIT_SUCCESS = False
# =======================================================

from functools import partial
from typing import Any, Dict, Tuple, List
import time
import builtins
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import flax
import optax

from brax import envs
from brax.envs import wrappers 
from brax.training.acme import running_statistics
from brax.training.acme import specs

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import optuna

from networks import NETWORKS
from networks.conn_snn import ConnSNN_Selected, ConnSNN
NETWORKS["ConnSNN_Selected"] = ConnSNN_Selected
NETWORKS["ConnSNN"] = ConnSNN

from utils.functions import mean_weight_abs, finitemean, save_obj_to_file

# 强制使用安全随机数实现
jax.config.update("jax_default_prng_impl", "unsafe_rbg")
builtins.bfloat16 = jnp.dtype("bfloat16").type

# ==================== [2] 配置与状态类 ====================

@flax.struct.dataclass
class ESConfig:
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None

    pop_size:       int = 10240
    lr:           float = 0.15
    eps:          float = 1e-3
    weight_decay: float = 0.
    warmup_steps:   int = 0
    eval_size:      int = 128

    action_dtype: Any   = jnp.float32
    p_dtype:       Any  = jnp.float32
    network_dtype: Any  = jnp.float32
    
    clip_action:   bool = True  
    normalize_obs: bool = True  

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

# ==================== [3] 辅助函数 ====================

def get_l5_excitatory_indices(csv_path, total_neurons, n_out=10):
    """筛选 L5 兴奋性神经元索引"""
    if not os.path.exists(csv_path):
        return tuple(range(n_out))
    df = pd.read_csv(csv_path)
    df['EI_rank'] = df['EI'].map({'E': 0, 'I': 1})
    df_sorted = df.sort_values(['EI_rank', 'simple_id']).reset_index(drop=True)
    l5e_mask = (df_sorted['layer'] == 'L5') & (df_sorted['EI'] == 'E')
    l5e_indices = df_sorted[l5e_mask].index.to_numpy()
    if len(l5e_indices) < n_out: return tuple(l5e_indices.tolist()[:n_out])
    selected = np.linspace(0, len(l5e_indices)-1, n_out, dtype=int)
    return tuple(l5e_indices[selected].tolist())

def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    shape = x.shape
    x = x.ravel()
    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)

def _sample_bernoulli_parameter(key: Any, params: Any, sampling_dtype: Any, batch_size: Tuple = ()) -> Any:
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), sampling_dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))
    return noise

def _deterministic_bernoulli_parameter(params: Any, batch_size: Tuple = ()) -> Any:
    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)

# ==================== [4] 核心训练逻辑 ====================

def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig) -> PopulationState:
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))
    obs_in = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state) if conf.normalize_obs else pop.env_states.obs
    new_network_states, act = vmapped_apply({"params": pop.network_params, "fixed_weights": runner.fixed_weights}, pop.network_states, obs_in)
    if conf.clip_action: act = jnp.clip(act, -1, 1)
    if act.dtype != conf.action_dtype: act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)
    new_env_states = conf.env_cls.step(pop.env_states, act)
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward
    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n   + 1,                  pop.fitness_n)
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)
    def _where_done(x, y):
        done = new_env_states.done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)
    new_env_states = jax.tree_util.tree_map(_where_done, runner.env_reset_pool, new_env_states)
    return pop.replace(network_states=new_network_states, env_states=new_env_states, fitness_totrew=new_fitness_totrew, fitness_sum=new_fitness_sum, fitness_n=new_fitness_n)

@partial(jax.jit, static_argnums=(2,), donate_argnums=(3,))
def _runner_init(key: Any, network_init_key: Any, conf: ESConfig, init_prob_matrix: Any = None) -> RunnerState:
    key, env_init_key = jax.random.split(key)
    env_reset_pool = conf.env_cls.reset(jax.random.split(env_init_key, conf.pop_size))
    network_variables = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
        {"params": network_init_key, "fixed_weights": network_init_key},
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs
    )
    network_params = network_variables["params"]
    if init_prob_matrix is not None:
        bio_prob_jnp = jnp.array(init_prob_matrix, dtype=conf.p_dtype)
        def _init_mapper(path, param):
            if path[-1] == 'kernel_h': return bio_prob_jnp if param.shape == bio_prob_jnp.shape else jnp.full_like(param, 0.5)
            return jnp.full_like(param, 0.5)
        network_params = jax.tree_util.tree_map_with_path(_init_mapper, network_params)
    else:
        network_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_params)
    optim_state = conf.optim_cls.init(network_params)
    runner = RunnerState(key=key, normalizer_state=running_statistics.init_state(specs.Array((conf.env_cls.observation_size, ), jnp.float32)),
        env_reset_pool=env_reset_pool, params=network_params, fixed_weights=network_variables["fixed_weights"], opt_state=optim_state)
    return runner

@partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def _runnner_run(runner: RunnerState, conf: ESConfig) -> Tuple[RunnerState, Dict]:
    metrics = {}
    new_key, run_key, carry_key, env_key = jax.random.split(runner.key, 4)
    runner = runner.replace(key=new_key)
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype, (conf.pop_size - conf.eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))
    network_params = jax.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)
    synchronized_env_states = conf.env_cls.reset(jnp.repeat(jnp.expand_dims(jax.random.split(env_key, 1)[0], 0), conf.pop_size, axis=0))
    pop = PopulationState(network_params=network_params, network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        env_states=synchronized_env_states, fitness_totrew=jnp.zeros(conf.pop_size), fitness_sum=jnp.zeros(conf.pop_size), fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32))
    def _cond(p): return ~jnp.all(p.fitness_n >= 1)
    pop = jax.lax.while_loop(_cond, (lambda p: _evaluate_step(p, runner, conf)), pop)
    fitness, eval_fitness = jnp.split(pop.fitness_sum / pop.fitness_n, [conf.pop_size - conf.eval_size])
    weight = _centered_rank_transform(fitness)
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
        return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)
    grads = jax.tree_map(lambda p, t: _nes_grad(p, t[:(conf.pop_size - conf.eval_size)]), runner.params, pop.network_params)
    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = jax.tree_util.tree_map(lambda p: jnp.clip(optax.apply_updates(runner.params, updates), conf.eps, 1 - conf.eps), new_params)
    runner = runner.replace(params=new_params, opt_state=new_opt_state)
    metrics.update({"fitness": jnp.mean(fitness), "eval_fitness": jnp.mean(eval_fitness), "sparsity": mean_weight_abs(new_params)})
    return runner, metrics

# ==================== [5] MNIST 动态平衡批次逻辑 ====================

@partial(jax.jit, static_argnums=(0,))
def evaluate_batch_fitness(network, params, fixed_weights, single_obs, single_label):
    pop_size = jax.tree_util.tree_leaves(params)[0].shape[0]
    obs_broadcast = jnp.repeat(jnp.expand_dims(single_obs, 0), pop_size, axis=0)
    carry = network.initial_carry(jax.random.PRNGKey(0), pop_size)
    vmapped_apply = jax.vmap(network.apply, in_axes=({'params': 0, 'fixed_weights': None}, 0, 0))
    _, output = vmapped_apply({'params': params, 'fixed_weights': fixed_weights}, carry, obs_broadcast)
    probs = jax.nn.softmax(output - jnp.max(output, axis=-1, keepdims=True))
    return probs[:, single_label]

@partial(jax.jit, donate_argnums=(0,), static_argnums=(3, 4))
def train_step_balanced(runner, batch_imgs, batch_lbls, es_conf, network):
    conf_pop_size, conf_eval_size = es_conf.pop_size, es_conf.eval_size
    new_key, run_key = jax.random.split(runner.key)
    runner = runner.replace(key=new_key)
    train_params = _sample_bernoulli_parameter(run_key, runner.params, es_conf.network_dtype, (conf_pop_size - conf_eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf_eval_size, ))
    pop_params = jax.tree_util.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)
    def _scan_body(cum, idx): return cum + evaluate_batch_fitness(network, pop_params, runner.fixed_weights, batch_imgs[idx], batch_lbls[idx]), None
    total_fitness, _ = jax.lax.scan(_scan_body, jnp.zeros(conf_pop_size), jnp.arange(batch_imgs.shape[0]))
    avg_fitness = total_fitness / batch_imgs.shape[0]
    fit_train, fit_eval = jnp.split(avg_fitness, [conf_pop_size - conf_eval_size])
    weight = _centered_rank_transform(fit_train)
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(es_conf.p_dtype)
        return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)
    grads = jax.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf_pop_size - conf_eval_size)]), runner.params, pop_params)
    updates, new_opt_state = es_conf.optim_cls.update(grads, runner.opt_state, runner.params)
    # 1. 先应用更新得到新参数
    new_params = optax.apply_updates(runner.params, updates)
    # 2. 再对新参数进行裁剪（确保概率在 [eps, 1-eps] 之间）
    new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, es_conf.eps, 1 - es_conf.eps), new_params)
    
    runner = runner.replace(params=new_params, opt_state=new_opt_state)
    return runner, jnp.mean(fit_train), jnp.mean(fit_eval), jnp.mean(jnp.abs(grads['kernel_h']))

def probe_network_10class(network, runner, env, key):
    binary_params = jax.tree_util.tree_map(lambda p: p > 0.5, runner.params)
    results = {i: {"logit": None} for i in range(10)}
    found_count, rng = 0, key
    for _ in range(150):
        rng, subkey = jax.random.split(rng); state = env.reset(subkey); label = int(state.current_label)
        if results[label]["logit"] is None:
            carry = network.initial_carry(subkey, 1)
            final_carry, output = network.apply({'params': binary_params, 'fixed_weights': runner.fixed_weights}, carry, state.obs)
            results[label] = {"logit": output[0], "rate": jnp.mean(final_carry[2])}; found_count += 1
        if found_count == 10: break
    return results

# ==================== [6] Main 主函数 ====================

def main(conf):
    conf = OmegaConf.merge({
        "seed": 0, "task": "mnist", "total_generations": 2000, "save_every": 50,
        "use_bio_probability": True, "bio_prob_mix_factor": 0.5,
        "network_type": "ConnSNN", "network_conf": {}, "es_conf": {},
        "episode_conf": {"max_episode_length": 1, "action_repeat": 1}
    }, conf)

    has_bio_data, bio_tau_Vm, bio_prob_matrix, l5e_indices = False, None, None, None
    num_neurons_loaded = 509

    if os.path.exists('neuron_physics.npz'):
        try:
            phys = np.load('neuron_physics.npz')
            num_neurons_loaded = int(phys['num_neurons'])
            bio_tau_Vm = tuple(phys['tau_Vm'].tolist())
            l5e_indices = get_l5_excitatory_indices('../dataset/mice_unnamed/neurons.csv.gz', num_neurons_loaded, 10)
            conf = OmegaConf.merge(conf, {"network_conf": {"num_neurons": num_neurons_loaded, "excitatory_ratio": float(phys['excitatory_ratio'])}})
            has_bio_data = True
            if conf.use_bio_probability and os.path.exists('init_probability.npy'):
                raw_p = np.load('init_probability.npy')
                mix = float(conf.bio_prob_mix_factor)
                bio_prob_matrix = mix * raw_p + (1.0 - mix) * 0.5
        except Exception as e: print(f"❌ 生物数据加载失败: {e}")

    conf = OmegaConf.merge({"project_name": f"E-SNN-{conf.task}", "run_name": f"EC {conf.seed} {time.strftime('%m-%d %H:%M')}"}, conf)
    es_conf = ESConfig(**conf.es_conf)

    if conf.task == "mnist":
        from envs.mnist_env import MnistEnv
        images, labels = load_mnist_data('train')
        # 预先按类别分组索引
        class_indices = [np.where(labels == i)[0] for i in range(10)]
        base_env = MnistEnv(images, labels, steps_pre_stim=100, steps_stim=200, steps_response=100, dt_ms=0.5)
        env = wrappers.VmapWrapper(base_env)
        es_conf = es_conf.replace(clip_action=False, normalize_obs=False)
        conf.network_type = "ConnSNN_Selected"
        conf.network_conf.update({"readout_indices": l5e_indices, "readout_start_step": 300, "readout_end_step": 400})
    else:
        env = envs.get_environment(conf.task, **conf.task_conf)
        env = wrappers.VmapWrapper(wrappers.EpisodeWrapper(env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat))

    network_cls = NETWORKS[conf.network_type]
    network = network_cls(out_dims=env.action_size, tau_Vm_vector=bio_tau_Vm, **conf.network_conf)
    optim = optax.chain(optax.scale_by_adam(mu_dtype=es_conf.p_dtype), optax.scale(-es_conf.lr))
    es_conf = es_conf.replace(network_cls=network, optim_cls=optim, env_cls=env)

    key_run, key_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runner = _runner_init(key_run, key_init, es_conf, init_prob_matrix=bio_prob_matrix)

    wandb.init(project=conf.project_name, group=conf.get("log_group", "Default"), name=conf.run_name, config=OmegaConf.to_container(conf))

    for step in tqdm(range(1, conf.total_generations + 1)):
        if conf.task == "mnist":
            # --- [动态采样平衡批次] ---
            batch_idx = np.concatenate([np.random.choice(class_indices[i], 8, replace=False) for i in range(10)])
            def _gen_spikes(imgs, s):
                probs = jnp.repeat(jnp.expand_dims(imgs * 0.1, 1), 400, axis=1) # 1000Hz * 0.5ms = 0.5 prob
                return jax.random.bernoulli(jax.random.PRNGKey(s), probs).astype(jnp.float32)
            train_spikes = _gen_spikes(images[batch_idx], step)
            train_lbls = jnp.array(labels[batch_idx], dtype=jnp.int32)
            
            runner, fit, eval_fit, grad = train_step_balanced(runner, train_spikes, train_lbls, es_conf, network)
            metrics = {"fitness": fit, "eval_fitness": eval_fit, "grad_norm": grad, "sparsity": mean_weight_abs(runner.params)}
        else:
            runner, metrics = _runnner_run(runner, es_conf)

        wandb.log(metrics, step=step)
        if conf.task == "mnist" and step % 20 == 0:
            res = probe_network_10class(network, runner, base_env, jax.random.PRNGKey(step))
            correct = sum([1 for i in range(10) if np.argmax(res[i]["logit"]) == i])
            tqdm.write(f"\n[Gen {step}] Probe Acc: {correct/10:.2f} | Fit: {metrics['fitness']:.3f}")

        if not (step % conf.save_every):
            save_obj_to_file(f"models/{conf.project_name}/{conf.run_name}/{step}", {"params": runner.params, "fixed_weights": runner.fixed_weights})

    return metrics

if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    main(_config)