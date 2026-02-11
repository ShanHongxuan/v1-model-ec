# ==================== [1] TF 预初始化 ====================
try:
    from utils.mnist_loader import load_mnist_data
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(">>> [System] 预加载 MNIST 数据以初始化 GPU 环境...")
    _ = load_mnist_data()
    print(">>> [System] 预加载完成。")
    _TF_PREINIT_SUCCESS = True
except (ImportError, ModuleNotFoundError):
    print(">>> [System] 警告: 未找到 MNIST 环境依赖。")
    _TF_PREINIT_SUCCESS = False
# =========================================================

from functools import partial
from typing import Any, Dict, Tuple
import time
import builtins
import os
import pickle

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

from networks import NETWORKS
from networks.conn_snn import ConnSNN_Selected
NETWORKS["ConnSNN_Selected"] = ConnSNN_Selected
from utils.functions import mean_weight_abs, save_obj_to_file

# 强制使用安全随机数实现
jax.config.update("jax_default_prng_impl", "unsafe_rbg")
builtins.bfloat16 = jnp.dtype("bfloat16").type

# ==================== 配置与状态类 ====================

@flax.struct.dataclass
class ESConfig:
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None
    pop_size:       int = 1024
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

# ==================== 辅助函数 ====================

def get_l5_excitatory_indices(csv_path, total_neurons, n_out=2):
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

# ==================== 训练逻辑 (Balanced Batch) ====================

@partial(jax.jit, static_argnums=(0,))
def evaluate_batch_fitness(network, params, fixed_weights, single_obs, single_label):
    pop_size = jax.tree_util.tree_leaves(params)[0].shape[0]
    obs_broadcast = jnp.repeat(jnp.expand_dims(single_obs, 0), pop_size, axis=0)
    carry = network.initial_carry(jax.random.PRNGKey(0), pop_size)
    
    vmapped_apply = jax.vmap(network.apply, in_axes=({'params': 0, 'fixed_weights': None}, 0, 0))
    _, output = vmapped_apply({'params': params, 'fixed_weights': fixed_weights}, carry, obs_broadcast)
    
    logits = output - jnp.max(output, axis=-1, keepdims=True)
    probs = jax.nn.softmax(logits)
    return probs[:, single_label]

@partial(jax.jit, donate_argnums=(0,), static_argnums=(3, 4))
def train_step_balanced(runner, batch_imgs, batch_lbls, es_conf, network):
    conf_pop_size = es_conf.pop_size
    conf_eval_size = es_conf.eval_size
    
    new_key, run_key = jax.random.split(runner.key)
    runner = runner.replace(key=new_key)
    
    train_params = _sample_bernoulli_parameter(run_key, runner.params, es_conf.network_dtype, (conf_pop_size - conf_eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf_eval_size, ))
    pop_params = jax.tree_util.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)
    
    def _scan_body(cum_fitness, idx):
        img = batch_imgs[idx] 
        lbl = batch_lbls[idx] 
        rewards = evaluate_batch_fitness(network, pop_params, runner.fixed_weights, img, lbl)
        return cum_fitness + rewards, None

    total_fitness, _ = jax.lax.scan(_scan_body, jnp.zeros(conf_pop_size), jnp.arange(batch_imgs.shape[0]))
    avg_fitness = total_fitness / batch_imgs.shape[0]
    
    fit_train, fit_eval = jnp.split(avg_fitness, [conf_pop_size - conf_eval_size])
    weight = _centered_rank_transform(fit_train)
    
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(es_conf.p_dtype)
        return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)

    grads = jax.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf_pop_size - conf_eval_size)]), runner.params, pop_params)
    
    updates, new_opt_state = es_conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)
    new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, es_conf.eps, 1 - es_conf.eps), new_params)
    
    runner = runner.replace(params=new_params, opt_state=new_opt_state)
    grad_norm = jnp.mean(jnp.abs(grads['kernel_h']))
    
    return runner, jnp.mean(fit_train), jnp.mean(fit_eval), grad_norm

# ==================== 初始化逻辑 ====================

@partial(jax.jit, static_argnums=(2,), donate_argnums=(3,))
def _runner_init(key: Any, network_init_key: Any, conf: ESConfig, init_prob_matrix: Any = None) -> RunnerState:
    env_reset_pool = conf.env_cls.reset(jax.random.split(key, conf.pop_size))

    network_variables = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
        {"params": network_init_key, "fixed_weights": network_init_key},
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs 
    )
    
    network_params = network_variables["params"]
    if init_prob_matrix is not None:
        bio_prob_jnp = jnp.array(init_prob_matrix, dtype=conf.p_dtype)
        def _init_mapper(path, param):
            if path[-1] == 'kernel_h':
                if param.shape == bio_prob_jnp.shape: return bio_prob_jnp
            return jnp.full_like(param, 0.5, conf.p_dtype)
        network_params = jax.tree_util.tree_map_with_path(_init_mapper, network_params)
    else:
        network_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_params)
    
    optim_state = conf.optim_cls.init(network_params)

    runner = RunnerState(
        key=key,
        normalizer_state=running_statistics.init_state(specs.Array((conf.env_cls.observation_size, ), jnp.float32)),
        env_reset_pool=env_reset_pool,
        params=network_params,
        fixed_weights=network_variables["fixed_weights"],
        opt_state=optim_state
    )
    return runner

# ==================== 自动测试逻辑 ====================

def run_full_test(network, params, fixed_weights, test_images, test_labels, prob_scale, batch_size=1000):
    """在测试集上运行推理并计算准确率"""
    
    inference_params = jax.tree_util.tree_map(lambda p: p > 0.5, params)
    
    #batch_inference 接受 prob_scale (input_hz * dt)
    @jax.jit
    def batch_inference(imgs):
        B = imgs.shape[0]
        # 使用传入的 prob_scale 生成脉冲
        probs = jnp.expand_dims(imgs * prob_scale, 1)
        probs = jnp.repeat(probs, 400, axis=1) # 400 steps
        spikes = jax.random.bernoulli(jax.random.PRNGKey(0), probs).astype(jnp.float32)
        
        init_carry = network.initial_carry(jax.random.PRNGKey(0), B)
        vmapped_apply = jax.vmap(network.apply, in_axes=({'params': None, 'fixed_weights': None}, 0, 0))
        _, output = vmapped_apply({'params': inference_params, 'fixed_weights': fixed_weights}, init_carry, spikes)
        return output

    num_test = test_images.shape[0]
    correct_count = 0
    num_batches = int(np.ceil(num_test / batch_size))
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_test)
        batch_imgs = test_images[start:end]
        batch_lbls = test_labels[start:end]
        
        logits = batch_inference(batch_imgs)
        preds = np.argmax(logits, axis=-1)
        correct_count += np.sum(preds == batch_lbls)
        
    accuracy = correct_count / num_test
    return accuracy

# ==================== 主程序 ====================

def main(conf):
    # 1. 默认配置
    conf = OmegaConf.merge({
        "seed": 0, "task": "mnist", "total_generations": 1000, "save_every": 50,
        "use_bio_probability": True, "bio_prob_mix_factor": 1.0,
        "network_type": "ConnSNN", "network_conf": {}, "es_conf": {},
        "episode_conf": {"max_episode_length": 1, "action_repeat": 1},
        "test_every": 100 
    }, conf)

    print("=== 二分类 (0 vs 1) 自动测试训练 (200Hz) ===")

    # 2. 数据准备
    print(">>> 加载数据...")
    train_imgs_all, train_lbls_all = load_mnist_data('train')
    test_imgs_all, test_lbls_all = load_mnist_data('test')

    def filter_01(imgs, lbls):
        mask = (lbls == 0) | (lbls == 1)
        return imgs[mask], lbls[mask]

    train_imgs, train_lbls = filter_01(train_imgs_all, train_lbls_all)
    test_imgs, test_lbls = filter_01(test_imgs_all, test_lbls_all)
    
    print(f"    Train Set: {len(train_lbls)} (0/1)")
    print(f"    Test Set:  {len(test_lbls)} (0/1)")

    # 3. 加载生物数据
    has_bio_data, bio_tau_Vm, bio_prob_matrix, l5e_indices = False, None, None, None
    num_neurons_loaded = 509

    if os.path.exists('neuron_physics.npz'):
        try:
            phys = np.load('neuron_physics.npz')
            num_neurons_loaded = int(phys['num_neurons'])
            bio_tau_Vm = tuple(phys['tau_Vm'].tolist())
            # 只取 2 个 L5E
            l5e_indices = get_l5_excitatory_indices('../dataset/mice_unnamed/neurons.csv.gz', num_neurons_loaded, 2)
            
            conf = OmegaConf.merge(conf, {"network_conf": {"num_neurons": num_neurons_loaded, "excitatory_ratio": float(phys['excitatory_ratio'])}})
            has_bio_data = True
            if conf.use_bio_probability and os.path.exists('init_probability.npy'):
                raw_prob = np.load('init_probability.npy')
                mix = float(conf.bio_prob_mix_factor)
                bio_prob_matrix = mix * raw_prob + (1.0 - mix) * 0.5
                print(f"✅ 生物概率混合完成 (Mix={mix})")
        except Exception as e: print(f"❌ 加载失败: {e}")

    # 4. 配置与环境
    conf = OmegaConf.merge({"project_name": f"E-SNN-2Class", "run_name": f"2C {conf.seed} {time.strftime('%H:%M')}"}, conf)
    
    # 定义物理参数
    INPUT_HZ = 200.0  
    DT = 0.5          
    PROB_SCALE = INPUT_HZ * DT / 1000.0 # 计算每步概率因子
    
    K_IN = 2.0
    K_H = 0.1
    K_OUT = 20.0
    
    conf.network_type = "ConnSNN_Selected"
    conf.network_conf.update({
        "readout_indices": l5e_indices, 
        "readout_start_step": 300, 
        "readout_end_step": 400,
        "K_in": K_IN, "K_h": K_H, "K_out": K_OUT, "dt": DT
    })
    
    from envs.mnist_env import MnistEnv
    base_env = MnistEnv(
        train_imgs[:1], train_lbls[:1], 
        steps_pre_stim=100, steps_stim=200, steps_response=100, 
        dt_ms=DT, input_hz=INPUT_HZ # <--- 传入
    )
    base_env.action_size = 2
    env = wrappers.VmapWrapper(base_env)
    
    es_conf = ESConfig(**conf.es_conf)
    es_conf = es_conf.replace(clip_action=False, normalize_obs=False)

    # 5. 网络与优化器
    network_cls = NETWORKS[conf.network_type]
    network = network_cls(out_dims=2, tau_Vm_vector=bio_tau_Vm, **conf.network_conf)
    
    optim = optax.chain(optax.scale_by_adam(mu_dtype=es_conf.p_dtype), optax.scale(-es_conf.lr))
    es_conf = es_conf.replace(network_cls=network, optim_cls=optim, env_cls=env)

    # 6. 初始化 Runner
    key_run, key_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runner = _runner_init(key_run, key_init, es_conf, init_prob_matrix=bio_prob_matrix)

    wandb.init(project=conf.project_name, group=conf.get("log_group", "Default"), name=conf.run_name, config=OmegaConf.to_container(conf))

    # 7. 训练循环
    print(f">>> 开始训练 (InputHz={INPUT_HZ}, ProbScale={PROB_SCALE:.4f})...")
    
    idx_0 = np.where(train_lbls == 0)[0]
    idx_1 = np.where(train_lbls == 1)[0]
    
    def get_balanced_batch(seed, batch_size=32):
        n = batch_size // 2
        sel_0 = np.random.choice(idx_0, n, replace=False)
        sel_1 = np.random.choice(idx_1, n, replace=False)
        
        batch_imgs = np.concatenate([train_imgs[sel_0], train_imgs[sel_1]])
        batch_lbls = np.concatenate([train_lbls[sel_0], train_lbls[sel_1]])
        
        #使用 PROB_SCALE 生成泊松脉冲
        probs = jnp.expand_dims(batch_imgs * PROB_SCALE, 1)
        probs = jnp.repeat(probs, 400, axis=1)
        spikes = jax.random.bernoulli(jax.random.PRNGKey(seed), probs).astype(jnp.float32)
        
        return spikes, jnp.array(batch_lbls, dtype=jnp.int32)

    pbar = tqdm(range(1, conf.total_generations + 1))
    
    for step in pbar:
        spikes, lbls = get_balanced_batch(step, batch_size=32)
        
        runner, fit, eval_fit, grad = train_step_balanced(runner, spikes, lbls, es_conf, network)
        metrics = {"fitness": fit, "grad_norm": grad, "sparsity": mean_weight_abs(runner.params)}

        # --- [自动测试] ---
        if step % conf.test_every == 0:
            test_acc = run_full_test(network, runner.params, runner.fixed_weights, test_imgs, test_lbls, PROB_SCALE)
            metrics["test_accuracy"] = test_acc
            tqdm.write(f"\n[Gen {step}] TrainFit: {fit:.3f} | TestAcc: {test_acc*100:.2f}%")

        wandb.log(metrics, step=step)
        pbar.set_description(f"Fit:{fit:.3f} | Grad:{grad:.5f}")

        if not (step % conf.save_every):
            save_obj_to_file(f"models/{conf.project_name}/{conf.run_name}/{step}", {"params": runner.params, "fixed_weights": runner.fixed_weights})

    return metrics

if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    main(_config)