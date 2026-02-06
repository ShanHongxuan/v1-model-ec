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

# ==================== [6] 探针 (10分类版) ====================
def probe_network(network, runner, env, key):
    binary_params = jax.tree_util.tree_map(lambda p: p > 0.5, runner.params)
    variables = {'params': binary_params, 'fixed_weights': runner.fixed_weights}
    
    # 存储结果: 0-9 每个数字的 Logit, Prob, Rate
    results = {i: {"logit": None, "prob": None, "rate": None} for i in range(10)}
    found_count = 0
    
    rng = key
    # 尝试多次以覆盖所有数字
    for _ in range(100):
        rng, subkey = jax.random.split(rng)
        state = env.reset(subkey) 
        label = int(state.current_label)
        
        if results[label]["logit"] is None:
            carry = network.initial_carry(subkey, 1)
            # 运行网络
            final_carry, output = network.apply(variables, carry, state.obs)
            
            logits = output[0]
            probs = jax.nn.softmax(logits)
            avg_rate = jnp.mean(final_carry[2]) 
            
            results[label]["logit"] = logits
            results[label]["prob"] = probs
            results[label]["rate"] = avg_rate
            
            found_count += 1
            
        if found_count == 10: # 找齐了所有数字
            break
            
    return results

# ==================== [7] 主程序 ====================
def main():
    print("=== 10分类平衡批次训练 (Balanced Batch Training) ===")
    
    # 1. 数据准备
    print(">>> 准备 10 类平衡数据集...")
    images, labels = load_mnist_data('train')
    
    # 按类别整理图片
    class_images = []
    for i in range(10):
        mask = labels == i
        imgs = images[mask]
        class_images.append(imgs)
        print(f"    Class {i}: {len(imgs)} images")
    
    # 构造验证/训练批次：每个类别取 8 张，共 80 张
    # 80 张对于显存来说仍然很轻松 (2048 * 80 * 196 * 4bytes / 1024^2 ≈ 122MB)
    imgs_per_class = 8
    val_imgs_list = []
    val_lbls_list = []
    
    for i in range(10):
        val_imgs_list.append(class_images[i][:imgs_per_class])
        val_lbls_list.append(np.full(imgs_per_class, i))
        
    val_imgs = np.concatenate(val_imgs_list)
    val_labels = np.concatenate(val_lbls_list)
    
    # 2. 配置 (继承自二分类成功的经验)
    K_IN = 2.0     # 温和输入
    K_H = 0.05     # 弱递归
    K_OUT = 0.3  # 强输出
    
    conf = OmegaConf.create({
        "seed": 42,
        "pop_size": 1024,
        "lr": 0.1,        # 稍微降低学习率，因为任务变难了
        "total_generations": 500, # 增加代数
        "batch_size": 80,  # 10 classes * 8 images
        
        "eval_size": 128,
        "eps": 0.001,
        
        "network_conf": {
            "num_neurons": 509, "excitatory_ratio": 0.76,
            "K_in": K_IN, "K_h": K_H, "K_out": K_OUT, "dt": 0.5
        },
        "use_bio": True, "mix": 0.5
    })
    
    # 3. 环境与网络
    snn_steps = 200
    
    # 初始化环境 (使用全部数据)
    base_env = MnistEnv(images, labels, presentation_steps=snn_steps, input_hz=100.0, dt_ms=0.5)
    # [关键] 设为 10 分类
    base_env.action_size = 10 
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

    # [关键] out_dims = 10
    network = network_cls(out_dims=10, tau_Vm_vector=tau_vec, **conf.network_conf)
    
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
    env_pool = env.reset(jax.random.split(key_init, conf.pop_size))
    
    runner = RunnerState(
        key=key_run,
        normalizer_state=running_statistics.init_state(specs.Array((196,), jnp.float32)),
        env_reset_pool=env_pool,
        params=net_params,
        fixed_weights=vars_init['fixed_weights'],
        opt_state=opt_state
    )

    # ================= [关键] 自定义单步训练函数 =================
    @jax.jit
    def evaluate_batch(params, fixed_weights, batch_obs, batch_labels):
        pop_size = jax.tree_util.tree_leaves(params)[0].shape[0]
        
        obs_broadcast = jnp.repeat(jnp.expand_dims(batch_obs, 0), pop_size, axis=0)
        carry = network.initial_carry(jax.random.PRNGKey(0), pop_size)
        
        vmapped_apply = jax.vmap(
            network.apply, 
            in_axes=({'params': 0, 'fixed_weights': None}, 0, 0)
        )
        
        _, output = vmapped_apply(
            {'params': params, 'fixed_weights': fixed_weights}, 
            carry, 
            obs_broadcast
        )
        
        # Softmax
        logits = output - jnp.max(output, axis=-1, keepdims=True)
        probs = jax.nn.softmax(logits)
        rewards = probs[:, batch_labels] 
        
        return rewards

    @partial(jax.jit, donate_argnums=(0,))
    def train_step_balanced(runner, batch_imgs, batch_lbls):
        new_key, run_key = jax.random.split(runner.key)
        runner = runner.replace(key=new_key)
        
        train_params = _sample_bernoulli_parameter(run_key, runner.params, es_conf.network_dtype, (conf.pop_size - conf.eval_size, ))
        eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))
        pop_params = jax.tree_util.tree_map(lambda t, e: jnp.concatenate([t, e], axis=0), train_params, eval_params)
        
        def _scan_body(cum_fitness, idx):
            img = batch_imgs[idx] 
            lbl = batch_lbls[idx] 
            rewards = evaluate_batch(pop_params, runner.fixed_weights, img, lbl)
            return cum_fitness + rewards, None

        total_fitness, _ = jax.lax.scan(_scan_body, jnp.zeros(conf.pop_size), jnp.arange(batch_imgs.shape[0]))
        
        avg_fitness = total_fitness / batch_imgs.shape[0]
        
        fit_train, fit_eval = jnp.split(avg_fitness, [conf.pop_size - conf.eval_size])
        weight = _centered_rank_transform(fit_train)
        
        def _nes_grad(p, theta):
            w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
            return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)

        grads = jax.tree_util.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]), runner.params, pop_params)
        
        updates, new_opt_state = es_conf.optim_cls.update(grads, runner.opt_state, runner.params)
        new_params = optax.apply_updates(runner.params, updates)
        new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)
        
        runner = runner.replace(params=new_params, opt_state=new_opt_state)
        grad_norm = jnp.mean(jnp.abs(grads['kernel_h']))
        
        return runner, jnp.mean(fit_train), jnp.mean(fit_eval), grad_norm

    # --- 训练循环 ---
    print(">>> 开始训练 (Batch Size = 80, Balanced 10-Class)...")
    pbar = tqdm(range(1, conf.total_generations + 1))
    
    rng_data = jax.random.PRNGKey(999)
    def make_poisson_batch(key, imgs, lbls):
        probs = imgs * (1000.0 * 0.5 / 1000.0) # 1000Hz
        probs = jnp.expand_dims(probs, 1) 
        probs = jnp.repeat(probs, snn_steps, axis=1)
        spikes = jax.random.bernoulli(key, probs).astype(jnp.float32)
        return spikes, lbls

    train_spikes, train_lbls = make_poisson_batch(rng_data, val_imgs, val_labels)

    for step in pbar:
        runner, fit, eval_fit, grad = train_step_balanced(runner, train_spikes, train_lbls.astype(jnp.int32))
        
        desc = f"Fit:{fit:.3f} | Eval:{eval_fit:.3f} | Grad:{grad:.5f}"
        
        # 探针: 每 20 代检查一次 (因为10分类检查比较耗时)
        if step % 20 == 0:
            results = probe_network(network, runner, base_env, jax.random.PRNGKey(step))
            
            # 打印 0, 1, 2 的情况作为代表
            tqdm.write(f"\n[Gen {step} Diagnostic]")
            
            # 计算 Top-1 准确率 (基于采样的 10 个样本)
            correct_count = 0
            found_count = 0
            for i in range(10):
                res = results[i]
                if res["logit"] is not None:
                    found_count += 1
                    pred_label = np.argmax(res["logit"])
                    if pred_label == i:
                        correct_count += 1
                        
                    # 打印前 3 个类别的详细信息
                    if i < 3:
                        tqdm.write(f"  Input {i} -> Rate:{res['rate']:.3f} | Prob({i}):{res['prob'][i]:.4f} | Pred:{pred_label}")
            
            if found_count > 0:
                acc = correct_count / found_count
                tqdm.write(f"  >>> Probe Accuracy: {acc:.2f} ({correct_count}/{found_count})")

        pbar.set_description(desc)

if __name__ == "__main__":
    main()