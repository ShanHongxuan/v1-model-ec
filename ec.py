# ==================== [TF 预初始化] ====================
try:
    from utils.mnist_loader import load_mnist_data
    print(">>> 正在预加载 MNIST 数据以初始化 TensorFlow GPU 环境...")
    _ = load_mnist_data()
    print(">>> MNIST 数据预加载完成。")
    _TF_PREINIT_SUCCESS = True
except (ImportError, ModuleNotFoundError):
    print(">>> 未找到 MNIST 环境，将仅支持 Brax 任务。")
    _TF_PREINIT_SUCCESS = False
# =======================================================

from functools import partial
from typing import Any, Dict, Tuple
import time
import builtins
import os

import jax
import jax.numpy as jnp
import numpy as np

import flax
import optax

from brax import envs
from brax.training.acme import running_statistics
from brax.training.acme import specs

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import optuna

from networks import NETWORKS
from utils.functions import mean_weight_abs, finitemean, save_obj_to_file

jax.config.update("jax_default_prng_impl", "unsafe_rbg")
builtins.bfloat16 = jnp.dtype("bfloat16").type


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
    
    # [新增] 控制开关
    clip_action:   bool = True  # 默认开启 (Brax需要)
    normalize_obs: bool = True  # 默认开启 (Brax需要)


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


def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    shape = x.shape
    x     = x.ravel()
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


# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig) -> PopulationState:
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))

    # [修改] 根据配置决定是否归一化观测
    if conf.normalize_obs:
        obs_norm = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
    else:
        obs_norm = pop.env_states.obs

    new_network_states, act = vmapped_apply({"params": pop.network_params, "fixed_weights": runner.fixed_weights}, pop.network_states, obs_norm)
    assert act.dtype == conf.network_dtype

    # [修改] 根据配置决定是否截断动作
    if conf.clip_action:
        act = jnp.clip(act, -1, 1)

    if act.dtype != conf.action_dtype:
        act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)

    new_env_states = conf.env_cls.step(pop.env_states, act)

    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward
    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n   + 1,                  pop.fitness_n)
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)

    def _where_done(x, y):
        done = new_env_states.done
        done = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)

    # 对于同步训练，这里主要用于兼容接口
    new_env_states = jax.tree_util.tree_map(_where_done, runner.env_reset_pool, new_env_states)

    return pop.replace(
        network_states=new_network_states,
        env_states=new_env_states,
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )


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
    network_fixed_weights = network_variables["fixed_weights"]

    if init_prob_matrix is not None:
        bio_prob_jnp = jnp.array(init_prob_matrix, dtype=conf.p_dtype)
        def _init_mapper(path, param):
            if path[-1] == 'kernel_h':
                if param.shape != bio_prob_jnp.shape:
                    return jnp.full_like(param, 0.5, conf.p_dtype)
                return bio_prob_jnp
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
        fixed_weights=network_fixed_weights,
        opt_state=optim_state
    )
    return runner


@partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def _runnner_run(runner: RunnerState, conf: ESConfig) -> Tuple[RunnerState, Dict]:
    metrics = {}
    new_key, run_key, carry_key, env_key = jax.random.split(runner.key, 4)
    runner = runner.replace(key=new_key)

    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype, (conf.pop_size - conf.eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))
    network_params = jax.tree_map(lambda train, eval: jnp.concatenate([train, eval], axis=0), train_params, eval_params)

    def _split_fitness(x):
        return jnp.split(x, [conf.pop_size - conf.eval_size, ])

    # [同步训练逻辑]
    common_env_key = jax.random.split(env_key, 1)[0]
    broadcasted_keys = jnp.repeat(jnp.expand_dims(common_env_key, 0), conf.pop_size, axis=0)
    synchronized_env_states = conf.env_cls.reset(broadcasted_keys)
    
    pop = PopulationState(
        network_params=network_params,
        network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        env_states=synchronized_env_states,
        fitness_totrew=jnp.zeros(conf.pop_size),
        fitness_sum=jnp.zeros(conf.pop_size),
        fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
    )

    def _eval_stop_cond(p: PopulationState) -> jnp.ndarray:
        return ~jnp.all(p.fitness_n >= 1)

    pop = jax.lax.while_loop(_eval_stop_cond, (lambda p: _evaluate_step(p, runner, conf)), pop)

    # 仅当启用归一化时更新 Normalizer
    if conf.warmup_steps <= 0 and conf.normalize_obs:
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, pop.env_states.obs))

    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(pop.network_states))

    fitness, eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)

    weight = _centered_rank_transform(fitness)
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
        # [修复] 显式转换 theta
        return -jnp.mean(w * (theta.astype(jnp.float32) - p), axis=0)

    grads = jax.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]), runner.params, pop.network_params)

    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)
    new_params = jax.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)

    runner = runner.replace(params=new_params, opt_state=new_opt_state)

    metrics.update({
        "fitness":        jnp.mean(fitness),
        "eval_fitness":   jnp.mean(eval_fitness),
        "sparsity":       mean_weight_abs(new_params)
    })
    return runner, metrics


def main(conf):
    conf = OmegaConf.merge({
        "seed": 0,
        "task": "humanoid",
        "task_conf": {},
        "episode_conf": {
            "max_episode_length": 1000,
            "action_repeat": 1
        },
        "total_generations": 1000,
        "save_every": 50,
        "use_bio_probability": True,
        "bio_prob_mix_factor": 1.0,
        "network_type": "ConnSNN",
        "network_conf": {},
        "es_conf": {}
    }, conf)

    print(">>> 正在检查并加载生物数据...")
    has_bio_data = False
    bio_tau_Vm = None
    bio_prob_matrix = None
    
    if os.path.exists('neuron_physics.npz'):
        try:
            physics_data = np.load('neuron_physics.npz')
            bio_num_neurons = int(physics_data['num_neurons'])
            bio_exc_ratio = float(physics_data['excitatory_ratio'])
            bio_tau_Vm = physics_data['tau_Vm']
            
            conf = OmegaConf.merge(conf, {
                "network_conf": {
                    "num_neurons": bio_num_neurons,
                    "excitatory_ratio": bio_exc_ratio,
                }
            })
            has_bio_data = True
            print(f"✅ 成功加载物理参数: 神经元={bio_num_neurons}")

            if conf.use_bio_probability:
                if os.path.exists('init_probability.npy'):
                    raw_prob_matrix = np.load('init_probability.npy')
                    mix = float(conf.bio_prob_mix_factor)
                    bio_prob_matrix = mix * raw_prob_matrix + (1.0 - mix) * 0.5
                    print(f"✅ 已应用混合因子 {mix}")
                else:
                    print("⚠️ 未找到 init_probability.npy")
            else:
                bio_prob_matrix = None
        except Exception as e:
            print(f"❌ 加载出错: {e}")
            has_bio_data = False
    else:
        print("ℹ️ 未找到 neuron_physics.npz")

    conf = OmegaConf.merge({
        "project_name": f"E-SNN-{conf.task}",
        "run_name":     f"EC {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    es_conf = ESConfig(**conf.es_conf)

    # [Task Logic]
    if conf.task == "mnist":
        if not _TF_PREINIT_SUCCESS:
            raise RuntimeError("MNIST Env 缺失")
        from envs.mnist_env import MnistEnv
        from utils.mnist_loader import load_mnist_data
        
        mnist_images, mnist_labels = load_mnist_data('train')
        dt_ms = conf.network_conf.get('dt', 0.5)
        snn_steps = 200 
        
        base_env = MnistEnv(images=mnist_images, labels=mnist_labels, 
                            presentation_steps=snn_steps, dt_ms=dt_ms)
        env = envs.wrappers.VmapWrapper(base_env)
        conf.episode_conf.max_episode_length = 1 
        
        # [修改] 针对 MNIST 的关键配置
        # 禁用动作截断，允许 K_out 发挥作用
        es_conf = es_conf.replace(clip_action=False)
        # 禁用观测归一化，保持 0/1 稀疏性
        es_conf = es_conf.replace(normalize_obs=False)
        
    else:
        # Brax 任务保持默认 (Clip=True, Norm=True)
        env = envs.get_environment(conf.task, **conf.task_conf)
        env = envs.wrappers.EpisodeWrapper(env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat)
        env = envs.wrappers.VmapWrapper(env)

    # Network
    network_cls = NETWORKS[conf.network_type]
    network_kwargs = {
        "out_dims": env.action_size,
        "neuron_dtype": es_conf.network_dtype,
        **conf.network_conf
    }
    if has_bio_data and bio_tau_Vm is not None:
        network_kwargs["tau_Vm_vector"] = tuple(bio_tau_Vm.tolist())

    network = network_cls(**network_kwargs)

    optim = optax.chain(
        optax.scale_by_adam(mu_dtype=es_conf.p_dtype),
        (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
        optax.scale(-es_conf.lr)
    )

    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=env
    )

    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    
    runner = _runner_init(key_run, key_network_init, es_conf, init_prob_matrix=bio_prob_matrix)

    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)

    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) E-SNN-{conf.task}", group=conf.log_group, name=conf.run_name, config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf))

    for step in tqdm(range(1, conf.total_generations + 1)):
        runner, metrics = _runnner_run(runner, es_conf)
        metrics = jax.device_get(metrics)
        wandb.log(metrics, step=step)

        if not (step % conf.save_every):
            fn = conf.save_model_path + str(step)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runner.normalizer_state,
                    fixed_weights=runner.fixed_weights,
                    params=runner.params
                )
            ))
            wandb.save(fn)

    return metrics

if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    main(_config)