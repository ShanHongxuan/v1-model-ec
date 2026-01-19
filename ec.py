# ==================== [新增] 预初始化 TF 和数据加载 ====================
# 将这个块放在文件的绝对顶部，在所有其他 import 之前
try:
    # 尝试导入 mnist_loader，这将触发 TensorFlow 的 GPU 初始化
    from utils.mnist_loader import load_mnist_data
    # 运行一次加载，确保所有 TF/GPU 设置都已完成
    print(">>> 正在预加载 MNIST 数据以初始化 TensorFlow GPU 环境...")
    _ = load_mnist_data()
    print(">>> MNIST 数据预加载完成。")
    _TF_PREINIT_SUCCESS = True
except (ImportError, ModuleNotFoundError):
    print(">>> 未找到 MNIST 环境，将仅支持 Brax 任务。")
    _TF_PREINIT_SUCCESS = False

from functools import partial
from typing import Any, Dict, Tuple
import time
import builtins
import os

import jax
import jax.numpy as jnp
import numpy as np  # [新增] 用于加载生物数据

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


# Use RBG generator for less memory consumption
# Default RNG needs 2*N extra memory, while RBG needs none, when generating array with size N
# https://jax.readthedocs.io/en/latest/jax.random.html
jax.config.update("jax_default_prng_impl", "unsafe_rbg")

# Hack for resolving bfloat16 pickling issue https://github.com/google/jax/issues/8505
builtins.bfloat16 = jnp.dtype("bfloat16").type


@flax.struct.dataclass
class ESConfig:
    # Network, optim & env class
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None

    # [Hyperparameters] ES
    pop_size:       int = 1024
    lr:           float = 0.15

    eps:          float = 1e-3

    weight_decay: float = 0.    # For sparsity regularization

    # [Hyperparameters] Warmup
    warmup_steps:   int = 0

    # [Hyperparameters] Eval
    eval_size:      int = 128

    # [Computing] Data types
    action_dtype: Any   = jnp.float32  # brax uses fp32

    p_dtype:       Any  = jnp.float32
    network_dtype: Any  = jnp.float32


@flax.struct.dataclass
class RunnerState:
    key: Any
    # Normalizer
    normalizer_state: running_statistics.RunningStatisticsState
    # Env reset state pool
    env_reset_pool: Any
    # Network optimization
    params:        Any
    fixed_weights: Any
    opt_state:     Any


@flax.struct.dataclass
class PopulationState:
    # Network
    network_params: Any
    network_states: Any
    # Env
    env_states:     Any
    # Fitness
    fitness_totrew: jnp.ndarray
    fitness_sum:    jnp.ndarray
    fitness_n:      jnp.ndarray


def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Centered rank from: https://arxiv.org/pdf/1703.03864.pdf"""

    shape = x.shape
    x     = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)


def _sample_bernoulli_parameter(key: Any, params: Any, sampling_dtype: Any, batch_size: Tuple = ()) -> Any:
    """Sample parameters from Bernoulli distribution. """

    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)

    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), sampling_dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

    return noise


def _deterministic_bernoulli_parameter(params: Any, batch_size: Tuple = ()) -> Any:
    """Deterministic evaluation, using p > 0.5 as True, p <= 0.5 as False"""

    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)


# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig) -> PopulationState:
    # step env
    # NOTE: vmapping apply for multiple set of parameters (broadcast fixed weights)
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))

    obs_norm                = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
    new_network_states, act = vmapped_apply({"params": pop.network_params, "fixed_weights": runner.fixed_weights}, pop.network_states, obs_norm)
    assert act.dtype == conf.network_dtype   # Sanity check, avoid silent promotion

    act = jnp.clip(act, -1, 1)  # brax do not clip actions internally.

    # NOTE: Cast type and avoid NaNs, set them to 0
    if act.dtype != conf.action_dtype:
        act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)

    new_env_states = conf.env_cls.step(pop.env_states, act)

    # calculate episodic rewards
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward

    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n   + 1,                  pop.fitness_n)
    # clear tot rew
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)

    # reset done envs
    # Reference: brax / envs / wrapper.py
    def _where_done(x, y):
        done = new_env_states.done
        done = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)

    new_env_states = jax.tree_map(_where_done, runner.env_reset_pool, new_env_states)

    return pop.replace(
        # Network
        network_states=new_network_states,
        # Env
        env_states=new_env_states,
        # Fitness
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )


# [修改] 增加了 init_prob_matrix 参数
@partial(jax.jit, static_argnums=(2,), donate_argnums=(3,))
def _runner_init(key: Any, network_init_key: Any, conf: ESConfig, init_prob_matrix: Any = None) -> RunnerState:
    # split run keys for initializing env
    key, env_init_key = jax.random.split(key)

    # init env
    env_reset_pool = conf.env_cls.reset(jax.random.split(env_init_key, conf.pop_size))

    # init network params + opt state
    network_variables = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
        {"params": network_init_key, "fixed_weights": network_init_key},
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs
    )
    network_params = network_variables["params"]
    network_fixed_weights = network_variables["fixed_weights"]

    # [修改] 设置初始概率
    if init_prob_matrix is not None:
        # 将 numpy 矩阵转为 jax array
        bio_prob_jnp = jnp.array(init_prob_matrix, dtype=conf.p_dtype)
        
        # 定义映射函数：如果是 kernel_h (递归权重)，则使用 bio 矩阵；否则使用 0.5
        def _init_mapper(path, param):
            if path[-1] == 'kernel_h':
                if param.shape != bio_prob_jnp.shape:
                    return jnp.full_like(param, 0.5, conf.p_dtype)
                
                # [修改] 使用配置中的混合因子
                mix_factor = conf.bio_prob_mix_factor
                mixed_prob = mix_factor * bio_prob_jnp + (1.0 - mix_factor) * 0.5
                
                return mixed_prob
            
            return jnp.full_like(param, 0.5, conf.p_dtype)

        # 使用 tree_map_with_path 进行精细初始化
        network_params = jax.tree_util.tree_map_with_path(_init_mapper, network_params)
    else:
        # 原有逻辑：全部初始化为 0.5
        network_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_params)
    
    optim_state = conf.optim_cls.init(network_params)

    # runner state
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

    # split keys for this run
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # Generate params with bernoulli distribution
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype, (conf.pop_size - conf.eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))

    network_params = jax.tree_map(lambda train, eval: jnp.concatenate([train, eval], axis=0), train_params, eval_params)

    # Split the eval and train fitness, returns [fitness, eval_fitness]
    def _split_fitness(x):
        return jnp.split(x, [conf.pop_size - conf.eval_size, ])

    # Initialize population
    pop = PopulationState(
        # Network
        network_params=network_params,
        network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        # Env
        env_states=runner.env_reset_pool,
        # Fitness
        fitness_totrew=jnp.zeros(conf.pop_size),
        fitness_sum=jnp.zeros(conf.pop_size),
        fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
    )

    # (PNN) Run some steps to warm up states
    if conf.warmup_steps > 0:
        pop, _ = jax.lax.scan(lambda p, x: (_evaluate_step(p, runner, conf), None), pop, None, length=conf.warmup_steps)

        warmup_fitness, warmup_eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)
        metrics.update({
            "warmup_fitness":      finitemean(warmup_fitness),
            "warmup_eval_fitness": finitemean(warmup_eval_fitness)
        })

        # (PNN) Update normalizer using warmup data
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, pop.env_states.obs))
        # (PNN) Reset envs + Clear fitness
        pop = pop.replace(
            # Env
            env_states=runner.env_reset_pool,
            # Fitness
            fitness_totrew=jnp.zeros(conf.pop_size),
            fitness_sum=jnp.zeros(conf.pop_size),
            fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
        )

    # Evaluate
    def _eval_stop_cond(p: PopulationState) -> jnp.ndarray:
        # Stop when finished
        return ~jnp.all(p.fitness_n >= 1)

    pop = jax.lax.while_loop(_eval_stop_cond, (lambda p: _evaluate_step(p, runner, conf)), pop)

    # Update normalizer using terminal states
    # FIXME: May be biased towards states near episode terminal
    if conf.warmup_steps <= 0:
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, pop.env_states.obs))

    # Calculate population metrics
    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(pop.network_states))

    # Calculate fitness
    fitness, eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)

    # Reconstruct noise using network parameters
    # NOTE: use -grads to do gradient ascent
    weight = _centered_rank_transform(fitness)
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)

        return -jnp.mean(w * (theta - p), axis=0)

    grads = jax.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]), runner.params, pop.network_params)

    # Gradient step
    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)

    # Clip to Bernoulli range with exploration
    new_params = jax.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)

    runner = runner.replace(
        params=new_params,
        opt_state=new_opt_state
    )

    # Metrics
    metrics.update({
        "fitness":        jnp.mean(fitness),
        "eval_fitness":   jnp.mean(eval_fitness),

        "sparsity": mean_weight_abs(new_params)
    })
    return runner, metrics


def main(conf):
    # [修改] 在这里明确声明 use_bio_probability
    conf = OmegaConf.merge({
        # Task
        "seed": 0,
        "task": "humanoid",
        "task_conf": {
        },
        "episode_conf": {
            "max_episode_length": 1000,
            "action_repeat": 1
        },

        # Train & Checkpointing
        "total_generations": 1000,
        "save_every": 50,

        # [新增开关] 明确在这里定义
        "use_bio_probability": True,
        "bio_prob_mix_factor": 1.0,

        # Network
        "network_type": "ConnSNN",
        "network_conf": {},

        # ES hyperparameter (see ESConfig)
        "es_conf": {}
    }, conf)

    # ==================== [修改] 加载生物预处理数据逻辑 ====================
    print(">>> 正在检查并加载生物数据 (neuron_physics.npz, init_probability.npy)...")
    
    has_bio_data = False
    bio_tau_Vm = None
    bio_prob_matrix = None
    
    # 1. 总是尝试加载物理参数
    if os.path.exists('neuron_physics.npz'):
        try:
            physics_data = np.load('neuron_physics.npz')
            bio_num_neurons = int(physics_data['num_neurons'])
            bio_exc_ratio = float(physics_data['excitatory_ratio'])
            bio_tau_Vm = physics_data['tau_Vm']
            
            # 基础网络参数覆盖
            conf = OmegaConf.merge(conf, {
                "network_conf": {
                    "num_neurons": bio_num_neurons,
                    "excitatory_ratio": bio_exc_ratio,
                }
            })
            has_bio_data = True
            print(f"✅ 成功加载物理参数: 神经元={bio_num_neurons}, 兴奋性比例={bio_exc_ratio:.2f}")

            # 2. 根据开关决定是否加载连接概率矩阵
            # [重要] 这里的 conf.use_bio_probability 现在肯定存在了
            if conf.use_bio_probability:
                if os.path.exists('init_probability.npy'):
                    bio_prob_matrix = np.load('init_probability.npy')
                    print(f"✅ [use_bio_probability=True] 已加载生物概率矩阵: {bio_prob_matrix.shape}")
                else:
                    print("⚠️ [use_bio_probability=True] 但未找到 init_probability.npy 文件。将回退到默认 0.5。")
            else:
                print("ℹ️ [use_bio_probability=False] 用户选择忽略生物概率矩阵。将使用默认 0.5 初始化。")
                bio_prob_matrix = None

        except Exception as e:
            print(f"❌ 加载生物数据时出错: {e}")
            has_bio_data = False
    else:
        print("ℹ️ 未找到 neuron_physics.npz，使用默认配置。")
    # ======================================================================

    # Naming
    conf = OmegaConf.merge({
        "project_name": f"E-SNN-{conf.task}",
        "run_name":     f"EC {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    # ES Config
    es_conf = ESConfig(**conf.es_conf)

    print(OmegaConf.to_yaml(conf))
    print(es_conf)

    if conf.task == "mnist":
        if not _TF_PREINIT_SUCCESS:
            raise RuntimeError("MNIST 任务需要 tensorflow_datasets，但预初始化失败。")
        
        from envs.mnist_env import MnistEnv
        from utils.mnist_loader import load_mnist_data
        
        # [修改] 再次加载数据（这次会从缓存中读取，很快）并传给环境
        mnist_images, mnist_labels = load_mnist_data('train')
        base_env = MnistEnv(images=mnist_images, labels=mnist_labels, presentation_steps=50)
        
        env = envs.wrappers.VmapWrapper(base_env) 
        
        conf.episode_conf.action_repeat = 1
        conf.episode_conf.max_episode_length = 50 
        
    else:
        # 原有的 Brax 任务逻辑
        env = envs.get_environment(conf.task, **conf.task_conf)
        env = envs.wrappers.EpisodeWrapper(env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat)
        env = envs.wrappers.VmapWrapper(env)

    # create network cls
    network_cls = NETWORKS[conf.network_type]
    
    # 准备网络参数
    network_kwargs = {
        "out_dims": env.action_size,
        "neuron_dtype": es_conf.network_dtype,
        **conf.network_conf
    }
    
    # [新增] 如果有生物数据，传入 tau_Vm_vector
    # 注意：前提是 ConnSNN 类已经被修改为接受 tau_Vm_vector 参数
    if has_bio_data and bio_tau_Vm is not None:
        network_kwargs["tau_Vm_vector"] = tuple(bio_tau_Vm.tolist())

    network = network_cls(**network_kwargs)

    # create optim cls
    optim = optax.chain(
        optax.scale_by_adam(mu_dtype=es_conf.p_dtype),
        (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
        optax.scale(-es_conf.lr)
    )

    # [initialize]
    # initialize cls in es conf
    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=env
    )

    # runner state
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    
    # [修改] 调用 runner_init，传入概率矩阵
    runner = _runner_init(key_run, key_network_init, es_conf, 
                          init_prob_matrix=bio_prob_matrix if has_bio_data else None)

    # save model path
    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)

    # wandb
    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) E-SNN-{conf.task}", group=conf.log_group, name=str(conf.seed), config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf))

    # run
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


def sweep(seed: int, conf_override: OmegaConf):
    def _objective(trial: optuna.Trial):
        conf = OmegaConf.merge(conf_override, {
            "seed": seed * 1000 + trial.number,

            "project_name": f"E-SNN-sweep",

            "es_conf": {
                "lr":           trial.suggest_categorical("lr",  [0.01, 0.05, 0.1, 0.15, 0.2]),
                "eps":          trial.suggest_categorical("eps", [1e-4, 1e-3, 0.01, 0.1, 0.2]),
            },
            "network_conf": {
                "num_neurons":  trial.suggest_categorical("num_neurons", [128, 256]),
            }
        })

        metrics = main(conf)
        return metrics["eval_fitness"]

    optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed)).optimize(_objective)


if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    if hasattr(_config, "sweep"):
        sweep(_config.sweep, _config)
    else:
        main(_config)