# ==================== TF 预初始化 (与 ec.py 保持一致) ====================
try:
    from utils.mnist_loader import load_mnist_data
    print(">>> [Init] 正在预加载 MNIST 数据...")
    _ = load_mnist_data()
    print(">>> [Init] MNIST 数据预加载完成。")
except (ImportError, ModuleNotFoundError):
    pass
# =======================================================================

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from tqdm import tqdm
import os

# 导入核心模块
import ec
from networks import NETWORKS
from brax.envs import wrappers 
from envs.mnist_env import MnistEnv
from utils.mnist_loader import load_mnist_data

# ================= 辅助函数：探针 =================
def probe_logits(network, params, fixed_weights, input_obs):
    """
    使用当前进化的参数（取确定性均值）运行一次网络，获取 Logits。
    """
    # 1. 确定性采样：Prob > 0.5 视为连接存在
    # 这是 ec.py 中 eval_params 的逻辑
    binary_params = jax.tree_util.tree_map(lambda p: p > 0.5, params)
    
    # 2. 初始化状态
    key = jax.random.PRNGKey(0)
    # 获取单个 batch 的 carry (Batch=1)
    init_carry = network.initial_carry(key, 1)
    # 去除 batch 维 (vmap 内部逻辑)
    carry = jax.tree_util.tree_map(lambda x: x[0], init_carry)
    
    # 3. 运行前向传播
    variables = {'params': binary_params, 'fixed_weights': fixed_weights}
    _, logits = network.apply(variables, carry, input_obs)
    
    return logits

def analyze_logits(logits, target_label):
    """分析 Logits 并给出 K_out 建议"""
    logits_np = np.array(logits)
    probs = jax.nn.softmax(logits)
    probs_np = np.array(probs)
    
    target_score = logits_np[target_label]
    target_prob = probs_np[target_label]
    
    logit_range = np.max(logits_np) - np.min(logits_np)
    
    print(f"\n🔍 [Logits 探针]")
    print(f"    Raw Logits: {np.array2string(logits_np, precision=2, suppress_small=True)}")
    print(f"    Softmax:    {np.array2string(probs_np, precision=4, suppress_small=True)}")
    print(f"    Range: {logit_range:.2f} | Target Prob: {target_prob:.4f}")
    
    # K_out 建议逻辑
    if logit_range < 1.0:
        print("    ⚠️  Logits 差异太小！Softmax 无法区分。")
        print("    👉 建议: 增大 K_out (例如 x2 或 x5)")
    elif logit_range > 100.0:
        print("    ⚠️  Logits 差异极大！梯度可能饱和，但通常可以接受。")
        print("    👉 建议: 保持或稍微减小 K_out")
    else:
        print("    ✅ Logits 范围健康 (理想范围 2.0 ~ 50.0)。")

# ================= 主程序 =================
def main():
    print("\n=== EC 框架级一致性测试 (带 Logits 探针) ===")
    
    # 1. 配置 (模拟 train_mnist.sh)
    conf = OmegaConf.create({
        "seed": 42,
        "pop_size": 128,  # Debug 用小种群
        "eval_size": 32,
        "total_generations": 200,
        
        "lr": 0.1,
        "eps": 0.001,
        "weight_decay": 0.0,
        
        # [调参重点区域]
        "network_conf": {
            "K_in": 0.03,   # 输入增益
            "K_h": 0.18,     # 递归增益
            "K_out": 0.01, # 输出增益
            "dt": 0.5,
        },
        
        "episode_conf": {
            "max_episode_length": 1, 
            "action_repeat": 1
        },
        
        "use_bio_probability": True,
        "bio_prob_mix_factor": 0.5,
        "network_type": "ConnSNN",
        
        "es_conf": {},
        "warmup_steps": 0
    })

    # 2. 数据准备 (单图过拟合)
    all_images, all_labels = load_mnist_data('train')
    target_idx = 7 # 选择数字 3 作为目标
    single_image = all_images[target_idx:target_idx+1]
    single_label = all_labels[target_idx:target_idx+1]
    
    # 用于探针的单个观测输入 (去除 Batch 维)
    # MnistEnv 现在的 reset 生成 (Batch, Time, Feat)，这里我们手动模拟
    # 我们直接取 image，让探针函数里的 network 把它当做概率图处理 (如果是 Time-Tensorization 模式)
    # 或者是生成好脉冲。为了准确，我们复用 Env 的逻辑生成一次脉冲。
    dummy_env = MnistEnv(single_image, single_label, presentation_steps=200, dt_ms=0.5)
    probe_state = dummy_env.reset(jax.random.PRNGKey(0))
    probe_obs = probe_state.obs # (Time, 196)

    print(f">>> 目标标签: {single_label[0]}")

    # 3. 环境初始化
    snn_steps = 200
    base_env = MnistEnv(
        images=single_image, 
        labels=single_label, 
        presentation_steps=snn_steps,
        dt_ms=conf.network_conf.dt
    )
    env = wrappers.VmapWrapper(base_env)

    # 4. 生物参数
    bio_prob_matrix = None
    bio_tau_Vm = None
    if os.path.exists('neuron_physics.npz'):
        physics_data = np.load('neuron_physics.npz')
        conf.network_conf.num_neurons = int(physics_data['num_neurons'])
        conf.network_conf.excitatory_ratio = float(physics_data['excitatory_ratio'])
        bio_tau_Vm = tuple(physics_data['tau_Vm'].tolist())
        
        if conf.use_bio_probability and os.path.exists('init_probability.npy'):
            raw_prob = np.load('init_probability.npy')
            mix = conf.bio_prob_mix_factor
            bio_prob_matrix = mix * raw_prob + (1.0 - mix) * 0.5
            print(f">>> 生物数据已加载 (Mix={mix})")

    # 5. 网络初始化
    network_dtype = jnp.float32 
    network_cls = NETWORKS[conf.network_type]
    network_kwargs = {
        "out_dims": env.action_size,
        "neuron_dtype": network_dtype,
        **conf.network_conf
    }
    if bio_tau_Vm:
        network_kwargs["tau_Vm_vector"] = bio_tau_Vm
    
    network = network_cls(**network_kwargs)

    # 6. ESConfig (关键配置一致性)
    p_dtype = jnp.float32
    action_dtype = jnp.float32
    
    optim = optax.chain(
        optax.scale_by_adam(mu_dtype=p_dtype),
        optax.scale(-conf.lr)
    )
    
    es_conf = ec.ESConfig(
        network_cls=network,
        optim_cls=optim,
        env_cls=env,
        pop_size=conf.pop_size,
        lr=conf.lr,
        eps=conf.eps,
        eval_size=conf.eval_size,
        weight_decay=conf.weight_decay,
        warmup_steps=conf.warmup_steps,
        action_dtype=action_dtype,
        p_dtype=p_dtype,
        network_dtype=network_dtype,
        # [关键] 必须显式关闭 Brax 的默认处理，才能让 K_out 生效
        clip_action=False,
        normalize_obs=False
    )

    # 7. 运行器初始化
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runner = ec._runner_init(key_run, key_network_init, es_conf, init_prob_matrix=bio_prob_matrix)

    # 8. 训练循环
    print(">>> 开始训练...")
    pbar = tqdm(range(1, conf.total_generations + 1))
    
    for step in pbar:
        runner, metrics = ec._runnner_run(runner, es_conf)
        
        metrics_cpu = jax.device_get(metrics)
        fit = metrics_cpu['fitness']
        eval_fit = metrics_cpu['eval_fitness']
        
        pbar.set_description(f"Eval Acc: {eval_fit:.4f} | Train Fit: {fit:.4f}")
        
        # 每 10 代或者是刚开始，进行一次探针检查
        if step == 1 or step % 20 == 0:
            # 这里的 params 是 runner 中的均值参数
            logits = probe_logits(network, runner.params, runner.fixed_weights, probe_obs)
            analyze_logits(logits, single_label[0])
        
        if eval_fit > 0.99:
            print(f"\n🚀 [Success] 在第 {step} 代过拟合成功！")
            # 最后再看一眼 Logits
            logits = probe_logits(network, runner.params, runner.fixed_weights, probe_obs)
            analyze_logits(logits, single_label[0])
            return

    print("\n❌ 训练结束，未能完全收敛。")

if __name__ == "__main__":
    main()