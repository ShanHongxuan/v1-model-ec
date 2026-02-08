import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
from tqdm import tqdm

# 导入你的模块
from networks.conn_snn import ConnSNN_Selected
from utils.mnist_loader import load_mnist_data

def load_trained_weights(file_path):
    """加载模型参数"""
    print(f">>> 正在从 {file_path} 加载参数...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['fixed_weights']

def main():
    # 1. 路径设置 (请确保文件名匹配)
    MODEL_PATH = 'trained_model.pkl'
    PHYSICS_PATH = 'neuron_physics.npz'
    NEURONS_CSV = '../dataset/mice_unnamed/neurons.csv.gz'

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    # 2. 加载 MNIST 测试集 (10,000 张图片)
    print(">>> 正在加载 MNIST 测试集...")
    test_images, test_labels = load_mnist_data('test') # 使用 'test' 分割
    num_test = test_images.shape[0]

    # 3. 加载生物物理参数 (Tau 等)
    print(">>> 加载物理参数...")
    phys = np.load(PHYSICS_PATH)
    bio_tau_Vm = tuple(phys['tau_Vm'].tolist())
    num_neurons = int(phys['num_neurons'])
    exc_ratio = float(phys['excitatory_ratio'])

    # 4. 初始化网络定义
    # 注意：这些参数必须与训练时完全一致！
    network = ConnSNN_Selected(
        out_dims=10,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        tau_Vm_vector=bio_tau_Vm,
        K_in=2.0,   # 请确保这里是你训练成功时的 K 值
        K_h=0.05, 
        K_out=200.0,
        dt=0.5,
        readout_indices=tuple(range(10)), # 假设你训练时用的是 get_l5_excitatory_indices 的结果
        readout_start_step=300,
        readout_end_step=400
    )

    # 5. 加载并转换进化出的参数
    raw_params, fixed_weights = load_trained_weights(MODEL_PATH)
    # [关键步骤] 将连接概率转换为确定的布尔连接 (Inference Mode)
    inference_params = jax.tree_util.tree_map(lambda p: p > 0.5, raw_params)

    # 6. 定义 JIT 加速的批次推理函数
    @jax.jit
    def batch_inference(images):
        """
        images shape: (Batch, Features)
        返回: (Batch, Out_Dims) 的 Logits
        """
        batch_size = images.shape[0]
        # 生成时序泊松脉冲 [Batch, Time=400, Features=196]
        # 保持与 MnistEnv 逻辑一致
        probs = jnp.expand_dims(images * (200.0 * 0.5 / 1000.0), 1)
        probs = jnp.repeat(probs, 400, axis=1)
        
        # 使用固定的 Key 保证推理的可复现性
        spikes = jax.random.bernoulli(jax.random.PRNGKey(0), probs).astype(jnp.float32)
        
        # 初始化 carry
        init_carry = network.initial_carry(jax.random.PRNGKey(0), batch_size)
        
        # 运行网络 (vmap 处理 batch)
        vmapped_apply = jax.vmap(network.apply, in_axes=({'params': None, 'fixed_weights': None}, 0, 0))
        _, logits = vmapped_apply({'params': inference_params, 'fixed_weights': fixed_weights}, init_carry, spikes)
        
        return logits

    # 7. 分批运行推理 (防止显存爆炸)
    BATCH_SIZE = 100
    all_preds = []
    
    print(f">>> 开始对 {num_test} 张测试图进行推理...")
    for i in tqdm(range(0, num_test, BATCH_SIZE)):
        batch_img = test_images[i : i + BATCH_SIZE]
        logits = batch_inference(batch_img)
        preds = jnp.argmax(logits, axis=-1)
        all_preds.append(preds)

    # 8. 计算准确率
    final_preds = jnp.concatenate(all_preds)
    accuracy = jnp.mean(final_preds == test_labels)

    print("\n" + "="*30)
    print(f"📊 全量测试集准确率: {accuracy*100:.2f}%")
    print("="*30)

    # 打印一些混淆情况
    for i in range(10):
        class_mask = (test_labels == i)
        class_acc = jnp.mean(final_preds[class_mask] == i)
        print(f"数字 {i} 的准确率: {class_acc*100:.1f}%")

if __name__ == "__main__":
    main()