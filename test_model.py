import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm

from networks.conn_snn import ConnSNN_Selected
from utils.mnist_loader import load_mnist_data

# ==================== 必须与 ec.py 逻辑完全一致的辅助函数 ====================
def get_l5_excitatory_indices(csv_path, total_neurons, n_out=10):
    print(f">>> 正在同步 L5 读出神经元索引 (From {csv_path})...")
    df = pd.read_csv(csv_path)
    df['EI_rank'] = df['EI'].map({'E': 0, 'I': 1})
    df_sorted = df.sort_values(['EI_rank', 'simple_id']).reset_index(drop=True)
    l5e_mask = (df_sorted['layer'] == 'L5') & (df_sorted['EI'] == 'E')
    l5e_indices = df_sorted[l5e_mask].index.to_numpy()
    selected = np.linspace(0, len(l5e_indices)-1, n_out, dtype=int)
    final_indices = l5e_indices[selected]
    print(f"✅ 同步完成，读出索引为: {final_indices}")
    return tuple(final_indices.tolist())

def load_trained_weights(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['fixed_weights']

def main():
    # --- 1. 配置 (请根据你 WandB 成功时的参数填写) ---
    MODEL_PATH = 'test4.pkl'
    PHYSICS_PATH = 'neuron_physics.npz'
    NEURONS_CSV = '../dataset/mice_unnamed/neurons.csv.gz'
    
    # [核心参数] 请务必确认与你训练成功时的数值一模一样！
    K_IN = 2.0    
    K_H = 0.05    
    K_OUT = 200.0 
    INPUT_HZ = 200.0

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return

    # --- 2. 加载测试集 ---
    test_images, test_labels = load_mnist_data('test')
    num_test = test_images.shape[0]

    # --- 3. 加载物理参数与同步读出索引 ---
    phys = np.load(PHYSICS_PATH)
    num_neurons = int(phys['num_neurons'])
    bio_tau_Vm = tuple(phys['tau_Vm'].tolist())
    exc_ratio = float(phys['excitatory_ratio'])
    
    # 获取与训练时完全一致的读出索引
    l5e_indices = get_l5_excitatory_indices(NEURONS_CSV, num_neurons, 10)

    # --- 4. 初始化网络 ---
    network = ConnSNN_Selected(
        out_dims=10,
        num_neurons=num_neurons,
        excitatory_ratio=exc_ratio,
        tau_Vm_vector=bio_tau_Vm,
        K_in=K_IN, 
        K_h=K_H, 
        K_out=K_OUT,
        dt=0.5,
        readout_indices=l5e_indices, # 使用同步后的索引
        readout_start_step=300,
        readout_end_step=400
    )

    # --- 5. 加载权重 ---
    raw_params, fixed_weights = load_trained_weights(MODEL_PATH)
    inference_params = jax.tree_util.tree_map(lambda p: p > 0.5, raw_params)

    # --- 6. 推理函数 ---
    @jax.jit
    def batch_inference(images):
        batch_size = images.shape[0]
        # 时间窗 200ms (400步)
        probs = jnp.expand_dims(images * (INPUT_HZ * 0.5 / 1000.0), 1)
        probs = jnp.repeat(probs, 400, axis=1)
        spikes = jax.random.bernoulli(jax.random.PRNGKey(0), probs).astype(jnp.float32)
        
        init_carry = network.initial_carry(jax.random.PRNGKey(0), batch_size)
        vmapped_apply = jax.vmap(network.apply, in_axes=({'params': None, 'fixed_weights': None}, 0, 0))
        _, logits = vmapped_apply({'params': inference_params, 'fixed_weights': fixed_weights}, init_carry, spikes)
        return logits

    # --- 7. 运行测试 ---
    BATCH_SIZE = 100
    all_preds = []
    for i in tqdm(range(0, num_test, BATCH_SIZE)):
        batch_img = test_images[i : i + BATCH_SIZE]
        logits = batch_inference(batch_img)
        all_preds.append(jnp.argmax(logits, axis=-1))

    final_preds = jnp.concatenate(all_preds)
    accuracy = jnp.mean(final_preds == test_labels)

    print("\n" + "="*40)
    print(f"📊 最终测试集准确率: {accuracy*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()