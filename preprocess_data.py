import pandas as pd
import numpy as np
import os

# === 配置路径 ===
NEURONS_PATH = '../dataset/mice_unnamed/neurons.csv.gz'
CONNS_PATH = '../dataset/mice_unnamed/all_probabilities.csv.gz'
OUTPUT_PATH = '../dataset/mice_unnamed/init_params.npz'

def preprocess_bio_data():
    print("1. Loading data...")
    neurons_df = pd.read_csv(NEURONS_PATH)
    probs_df = pd.read_csv(CONNS_PATH)
    
    total_neurons = len(neurons_df)
    print(f"   Total neurons: {total_neurons}")

    # === 关键步骤：按照 EI 类型排序 ===
    # 我们定义排序规则：E 在前，I 在后
    # 假设 'EI' 列的值为 'E' 或 'I'
    print("2. Sorting neurons (Excitatory first, then Inhibitory)...")
    
    # 创建辅助列用于排序 (E=0, I=1)
    neurons_df['sort_key'] = neurons_df['EI'].map({'E': 0, 'I': 1})
    
    # 执行排序
    sorted_neurons = neurons_df.sort_values(by=['sort_key', 'simple_id']).reset_index(drop=True)
    
    # 计算 E 的数量和比例
    num_exc = len(sorted_neurons[sorted_neurons['EI'] == 'E'])
    exc_ratio = num_exc / total_neurons
    
    print(f"   Excitatory count: {num_exc}")
    print(f"   Inhibitory count: {total_neurons - num_exc}")
    print(f"   Calculated excitatory_ratio: {exc_ratio:.6f}")

    # === 建立映射表 ===
    # 旧的 simple_id -> 新的矩阵索引 (0 ~ N-1)
    # sorted_neurons 的 index 就是新的矩阵索引
    old_id_to_new_idx = dict(zip(sorted_neurons['simple_id'], sorted_neurons.index))

    # === 构建概率矩阵 ===
    print("3. Constructing probability matrix...")
    probability_matrix = np.zeros((total_neurons, total_neurons), dtype=np.float32)

    # 填充矩阵
    # 这是一个向量化的快速填充方法，或者可以用循环
    # 为了清晰起见，我们迭代处理（如果数据量极大，建议用 map）
    
    # 映射 simple_id 到 新索引
    # 注意：probs_df 里的 id 是旧的 simple_id
    
    # 过滤掉不在神经元列表里的边（以防万一）
    valid_mask = (probs_df['pre_simple_id'].isin(old_id_to_new_idx)) & \
                 (probs_df['post_simple_id'].isin(old_id_to_new_idx))
    clean_probs = probs_df[valid_mask].copy()
    
    # 转换 ID
    pre_indices = clean_probs['pre_simple_id'].map(old_id_to_new_idx).values
    post_indices = clean_probs['post_simple_id'].map(old_id_to_new_idx).values
    probs_values = clean_probs['probability'].values
    
    # 赋值
    # matrix[post, pre] 还是 [pre, post]? 
    # ec.py 的 conn_dense(kernel, x) 实现是 dot(x, kernel)
    # 这意味着 x (1, N) @ kernel (N, N) -> (1, N)
    # 所以 kernel 的行应该是 pre，列应该是 post。即 kernel[i, j] 表示 i -> j
    probability_matrix[pre_indices, post_indices] = probs_values

    print(f"   Matrix shape: {probability_matrix.shape}")
    print(f"   Non-zero connections: {np.count_nonzero(probability_matrix)}")

    # === 保存 ===
    print("4. Saving to .npz...")
    np.savez(
        OUTPUT_PATH, 
        kernel_h=probability_matrix,  # 递归连接概率
        num_neurons=total_neurons,
        excitatory_ratio=exc_ratio
    )
    print(f"Done! Saved to {OUTPUT_PATH}")
    print("You can now load this file in ec.py")

if __name__ == "__main__":
    preprocess_bio_data()