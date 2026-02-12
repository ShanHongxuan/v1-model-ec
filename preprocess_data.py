import pandas as pd
import numpy as np
import os
import json
import glob
from tqdm import tqdm

# ================= 配置路径 =================
# 输入数据路径 (由 main.py 生成)
NEURONS_PATH = '../dataset/mice_unnamed/neurons.csv.gz'
PROBS_PATH = '../dataset/mice_unnamed/all_probabilities.csv.gz'

# GLIF 模型文件夹路径
# 请修改为您实际的 GLIF 文件夹路径
GLIF_DIR = "/home/mzsk/code/mice_unnamed_new/glif_models_VISp"

# 输出文件路径
OUTPUT_PROB_MAT = 'init_probability.npy'
OUTPUT_PHYSICS = 'neuron_physics.npz'

# 默认参数 (当找不到 GLIF 文件时使用)
DEFAULT_PARAMS = {
    'tau': 10.0,   # ms
    'c_m': 100.0,  # pF
    'v_th': 1.0,   # relative
}
# ===========================================

def convert_allen_glif_json_to_params(raw: dict):
    """
    提取所需的物理参数。
    这里简化了 sim_poisson.py 中的逻辑，只提取 ConnSNN 需要的核心参数。
    """
    coeffs = raw.get('coeffs', {})
    
    # 1. 膜电容 C_m (pF)
    C_F = float(raw.get('C', 0.0))
    coeff_C = coeffs.get('C', 1.0)
    c_m_pF = C_F * float(coeff_C) * 1e12
    
    # 2. 膜时间常数 tau_m (ms)
    # tau = R * C
    R_input = float(raw.get('R_input', 0.0))
    tau_ms = R_input * C_F * 1000.0
    
    return {
        'tau': tau_ms,
        'c_m': c_m_pF
    }

def main():
    print("=== 1. 加载神经元数据 ===")
    if not os.path.exists(NEURONS_PATH):
        raise FileNotFoundError(f"未找到神经元文件: {NEURONS_PATH}")
    
    neurons_df = pd.read_csv(NEURONS_PATH)
    print(f"原始神经元数量: {len(neurons_df)}")
    
    # 检查必要列
    required_cols = ['simple_id', 'EI', 'cell_class']
    for col in required_cols:
        if col not in neurons_df.columns:
            raise ValueError(f"neurons.csv 缺少列: {col}")

    print("=== 2. 重排神经元 (Excitatory在前, Inhibitory在后) ===")
    # 映射 EI 为排序权重: E->0, I->1
    neurons_df['EI_rank'] = neurons_df['EI'].map({'E': 0, 'I': 1})
    
    # 排序：先按 EI 类型排，同类型内按原 simple_id 排
    neurons_sorted = neurons_df.sort_values(['EI_rank', 'simple_id']).reset_index(drop=True)
    
    # 获取统计信息
    n_total = len(neurons_sorted)
    n_exc = len(neurons_sorted[neurons_sorted['EI'] == 'E'])
    n_inh = len(neurons_sorted[neurons_sorted['EI'] == 'I'])
    excitatory_ratio = n_exc / n_total
    
    print(f"排序后: Total={n_total}, Exc={n_exc}, Inh={n_inh}")
    print(f"Exc Ratio: {excitatory_ratio:.4f}")
    
    # 创建 ID 映射: old_simple_id -> new_matrix_index
    id_map = dict(zip(neurons_sorted['simple_id'], neurons_sorted.index))
    
    print("=== 3. 构建连接概率矩阵 ===")
    prob_matrix = np.zeros((n_total, n_total), dtype=np.float32)
    
    if os.path.exists(PROBS_PATH):
        print(f"正在读取概率文件 (可能较慢)... {PROBS_PATH}")
        # chunksize 读取以防文件过大
        chunksize = 100000
        probs_chunks = pd.read_csv(PROBS_PATH, chunksize=chunksize)
        
        count = 0
        for chunk in tqdm(probs_chunks, desc="填充矩阵"):
            # 过滤掉不在当前神经元列表中的连接 (如果 main.py 导出时范围不一致)
            chunk = chunk[chunk['pre_simple_id'].isin(id_map) & chunk['post_simple_id'].isin(id_map)]
            
            # 映射索引
            pre_idx = chunk['pre_simple_id'].map(id_map).values
            post_idx = chunk['post_simple_id'].map(id_map).values
            probs = chunk['probability'].values
            
            # 填充矩阵
            prob_matrix[pre_idx, post_idx] = probs
            count += len(chunk)
            
        print(f"矩阵构建完成，处理了 {count} 条概率记录")
    else:
        print("警告: 未找到概率文件，将生成全 0.5 的矩阵作为占位符 (不推荐)")
        prob_matrix[:] = 0.5

    # 保存矩阵
    np.save(OUTPUT_PROB_MAT, prob_matrix)
    print(f"概率矩阵已保存至: {OUTPUT_PROB_MAT}")

    print("=== 4. 提取 GLIF 物理参数 ===")
    tau_arr = np.zeros(n_total, dtype=np.float32)
    c_m_arr = np.zeros(n_total, dtype=np.float32)
    
    # 预加载 GLIF 文件路径
    unique_classes = neurons_sorted['cell_class'].unique()
    class_to_files = {}
    
    print("正在索引 GLIF JSON 文件...")
    for cls in unique_classes:
        if pd.isna(cls): continue
        cls_path = os.path.join(GLIF_DIR, str(cls))
        files = glob.glob(os.path.join(cls_path, "*.json"))
        class_to_files[cls] = files
    
    rng = np.random.default_rng(42)
    
    print("正在为每个神经元分配参数...")
    for idx, row in tqdm(neurons_sorted.iterrows(), total=n_total):
        cls = row['cell_class']
        files = class_to_files.get(cls, [])
        
        params = DEFAULT_PARAMS
        
        if files:
            # 随机选择一个文件以引入同类内的多样性
            chosen_file = rng.choice(files)
            try:
                with open(chosen_file, 'r') as f:
                    raw_data = json.load(f)
                    params = convert_allen_glif_json_to_params(raw_data)
            except Exception as e:
                print(f"读取 {chosen_file} 失败: {e}")
        
        tau_arr[idx] = params['tau']
        c_m_arr[idx] = params['c_m']

    # 统计参数分布
    print(f"Tau (ms): Mean={tau_arr.mean():.2f}, Min={tau_arr.min():.2f}, Max={tau_arr.max():.2f}")
    print(f"C_m (pF): Mean={c_m_arr.mean():.2f}, Min={c_m_arr.min():.2f}, Max={c_m_arr.max():.2f}")

    # 保存参数
    np.savez(OUTPUT_PHYSICS, 
             tau_Vm=tau_arr, 
             c_m=c_m_arr, 
             num_neurons=n_total,
             num_excitatory=n_exc,
             excitatory_ratio=excitatory_ratio)
    
    print(f"物理参数已保存至: {OUTPUT_PHYSICS}")
    print("=== 预处理完成 ===")

if __name__ == "__main__":
    main()