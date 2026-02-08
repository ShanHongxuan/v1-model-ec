#!/bin/bash

# =================================================================
# MNIST 参数敏感度扫描脚本 (基于最新 ec.py)
# 
# 基准值: K_in=2.0, K_h=0.05
# 目标: 探索更优的信号放大与内部反馈平衡
# =================================================================

echo "开始 MNIST K_in 与 K_h 参数扫描..."

# --- 1. 固定配置 ---
TASK="mnist"
GENS=200           # 扫描时可以用 1000 代快速看趋势
POPSIZE=1024
LR="0.1"           # 使用之前建议的更平稳的学习率
K_OUT="20.0"       # 保持较强的输出增益
MIX="1.0"           # 保持生物混合比例

# WandB 项目设置
PROJECT="E-SNN-Mnist-Sweep"
GROUP="K_Sweep_$(date +%Y%m%d_%H%M)"

# --- 2. 探索范围 ---

# K_in (输入增益): 成功基准是 2.0
# 我们测试从轻微驱动到强力驱动
K_IN_VALUES=(2.0)

# K_h (递归增益): 成功基准是 0.05
# 我们测试极弱递归到中等递归
K_H_VALUES=(0.05)

# --- 3. 循环测试 ---

for kin in "${K_IN_VALUES[@]}"; do
  for kh in "${K_H_VALUES[@]}"; do
    
    # 构造易读的 Run Name
    RUN_NAME="kin${kin}_kh${kh}"
    
    echo "------------------------------------------------------"
    echo ">>> 正在启动: ${RUN_NAME}"
    echo "    K_in: $kin | K_h: $kh"
    echo "------------------------------------------------------"

    # 执行 ec.py
    # 注意：参数名必须与 ec.py 中的 network_conf 对应
    python ec.py \
      task="$TASK" \
      project_name="$PROJECT" \
      log_group="$GROUP" \
      run_name="$RUN_NAME" \
      total_generations="$GENS" \
      es_conf.pop_size="$POPSIZE" \
      es_conf.lr="$LR" \
      network_conf.K_in="$kin" \
      network_conf.K_h="$kh" \
      network_conf.K_out="$K_OUT" \
      use_bio_probability=true \
      bio_prob_mix_factor="$MIX"

  done
done

echo "=== 参数扫描任务全部完成 ==="