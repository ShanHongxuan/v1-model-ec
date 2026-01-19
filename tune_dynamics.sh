#!/bin/bash

# =================================================================
# SNN 动力学调优脚本 (针对 MNIST 任务) - [修正版]
#
# 修正了 OmegaConf 的参数传递格式，从 --key value 改为 key=value
# =================================================================

echo "开始 SNN 动力学参数扫描 (K_in, K_h)..."

# --- WandB 配置 ---
PROJECT_NAME="E-SNN-mnist-dynamics-tuning"
GROUP_NAME="Tune_K_in_K_h_$(date +%Y%m%d_%H%M)"

# --- 固定的超参数 ---
TASK="mnist"
TOTAL_GENS=300
LR=0.1
USE_BIO_PROB=true
MIX_FACTOR=0.2

# --- 需要扫描的参数 ---
K_IN_VALUES=(0.2 0.1 0.05 0.02)
K_H_VALUES=(0.5 0.2 0.1 0.05)

# --- 开始扫描 ---
for K_IN in "${K_IN_VALUES[@]}"; do
  for K_H in "${K_H_VALUES[@]}"; do
    
    RUN_NAME="kin=${K_IN}_kh=${K_H}"
    
    echo "------------------------------------------------------"
    echo "==> 🚀 正在运行: ${RUN_NAME}"
    echo "------------------------------------------------------"
    
    # [重要修改] 所有参数都使用 key=value 格式
    python ec.py \
      task="$TASK" \
      total_generations="$TOTAL_GENS" \
      use_bio_probability="$USE_BIO_PROB" \
      bio_prob_mix_factor="$MIX_FACTOR" \
      es_conf.lr="$LR" \
      network_conf.K_in="$K_IN" \
      network_conf.K_h="$K_H" \
      project_name="$PROJECT_NAME" \
      log_group="$GROUP_NAME" \
      run_name="$RUN_NAME"
      
  done
done

echo "✅ 动力学扫描完成!"
echo "请前往 WandB 查看 '${PROJECT_NAME}' 项目中的结果。"