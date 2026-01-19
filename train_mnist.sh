#!/bin/bash

# =================================================================
# MNIST 任务 - 混合因子与K值最终扫描脚本
#
# 实验目标:
# 1. 对比不同强度的生物先验 (mix_factor=0, 0.5, 1) 的影响。
# 2. 在每个先验强度下，找到最优的 (K_in, K_h) 动力学参数。
# =================================================================

echo "开始 MNIST 混合因子与K值最终扫描..."

# WandB 项目配置
PROJECT_NAME="E-SNN-mnist"
GROUP_NAME="BioMix_vs_K_Sweep_$(date +%Y%m%d_%H%M)"

# 基础任务配置
TASK="mnist"
TOTAL_GENS=1000
USE_BIO_PROB=true # 保持为 true，通过 mix_factor=0 来实现随机基线
LEARNING_RATE=0.1

# 定义要扫描的超参数范围
# [新增] 混合因子
MIX_FACTORS=(0 1.0) 

# K_h: 在已知的健康范围内进行精细搜索
K_H_VALUES=(0.012 0.014 0.016) 
# K_in: 在已知的健康范围内进行精细搜索
K_IN_VALUES=(0.04 0.06 0.08)

# 遍历所有参数组合
for MIX in "${MIX_FACTORS[@]}"; do
  for K_H in "${K_H_VALUES[@]}"; do
    for K_IN in "${K_IN_VALUES[@]}"; do
      
      # 为 WandB 运行命名，包含所有变量
      RUN_NAME="mix=${MIX}_kh=${K_H}_kin=${K_IN}"
      
      echo "------------------------------------------------------"
      echo "正在运行: ${RUN_NAME}"
      echo "------------------------------------------------------"
      
      python ec.py \
        task="$TASK" \
        total_generations="$TOTAL_GENS" \
        use_bio_probability="$USE_BIO_PROB" \
        bio_prob_mix_factor="$MIX" \
        es_conf.lr="$LEARNING_RATE" \
        network_conf.K_in="$K_IN" \
        network_conf.K_h="$K_H" \
        project_name="$PROJECT_NAME" \
        log_group="$GROUP_NAME" \
        run_name="$RUN_NAME"
        
    done
  done
done

echo "所有实验已完成！"