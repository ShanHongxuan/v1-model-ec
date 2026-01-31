#!/bin/bash

# =================================================================
# MNIST 任务 - 最佳 K 值邻域扫描 & 强/无生物先验对比
#
# 实验目标:
# 1. 在已知的最佳 K 值附近进行高密度采样，找到最优动力学参数。
# 2. 直接对比纯随机初始化 (mix=0) 与强生物先验 (mix=1) 的性能。
# =================================================================

echo "开始 MNIST 最佳 K 值邻域扫描..."

# WandB 项目配置
PROJECT_NAME="E-SNN-mnist"
GROUP_NAME="Final_K_Neighborhood_Sweep_$(date +%Y%m%d_%H%M)"

# 基础任务配置
TASK="mnist"
TOTAL_GENS=500
USE_BIO_PROB=true
LEARNING_RATE="0.1"

# 定义要对比的初始化策略
# mix=1.0: 强生物先验 (使用平滑后的生物概率)
# mix=0.0: 纯随机基线 (等同于 use_bio_probability=false)
MIX_FACTORS=(1.0) 

# 定义要在最佳 K 值附近扫描的范围
K_H_VALUES=(0.16 0.18)
K_IN_VALUES=(0.03 0.035 0.04)

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
        network_conf.K_out=2000.0 \
        network_conf.dt=0.5 \
        network_conf.sim_time=0.5 \
        episode_conf.max_episode_length=1 \
        env_conf.max_rate_hz="$MAX_RATE" \
        project_name="$PROJECT_NAME" \
        log_group="$GROUP_NAME" \
        run_name="$RUN_NAME"
        
    done
  done
done

echo "所有实验已完成！"