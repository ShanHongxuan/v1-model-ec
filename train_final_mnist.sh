#!/bin/bash

echo "=== MNIST 最终训练: 验证生物先验的优势 ==="

# 经过 debug 验证的最佳物理参数
K_IN="12.0"    # 给 10.0 加一点点余量，确保稳健发放
K_H="1.0"      # 递归增益 1.0 表现良好
K_OUT="100.0"  # 输出增益 100.0 确保了 30+ 的 Logits Range

# 训练配置
TASK="mnist"
GENS=1000
POPSIZE=2048 # 显存允许的话，越大越好
LR=0.1

PROJECT="E-SNN-Final-Run"
GROUP="Bio_vs_Random_$(date +%Y%m%d_%H%M)"

# --- 1. 随机初始化 (Baseline) ---
echo ">>> 启动实验: Random Initialization"
python ec.py \
  task="$TASK" \
  project_name="$PROJECT" log_group="$GROUP" \
  run_name="Random_Kin=${K_IN}" \
  total_generations="$GENS" \
  es_conf.pop_size="$POPSIZE" \
  es_conf.lr="$LR" \
  network_conf.K_in="$K_IN" \
  network_conf.K_h="$K_H" \
  network_conf.K_out="$K_OUT" \
  network_conf.dt=0.5 \
  episode_conf.max_episode_length=1 \
  use_bio_probability="false"

# --- 2. 生物初始化 (Bio 50%) ---
echo ">>> 启动实验: Bio Initialization (Mix 0.5)"
python ec.py \
  task="$TASK" \
  project_name="$PROJECT" log_group="$GROUP" \
  run_name="Bio_Mix0.5_Kin=${K_IN}" \
  total_generations="$GENS" \
  es_conf.pop_size="$POPSIZE" \
  es_conf.lr="$LR" \
  network_conf.K_in="$K_IN" \
  network_conf.K_h="$K_H" \
  network_conf.K_out="$K_OUT" \
  network_conf.dt=0.5 \
  episode_conf.max_episode_length=1 \
  use_bio_probability="true" \
  bio_prob_mix_factor="0.5"

# --- 3. 强生物初始化 (Bio 100%) ---
# 可选：看看完全保留生物结构是否也能工作
echo ">>> 启动实验: Bio Initialization (Mix 1.0)"
python ec.py \
  task="$TASK" \
  project_name="$PROJECT" log_group="$GROUP" \
  run_name="Bio_Mix1.0_Kin=${K_IN}" \
  total_generations="$GENS" \
  es_conf.pop_size="$POPSIZE" \
  es_conf.lr="$LR" \
  network_conf.K_in="$K_IN" \
  network_conf.K_h="$K_H" \
  network_conf.K_out="$K_OUT" \
  network_conf.dt=0.5 \
  episode_conf.max_episode_length=1 \
  use_bio_probability="true" \
  bio_prob_mix_factor="1.0"