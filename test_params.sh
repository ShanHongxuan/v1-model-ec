#!/bin/bash

echo "开始扫描 K_in 参数 (泊松输入模式)..."

# [修改] 固定 K_h，扩大 K_in 的扫描范围
FIXED_K_H="0.014" # 使用之前找到的最佳值
K_IN_VALUES=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

for K_IN in "${K_IN_VALUES[@]}"; do
  python debug_firing_rate.py --K_in "$K_IN" --K_h "$FIXED_K_H"
done
