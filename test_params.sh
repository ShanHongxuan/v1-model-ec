#!/bin/bash

echo "开始扫描 K_in 和 K_h 参数 (精细扫描)..."

# [修改] 使用更小、更精细的 K 值范围
K_IN_VALUES=(0.1 0.2 0.3 0.5)
K_H_VALUES=(0.01 0.02 0.05 0.08 0.1)

# 遍历所有参数组合
for K_IN in "${K_IN_VALUES[@]}"; do
  for K_H in "${K_H_VALUES[@]}"; do
    python debug_firing_rate.py --K_in "$K_IN" --K_h "$K_H"
  done
done

echo "扫描完成。"