#!/bin/bash

# =================================================================
# MNIST 自动循环训练脚本 (Time-Tensorization Mode)
#
# 功能:
# 1. 自动遍历 Bio_Mix, K_in, K_h 的组合
# 2. 自动区分 "纯随机基线" 和 "生物混合实验"
# 3. 核心架构: max_episode_length=1 (GPU内循环)
# =================================================================

echo "=== 开始 MNIST 参数循环扫描 ==="

# --- 1. 固定配置 (Fixed Configs) ---
TASK="mnist"
GENS=500           # 训练代数
POPSIZE=1024        # 种群大小
LR="0.1"            # 学习率 (0.1 表现稳定)
K_OUT="100.0"       # 输出增益 (之前验证过必须够大，Softmax才有梯度)
DT="0.5"            # 物理步长

# WandB 设置
PROJECT="E-SNN-Final-Loop"
GROUP="Sweep_$(date +%Y%m%d_%H%M)"

# --- 2. 扫描参数范围 (Sweep Ranges) ---

# A. 生物混合因子 (0 = 纯随机基线, 0.2 = 弱生物先验, 0.5 = 强生物先验)
# 建议: 对比 0 和 0.5 是最直观的
MIX_FACTORS=(1)

# B. 输入增益 (K_in)
# 之前 debug 测出 12.0~15.0 能激活网络。
# 我们在周围扫一下：10.0 (保守), 15.0 (推荐), 20.0 (激进)
K_IN_VALUES=(15)

# C. 递归增益 (K_h)
# 之前 1.0 是工作的。我们测试一下稍微抑制一点(0.5)和标准(1.0)
K_H_VALUES=(1)

# --- 3. 主循环 ---

for MIX in "${MIX_FACTORS[@]}"; do
  for K_IN in "${K_IN_VALUES[@]}"; do
    for K_H in "${K_H_VALUES[@]}"; do

      # --- 逻辑判断: 是否使用生物概率 ---
      # 如果混合因子是 0，我们直接关闭 use_bio_probability，作为纯随机对照组
      if [ "$MIX" == "0" ]; then
        USE_BIO="false"
        RUN_PREFIX="Random"
      else
        USE_BIO="true"
        RUN_PREFIX="Bio_Mix${MIX}"
      fi

      # 构造 WandB 的 Run Name
      RUN_NAME="${RUN_PREFIX}_Kin=${K_IN}_Kh=${K_H}"

      echo "------------------------------------------------------"
      echo ">>> 正在启动: ${RUN_NAME}"
      echo "    Mix: $MIX | K_in: $K_IN | K_h: $K_H"
      echo "------------------------------------------------------"

      python ec.py \
        task="$TASK" \
        project_name="$PROJECT" \
        log_group="$GROUP" \
        run_name="$RUN_NAME" \
        total_generations="$GENS" \
        es_conf.pop_size="$POPSIZE" \
        es_conf.lr="$LR" \
        network_conf.K_in="$K_IN" \
        network_conf.K_h="$K_H" \
        network_conf.K_out="$K_OUT" \
        network_conf.dt="$DT" \
        episode_conf.max_episode_length=1 \
        use_bio_probability="$USE_BIO" \
        bio_prob_mix_factor="$MIX"

    done
  done
done

echo "=== 所有扫描任务已完成 ==="