import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional

@struct.dataclass
class EnvState:
    # --- 所有没有默认值的字段放前面 ---
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    current_image: jnp.ndarray
    current_label: jnp.ndarray
    time_step: jnp.int32
    cumulative_output: jnp.ndarray
    # --- 所有有默认值的字段放后面 ---
    metrics: dict = struct.field(default_factory=dict)

class MnistEnv:
    def __init__(self, images, labels, presentation_steps=50):
        """
        presentation_steps: 每张图片展示多少个仿真步 (例如 50步)
        """
        self.presentation_steps = presentation_steps
        
        self._images = images
        self._labels = labels
        self._num_data = self._images.shape[0]
        
        self.observation_size = 784
        self.action_size = 10  # 10个数字分类

    def reset(self, rng: jnp.ndarray) -> EnvState:
        # 随机选择一张图片
        idx = jax.random.randint(rng, (), 0, self._num_data)
        image = self._images[idx]
        label = self._labels[idx]
        
        return EnvState(
            obs=image,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            current_image=image,
            current_label=label,
            time_step=jnp.int32(0),
            cumulative_output=jnp.zeros(self.action_size, dtype=jnp.float32)
        )

    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """
        action: 网络的输出 (10维向量, 通常是发放率或电压)
        """
        new_cumulative = state.cumulative_output + action
        new_time = state.time_step + 1
        done = new_time >= self.presentation_steps
        
        # ==================== [核心修改] ====================
        # 使用 Softmax 概率作为奖励，而不是 0/1 硬奖励
        
        # 1. 将累积的输出（logits）通过 Softmax 转换为概率分布
        #    为了数值稳定性，减去最大值
        logits = new_cumulative - jnp.max(new_cumulative)
        probs = jax.nn.softmax(logits)
        
        # 2. 奖励 = 正确类别对应的概率
        #    这样，即使预测错误，只要正确类别的概率在增加，网络就会得到正反馈
        reward_value = probs[state.current_label]
        
        # 3. 仅在回合结束时给予奖励
        reward = jnp.where(done, reward_value, 0.0)
        # =====================================================
        
        return state.replace(
            obs=state.current_image, # 保持输入不变
            reward=reward,
            done=done.astype(jnp.float32),
            time_step=new_time,
            cumulative_output=new_cumulative
        )