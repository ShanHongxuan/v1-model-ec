import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional

# [修改] 将有默认值的 metrics 字段移到最后
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
        
        self._images = images # [修改] 直接存储
        self._labels = labels
        self._num_data = self._images.shape[0]
        
        self.observation_size = 784
        self.action_size = 10

    def _load_data(self):
        """只在第一次需要时加载数据"""
        if self._images is None:
            from utils.mnist_loader import load_mnist_data
            self._images, self._labels = load_mnist_data('train')
            self._num_data = self._images.shape[0]

    def reset(self, rng: jnp.ndarray) -> EnvState:
        self._load_data() # 确保数据已加载
        
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
            # metrics 会自动使用 default_factory
        )

    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """
        action: 网络的输出 (10维向量, 通常是发放率或电压)
        """
        new_cumulative = state.cumulative_output + action
        new_time = state.time_step + 1
        done = new_time >= self.presentation_steps
        
        pred_label = jnp.argmax(new_cumulative)
        is_correct = (pred_label == state.current_label)
        
        reward = jnp.where(done, is_correct.astype(jnp.float32), 0.0)
        
        return state.replace(
            obs=state.current_image, # 保持输入不变
            reward=reward,
            done=done.astype(jnp.float32),
            time_step=new_time,
            cumulative_output=new_cumulative
        )