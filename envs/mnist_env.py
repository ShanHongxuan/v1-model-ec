# envs/mnist_env.py

import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class EnvState:
    # --- 必须与 reset 返回值匹配 ---
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    current_label: jnp.ndarray
    metrics: dict = struct.field(default_factory=dict)

class MnistEnv:
    def __init__(self, images, labels, presentation_steps=100, input_hz=100.0, dt_ms=0.5):
        # [核心] 这里记录总步数，用于生成张量
        self.total_steps = presentation_steps 
        # 对 ec.py 来说，这个环境只需要 "1步" 就跑完了整个仿真
        self.presentation_steps = 1 
        
        self._images = images
        self._labels = labels
        self._num_data = self._images.shape[0]
        
        # 计算发放概率
        self.prob_per_step_max = input_hz * (dt_ms / 1000.0)
        
        self.observation_size = self._images.shape[1]
        self.action_size = 10

    def reset(self, rng: jnp.ndarray) -> EnvState:
        # 分割随机数
        rng, img_key, poisson_key = jax.random.split(rng, 3)
        
        # 1. 随机选图
        idx = jax.random.randint(img_key, (), 0, self._num_data)
        image = self._images[idx] # (196,)
        label = self._labels[idx]
        
        # 2. [核心优化] 一次性生成所有时间步的脉冲
        # 扩展 image 维度: (196,) -> (1, 196)
        probs = image * self.prob_per_step_max
        probs = jnp.expand_dims(probs, axis=0) 
        # 广播到时间维度: (1, 196) -> (Time, 196)
        probs = jnp.repeat(probs, self.total_steps, axis=0)
        
        # 3. 伯努利采样: 得到形状为 (Time, 196) 的脉冲序列
        # 这就是这一整局的全部输入
        obs_sequence = jax.random.bernoulli(poisson_key, probs).astype(jnp.float32)
        
        return EnvState(
            obs=obs_sequence, # 这是一个巨大的 3D 张量 (Batch, Time, 196)
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            current_label=label
        )

    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        # action 是 SNN 跑完整个序列后的最终结果
        
        # Softmax 奖励
        logits = action - jnp.max(action)
        probs = jax.nn.softmax(logits)
        reward = probs[state.current_label]
        
        return state.replace(
            reward=reward,
            done=jnp.float32(1.0) # 一步即终局
        )