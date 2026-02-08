# envs/mnist_env.py

import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class EnvState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    current_label: jnp.ndarray
    metrics: dict = struct.field(default_factory=dict)

class MnistEnv:
    def __init__(self, images, labels, input_hz=200.0, dt_ms=0.5, 
                 # [新增] 时间阶段定义 (单位: step)
                 steps_pre_stim=100,  # 0-50ms
                 steps_stim=200,      # 50-150ms
                 steps_response=100): # 150-200ms
        
        self.presentation_steps = 1 # 对外部来说是一步
        
        self._images = images
        self._labels = labels
        self._num_data = self._images.shape[0]
        
        self.prob_per_step_max = input_hz * (dt_ms / 1000.0)
        self.observation_size = self._images.shape[1]
        self.action_size = 10
        
        # 记录各阶段长度
        self.t_pre = steps_pre_stim
        self.t_stim = steps_stim
        self.t_resp = steps_response
        self.total_steps = self.t_pre + self.t_stim + self.t_resp

    def reset(self, rng: jnp.ndarray) -> EnvState:
        rng, img_key, poisson_key = jax.random.split(rng, 3)
        
        idx = jax.random.randint(img_key, (), 0, self._num_data)
        image = self._images[idx]
        label = self._labels[idx]
        
        # --- [核心修改] 构造三段式输入概率 ---
        
        # 1. 基础概率图 (1, Features)
        base_probs = jnp.expand_dims(image * self.prob_per_step_max, 0)
        
        # 2. 静默概率图 (1, Features) -> 全 0
        silence = jnp.zeros_like(base_probs)
        
        # 3. 按时间拼接: [静默(50ms), 刺激(100ms), 静默(50ms)]
        # expand & repeat 是为了高效构造
        seq_pre  = jnp.repeat(silence, self.t_pre, axis=0)
        seq_stim = jnp.repeat(base_probs, self.t_stim, axis=0)
        seq_resp = jnp.repeat(silence, self.t_resp, axis=0)
        
        # 完整的概率序列 (400, 196)
        full_probs = jnp.concatenate([seq_pre, seq_stim, seq_resp], axis=0)
        
        # 4. 泊松采样
        obs_sequence = jax.random.bernoulli(poisson_key, full_probs).astype(jnp.float32)
        
        return EnvState(
            obs=obs_sequence, 
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            current_label=label
        )

    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        # action 是 ConnSNN 计算出的 Readout Window 内的平均发放率
        logits = action - jnp.max(action)
        probs = jax.nn.softmax(logits)
        reward = probs[state.current_label]
        
        return state.replace(
            reward=reward,
            done=jnp.float32(1.0) 
        )