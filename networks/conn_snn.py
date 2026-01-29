import math
from typing import Any  # [新增]

import jax
import jax.numpy as jnp
import flax.linen as nn


def lerp(y, x, alpha):
    # Linear interpolation
    # = alpha * y + (1 - alpha) * x
    # 如果 alpha 是向量 (N,)，x/y 是 (B, N)，JAX 会自动广播
    return x + alpha * (y - x)


def conn_dense(kernel, x):
    # Check dtypes
    assert kernel.dtype == jnp.bool_, "Kernel must be boolean."
    assert x.dtype      != jnp.bool_, "Inputs must not be boolean."

    # matmul
    return jax.lax.dot_general(x, kernel.astype(x.dtype), (((x.ndim - 1,), (0,)), ((), ())))


class ConnSNN(nn.Module):
    """Spiking neural network with connectome only, ExpLIF model"""

    # Network parameters
    out_dims: int
    expected_sparsity: float = 0.5

    num_neurons: int = 256
    excitatory_ratio: float = 0.5

    rand_init_Vm: bool = True

    neuron_dtype: jnp.dtype = jnp.float32
    spike_dtype:  jnp.dtype = jnp.int8

    # SNN simulation
    sim_time: float = 16.6  # ms
    dt: float       = 0.5   # ms

    # SNN parameters
    K_in:  float = 0.1
    K_h:   float = 1.0
    K_out: float = 5.0

    tau_syn:  float = 5.0   # ms
    tau_Vm:   float = 10.0  # ms
    tau_out:  float = 10.0  # ms
    
    # [新增] 接收向量化的 Tau 参数
    tau_Vm_vector: Any = None

    Vr:       float = 0.0
    Vth:      float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        # x 的形状可能是:
        # 1. (Batch, Time, Features) -> 在 _runner_init 中
        # 2. (Time, Features)        -> 在 _evaluate_step (vmap) 中
        
        self.variable("fixed_weights", "dummy", lambda: None)
        
        in_dims = x.shape[-1]
        num_excitatory = round(self.num_neurons * self.excitatory_ratio)

        kernel_in  = self.param("kernel_in",  nn.initializers.zeros, (2 * in_dims, self.num_neurons),      jnp.bool_)
        kernel_h   = self.param("kernel_h",   nn.initializers.zeros, (self.num_neurons, self.num_neurons), jnp.bool_)
        kernel_out = self.param("kernel_out", nn.initializers.zeros, (self.num_neurons, self.out_dims),    jnp.bool_)

        # Parameters
        if self.tau_Vm_vector is not None:
            tau_eff = jnp.array(self.tau_Vm_vector, dtype=self.neuron_dtype)
        else:
            tau_eff = self.tau_Vm

        R_in  = self.K_in  * self.Vth * tau_eff                * math.sqrt(2 / (self.expected_sparsity * in_dims))
        R     = self.K_h   * self.Vth * tau_eff / self.tau_syn * math.sqrt(2 / (self.expected_sparsity * self.num_neurons))
        R_out = self.K_out                                     * math.sqrt(1 / (self.expected_sparsity * self.num_neurons))

        alpha_syn = math.exp(-self.dt / self.tau_syn)
        alpha_out = math.exp(-self.dt / self.tau_out)
        alpha_Vm  = jnp.exp(-self.dt / tau_eff)

        # 1. 计算输入电流
        input_spikes = jnp.concatenate([x, -x], axis=-1)
        
        # 动态收缩：总是收缩 input_spikes 的最后一维 和 kernel_in 的第一维
        i_in_seq = jax.lax.dot_general(
            input_spikes, 
            kernel_in.astype(self.neuron_dtype), 
            (((input_spikes.ndim - 1,), (0,)), ((), ())) 
        )
        i_in_seq = i_in_seq * R_in

        # ================= [核心修复逻辑] =================
        # 根据维度判断是否需要交换轴，以确保 Time 轴始终在 axis 0
        
        if i_in_seq.ndim == 3: 
            # Case A: (Batch, Time, N) -> 来自 _runner_init
            # 我们需要让 scan 遍历 Time，所以把 Time (axis 1) 移到 axis 0
            # 变换后: (Time, Batch, N)
            i_in_seq = jnp.swapaxes(i_in_seq, 0, 1)
            
        # Case B: (Time, N) -> 来自 vmap
        # Time 已经在 axis 0，无需操作
        
        # =================================================

        # 2. 定义扫描函数
        def _snn_step_seq(loop_carry, i_in_t):
            # i_in_t 的形状现在保证是正确的：
            # Case A: (Batch, N)
            # Case B: (N,)
            # 这与 carry 中的 v_m 形状完美匹配
            
            v_m, i_syn, rate, spike = loop_carry

            i_spike = R * conn_dense(kernel_h, spike).astype(self.neuron_dtype)
            i_syn   = i_syn * alpha_syn + i_spike
            
            # 这里原本会报错的加法现在安全了
            v_m     = lerp(v_m, self.Vr + i_syn + i_in_t, alpha_Vm)

            spike_bool = v_m > self.Vth
            v_m        = jnp.where(spike_bool, self.Vr, v_m)

            spike_exc, spike_inh = jnp.split(spike_bool.astype(self.spike_dtype), [num_excitatory], axis=-1)
            spike = jnp.concatenate([spike_exc, -spike_inh], axis=-1)

            rate = lerp(rate, (1 / self.dt) * spike.astype(rate.dtype), alpha_out)
            
            return (v_m, i_syn, rate, spike), rate

        # 3. 执行扫描
        final_carry, rate_seq = jax.lax.scan(
            _snn_step_seq, 
            carry, 
            i_in_seq
        )

        rate_mean = jnp.mean(rate_seq, axis=0) # (N,)
        
        # [新增] 构建一个只允许兴奋性神经元输出的 mask
        # 假设前 num_excitatory 个是兴奋性的 (根据之前的 E-I 排序)
        # exc_mask: [1, 1, ..., 0, 0]
        exc_mask = jnp.arange(self.num_neurons) < num_excitatory
        # 扩展到 (N, 1) 以便广播
        exc_mask = jnp.expand_dims(exc_mask, -1) 
        
        # 应用 mask: 抑制性神经元的权重被强制视为 0
        masked_kernel_out = kernel_out & exc_mask
        
        output = R_out * conn_dense(masked_kernel_out, rate_mean)
        return final_carry, output
        
    def initial_carry(self, key: jax.random.PRNGKey, batch_size: int):
        v_m   = jnp.full((batch_size, self.num_neurons), self.Vr, self.neuron_dtype)
        i_syn = jnp.zeros((batch_size, self.num_neurons),         self.neuron_dtype)
        rate  = jnp.zeros((batch_size, self.num_neurons),         self.neuron_dtype)
        spike = jnp.zeros((batch_size, self.num_neurons),         self.spike_dtype)

        if self.rand_init_Vm:
            # Random init Vm to [Vr, Vth]
            v_m = jax.random.uniform(key, (batch_size, self.num_neurons), self.neuron_dtype, self.Vr, self.Vth)

        return v_m, i_syn, rate, spike

    def carry_metrics(self, carry):
        v_m, i_syn, rate, spike = carry

        return {
            "spikes_per_ms": jnp.mean(jnp.abs(rate)),
            "avg_i_syn":     jnp.mean(jnp.abs(i_syn))
        }