import math
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

def lerp(y, x, alpha):
    return x + alpha * (y - x)

def conn_dense(kernel, x):
    # kernel: bool matrix
    # x: float vector/matrix
    # 自动处理 x 的维度，总是沿着 x 的最后一维和 kernel 的第一维做乘法
    return jax.lax.dot_general(
        x, 
        kernel.astype(x.dtype), 
        (((x.ndim - 1,), (0,)), ((), ()))
    )

class ConnSNN(nn.Module):
    out_dims: int
    num_neurons: int = 256
    excitatory_ratio: float = 0.5
    expected_sparsity: float = 0.5
    rand_init_Vm: bool = True
    neuron_dtype: jnp.dtype = jnp.float32
    spike_dtype: jnp.dtype = jnp.int8 # Spike 是 0/1 事件
    
    dt: float = 0.5
    # 默认值 (会被 ec.py 覆盖)
    K_in: float = 10.0
    K_h: float = 1.0
    K_out: float = 100.0
    
    tau_syn: float = 5.0
    tau_Vm: float = 10.0
    tau_out: float = 10.0
    tau_Vm_vector: Any = None
    
    Vr: float = 0.0
    Vth: float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        self.variable("fixed_weights", "dummy", lambda: None)
        in_dims = x.shape[-1]
        num_excitatory = round(self.num_neurons * self.excitatory_ratio)

        kernel_in = self.param("kernel_in", nn.initializers.zeros, (in_dims, self.num_neurons), jnp.bool_)
        kernel_h = self.param("kernel_h", nn.initializers.zeros, (self.num_neurons, self.num_neurons), jnp.bool_)
        kernel_out = self.param("kernel_out", nn.initializers.zeros, (self.num_neurons, self.out_dims), jnp.bool_)

        if self.tau_Vm_vector is not None:
            tau_eff = jnp.array(self.tau_Vm_vector, dtype=self.neuron_dtype)
        else:
            tau_eff = self.tau_Vm

        R_in = self.K_in * self.Vth * tau_eff * math.sqrt(1 / (self.expected_sparsity * in_dims))
        R = self.K_h * self.Vth * tau_eff / self.tau_syn * math.sqrt(1 / (self.expected_sparsity * self.num_neurons))
        R_out = self.K_out * math.sqrt(1 / (self.expected_sparsity * self.num_neurons))

        alpha_syn = math.exp(-self.dt / self.tau_syn)
        alpha_out = math.exp(-self.dt / self.tau_out)
        alpha_Vm = jnp.exp(-self.dt / tau_eff)

        # ================= [新增] 动态输入能量归一化 =================
        # x 是泊松脉冲 (0 或 1)
        # 1. 计算每个时刻、每个样本的激活像素总数
        # axis=-1 是特征维度 (196)
        active_pixel_count = jnp.sum(x, axis=-1, keepdims=True)
        
        # 2. 设定目标能量水平
        # MNIST 数字 '1' 大约有 15-20 个像素，'0' 大约有 40-50 个。
        # 我们把所有输入都拉到 30.0 这个基准线上。
        target_activity = 30.0
        
        # 3. 计算缩放系数
        # 加上 1.0 防止除零 (针对全黑图片)
        scale_factor = target_activity / (active_pixel_count + 1.0)
        
        # 4. 安全截断 (Clip)
        # 防止微弱噪声被放大成巨型信号，也防止全黑输入导致 NaN
        # 允许放大倍数在 0 到 10 倍之间
        scale_factor = jnp.clip(scale_factor, 0.0, 10.0)
        
        # 5. 应用归一化
        # 现在，无论是 0 还是 1，x_norm 携带的总能量都是差不多的
        x_norm = x * scale_factor
        # =============================================================

        # --- 输入层计算 ---
        # [修改] 使用归一化后的 x_norm
        i_in_seq = conn_dense(kernel_in, x_norm) 
        i_in_seq = i_in_seq * R_in

        # 维度处理 (Time-Tensorization 兼容)
        if i_in_seq.ndim == 3:
            i_in_seq = jnp.swapaxes(i_in_seq, 0, 1)

        # --- 单步动力学 ---
        def _snn_step_seq(loop_carry, i_in_t):
            v_m, i_syn, rate, spike = loop_carry

            # --- [逻辑修正 2] Dale's Law ---
            # spike 是 0/1。我们需要在这里给它赋予正负号。
            exc_mask = jnp.arange(self.num_neurons) < num_excitatory
            # 广播 exc_mask 以匹配 Batch 维度 (如果有)
            dale_vector = jnp.where(exc_mask, 1.0, -1.0)
            
            signed_spike = spike.astype(self.neuron_dtype) * dale_vector
            
            i_spike = R * conn_dense(kernel_h, signed_spike)
            
            # --- 积分 ---
            i_syn = i_syn * alpha_syn + i_spike
            v_m = lerp(v_m, self.Vr + i_syn + i_in_t, alpha_Vm)

            spike_bool = v_m > self.Vth
            v_m = jnp.where(spike_bool, self.Vr, v_m)
            
            # 状态更新 (spike 保持 0/1)
            spike = spike_bool.astype(self.spike_dtype)
            rate = lerp(rate, (1 / self.dt) * spike.astype(rate.dtype), alpha_out)
            
            return (v_m, i_syn, rate, spike), rate

        # --- 扫描 ---
        final_carry, rate_seq = jax.lax.scan(_snn_step_seq, carry, i_in_seq)
        
        # --- [逻辑修正 3] 输出层 ---
        # 沿时间轴平均
        rate_mean = jnp.mean(rate_seq, axis=0)
        
        # 仅允许兴奋性神经元输出
        exc_mask_out = jnp.arange(self.num_neurons) < num_excitatory
        exc_mask_out = jnp.expand_dims(exc_mask_out, -1) # (N, 1)
        
        masked_kernel_out = kernel_out & exc_mask_out
        
        output = R_out * conn_dense(masked_kernel_out, rate_mean)
        
        return final_carry, output

    # --- [补回] 缺失的方法 ---
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