import jax.numpy as jnp
import numpy as np
import tensorflow as tf  # [新增]
import tensorflow_datasets as tfds

# [新增] === GPU 内存管理配置 ===
# 获取所有可用的物理 GPU 设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # 对每个 GPU 设置内存增长模式
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow 内存增长模式已为 {len(gpus)} 个GPU启用。")
  except RuntimeError as e:
    # 内存增长必须在 GPU 初始化之前设置
    print(e)
# ==============================


def load_mnist_data(split='train'):
    """加载 MNIST 数据并预处理为 JAX 友好的格式"""
    
    # 确保 TensorFlow 在加载数据时不会占用所有 GPU 显存
    # （上面的代码已经处理了）
    
    ds = tfds.load(
        'mnist',
        split=split,
        as_supervised=True,
        batch_size=-1
    )
    
    images, labels = tfds.as_numpy(ds)
    
    # 归一化: [N, 28, 28, 1] -> [N, 784]
    images = images.astype(np.float32).reshape(-1, 784) / 255.0
    labels = labels.astype(np.int32)
    
    return jnp.array(images), jnp.array(labels)