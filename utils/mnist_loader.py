import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# === GPU 内存管理配置 (保持不变) ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
# ==============================

def load_mnist_data(split='train'):
    """加载 MNIST 并降维到 14x14"""
    
    ds = tfds.load('mnist', split=split, as_supervised=True, batch_size=-1)
    images, labels = tfds.as_numpy(ds)
    
    # 1. 恢复图片形状 [N, 28, 28, 1]
    images = images.reshape(-1, 28, 28, 1)
    
    # 2. [核心修改] 使用 TF 调整大小到 14x14
    # 这会使用双线性插值
    images_resized = tf.image.resize(images, [14, 14]).numpy()
    
    # 3. 归一化并展平 [N, 196]
    # 196 = 14 * 14
    images_flat = images_resized.reshape(-1, 196).astype(np.float32) / 255.0
    
    # 4. (可选) 增强对比度，利于泊松发放
    # images_flat = images_flat * 1.2 
    # images_flat = np.clip(images_flat, 0, 1)

    labels = labels.astype(np.int32)
    
    print(f"MNIST 数据已加载并降维: shape={images_flat.shape}")
    
    return jnp.array(images_flat), jnp.array(labels)