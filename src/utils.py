from logging import getLogger

import tensorflow as tf

logger = getLogger(__name__)


def set_gpu_memory_growth():
    """
    Reference:
      Use a GPU | TensorFlow Core
      https://www.tensorflow.org/guide/gpu
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logger.info(
                f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}"
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.warning(e)


if __name__ == "__main__":
    set_gpu_memory_growth()
