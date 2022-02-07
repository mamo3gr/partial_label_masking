import datetime
import os
import tempfile
from logging import getLogger

import numpy as np
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


def add_regularization(model, regularizer: tf.keras.regularizers.Regularizer):
    """
    add arbitrary regularizer to a model.

    References:
      How to Add Regularization to Keras Pre-trained Models the Right Way
      https://sthalles.github.io/keras-regularizer/
    """
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        logger.warning(
            "Regularizer must be a subclass of tf.keras.regularizers.Regularizer"
        )
        return model

    for layer in model.layers:
        for attr in ["kernel_regularizer"]:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes,
    # the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp_weights.h5")
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_positive_ratio(dataset: tf.data.Dataset) -> np.ndarray:
    y = get_y_true(dataset)
    n_samples = len(y)
    n_positives = np.sum(y > 0, axis=0)
    positive_ratio = n_positives / n_samples
    return positive_ratio


def get_y_true(dataset: tf.data.Dataset):
    """
    References:
        python - Extract target from Tensorflow PrefetchDataset - Stack Overflow
        https://stackoverflow.com/a/68509722
    """
    y = list(map(lambda x: x[1], dataset))
    return np.concatenate(y, axis=0)
