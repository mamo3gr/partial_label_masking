import os
import tempfile
from logging import getLogger
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input

logger = getLogger(__name__)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def create_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    weights_decay: float,
    use_pretrain=True,
):
    weights = None
    if use_pretrain:
        weights = "imagenet"

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    backbone = add_regularization(backbone, _regularizer(weights_decay=weights_decay))

    x_in = Input(shape=input_shape, name="input")
    x = x_in
    x = backbone(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(
        n_classes,
        activation="sigmoid",
        kernel_regularizer=_regularizer(weights_decay=weights_decay),
    )(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x)

    return model


def add_regularization(model, regularizer: tf.keras.regularizers.Regularizer):
    """
    add arbitrary regularizer to a model.
    References:
      How to Add Regularization to Keras Pre-trained Models the Right Way
      https://sthalles.github.io/keras-regularizer/
    Args:
        model:
        regularizer:
    Returns:
        model_new
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
