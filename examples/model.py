from logging import getLogger
from typing import Tuple

import tensorflow as tf
import tensorflow.keras.applications
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from examples.utils import add_regularization

logger = getLogger(__name__)


def create_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    weights_decay: float,
    backbone_class,
    use_pretrain=False,
):
    weights = None
    if use_pretrain:
        weights = "imagenet"

    backbone = backbone_class(
        input_shape=input_shape, include_top=False, pooling=None, weights=weights
    )

    backbone = add_regularization(backbone, _regularizer(weights_decay=weights_decay))

    if use_pretrain:
        backbone.trainable = False

    x_in = Input(shape=input_shape, name="input")
    x = x_in

    if use_pretrain:
        x = backbone(x, training=False)
    else:
        x = backbone(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(
        1024,
        activation="relu",
        kernel_regularizer=_regularizer(weights_decay=weights_decay),
    )(x)
    x = Dropout(0.2)(x)
    x = Dense(
        512,
        activation="relu",
        kernel_regularizer=_regularizer(weights_decay=weights_decay),
    )(x)
    x = Dropout(0.2)(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=_regularizer(weights_decay=weights_decay),
    )(x)
    x = Dropout(0.2)(x)
    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=_regularizer(weights_decay=weights_decay),
    )(x)
    x = Dropout(0.2)(x)
    x = Dense(
        n_classes,
        activation="sigmoid",
        kernel_regularizer=_regularizer(weights_decay=weights_decay),
    )(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x)

    return model


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


if __name__ == "__main__":
    model = create_model(
        input_shape=(224, 224, 3),
        n_classes=7,
        weights_decay=5e-4,
        backbone_class=tensorflow.keras.applications.ResNet50V2,
        use_pretrain=False,
    )
    print(model.summary())
