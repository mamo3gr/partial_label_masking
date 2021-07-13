from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Union

import tensorflow as tf
from imdb_posters import load_dataset
from model import create_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from plm import PartialLabelMaskingLoss
from utils import set_gpu_memory_growth

logger = getLogger(__name__)
basicConfig(level=INFO)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_and_preprocess_image(path, target_height, target_width):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = preprocess_image(img, target_height, target_width)
    return img


def preprocess_image(img, target_height, target_width):
    img = tf.image.resize(img, (target_height, target_width))
    img = preprocess_input(img)
    return img


def main(root_dir: Union[str, Path]):
    set_gpu_memory_growth()

    image_height = 224
    image_width = 224
    input_shape = (image_height, image_width, 3)
    n_classes = 13
    weight_decay = 5e-4
    momentum = 0.9
    batch_size = 128
    epochs = 40
    model_path = "models/weights_simple.hdf5"

    paths_train, paths_test, y_train, y_test = load_dataset(root_dir)
    ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(paths_train).map(
                lambda x: load_and_preprocess_image(x, image_height, image_width),
                num_parallel_calls=AUTOTUNE,
            ),
            tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.int64)),
        )
    )

    ds_size = len(paths_train)
    valid_size = int(ds_size * 0.05)
    logger.info(f"# of validation samples: {valid_size}/{ds_size}")
    valid_ds = ds.take(valid_size).batch(batch_size).prefetch(AUTOTUNE)
    train_ds = (
        ds.skip(valid_size)
        .shuffle(buffer_size=ds_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    model = create_model(
        input_shape=input_shape,
        n_classes=n_classes,
        weights_decay=weight_decay,
        use_pretrain=False,
    )

    optimizer = tf.keras.optimizers.SGD(momentum=momentum)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        # "./model/weights.{epoch:03d}-{val_loss:.3f}.hdf5",
        model_path,
        monitor="val_loss",
        save_best_only=True,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)

    def scheduler(epoch, lr):
        if epoch < 10:
            return 1e-1
        elif epoch < 20:
            return 1e-2
        elif epoch < 30:
            return 1e-3
        else:
            return 1e-4

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
    )
    model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=[
            lr_scheduler,
            model_checkpoint,
            tensorboard_callback,
        ],
        verbose=1,
    )


class UpdateCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.loss.update()
        self.model.loss.test()


if __name__ == "__main__":
    root_dir = Path("/home/mamo/datasets/img_41K")
    main(root_dir)
