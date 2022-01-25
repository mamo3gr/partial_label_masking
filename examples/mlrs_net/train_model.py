from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from typing import Any, List, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from examples.config import Config, load_config
from examples.dataframe_loader import DataFrameLoader
from examples.dataset_generator import DatasetGenerator
from examples.model import create_model
from examples.utils import date_string, set_gpu_memory_growth

logger = getLogger(__name__)


def train_model(conf: Config):
    set_gpu_memory_growth()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, valid_ds = _create_train_valid_set(conf)
    model = _create_model(conf)
    optimizer = Adam(learning_rate=1e-3)
    callbacks = _create_callbacks(conf)

    labels = conf.labels
    n_classes = len(labels)
    metrics = (
        ["binary_accuracy"]
        + [
            tf.keras.metrics.Precision(class_id=i, name=f"precision_{labels[i]}")
            for i in range(n_classes)
        ]
        + [
            tf.keras.metrics.Recall(class_id=i, name=f"recall_{labels[i]}")
            for i in range(n_classes)
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=metrics,
    )
    model.fit(
        train_ds,
        batch_size=conf.batch_size,
        epochs=conf.epochs,
        validation_data=valid_ds,
        callbacks=callbacks,
        verbose=2,
    )


def _create_train_valid_set(conf: Config) -> Tuple[Any, Any]:
    loader = DataFrameLoader(
        filename_col=conf.filename_col,
        labels=conf.labels,
        image_dir=conf.image_dir,
    )
    paths, y = loader.load(conf.csv_path)

    paths_train, paths_valid, y_train, y_valid = train_test_split(
        paths, y, test_size=conf.validation_ratio, random_state=conf.random_seed
    )
    logger.info(f"# of train samples: {len(y_train)}")
    logger.info(f"# of validation samples: {len(y_valid)}")

    train_gen = DatasetGenerator(
        image_height=conf.image_height,
        image_width=conf.image_width,
        batch_size=conf.batch_size,
        drop_reminder=True,
        shuffle=True,
        random_seed=conf.random_seed,
        preprocess_func=tf.keras.applications.resnet_v2.preprocess_input,
        logger=logger,
    )
    train_ds = train_gen.generate(paths_train, y_train)

    valid_gen = DatasetGenerator(
        image_height=conf.image_height,
        image_width=conf.image_width,
        batch_size=conf.batch_size,
        preprocess_func=tf.keras.applications.resnet_v2.preprocess_input,
        logger=logger,
    )
    valid_ds = valid_gen.generate(paths_valid, y_valid)

    return train_ds, valid_ds


def _create_model(conf: Config):
    labels = conf.labels
    n_classes = len(labels)
    input_shape = (conf.image_height, conf.image_width, 3)
    model = create_model(
        input_shape=input_shape,
        n_classes=n_classes,
        weights_decay=conf.weight_decay,
        backbone_class=tf.keras.applications.ResNet50V2,
        use_pretrain=True,
    )
    return model


def _create_callbacks(conf: Config) -> List[Any]:
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        conf.model_path, monitor="val_loss", save_best_only=conf.save_best_only
    )

    log_dir = f"./logs/{date_string()}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, profile_batch=(10, 20)
    )

    callbacks = [
        model_checkpoint,
        tensorboard_callback,
    ]

    return callbacks


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    basicConfig(level=INFO)
    args = _parse_args()
    config = load_config(args.config_file)
    train_model(config)
