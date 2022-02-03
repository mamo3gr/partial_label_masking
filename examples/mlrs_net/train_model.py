from argparse import ArgumentParser
from copy import deepcopy
from logging import INFO, basicConfig, getLogger
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

from examples.config import Config, load_config
from examples.dataframe_loader import DataFrameLoader
from examples.dataset_generator import DatasetGenerator
from examples.model import create_model
from examples.utils import date_string, get_positive_ratio, set_gpu_memory_growth
from plm import ProbabilityHistograms, generate_mask

logger = getLogger(__name__)


def train_model(conf: Config):
    set_gpu_memory_growth()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, valid_ds = _create_train_valid_set(conf)
    model = _create_model(conf)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = Adam(learning_rate=1e-3)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")
    valid_loss = tf.keras.metrics.Mean(name="valid_loss")
    valid_accuracy = tf.keras.metrics.BinaryAccuracy(name="valid_accuracy")

    positive_ratio = get_positive_ratio(train_ds).astype(np.float32)
    ideal_positive_ratio = deepcopy(positive_ratio)
    change_rate = 1e-2
    n_bins = 10
    labels = conf.labels
    n_classes = len(labels)
    hist = ProbabilityHistograms(n_classes=n_classes, n_bins=n_bins)

    train_summary_writer, valid_summary_writer = _setup_summary_writers()

    n_epochs = conf.epochs

    @tf.function
    def train_step(images, target_vectors, mask):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            prediction_loss = loss_object(target_vectors, predictions)
            prediction_loss *= tf.cast(mask, prediction_loss.dtype)
            regularization_loss = tf.math.add_n(model.losses)
            total_loss = prediction_loss + tf.cast(
                regularization_loss, prediction_loss.dtype
            )
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(total_loss)
        train_accuracy(target_vectors, predictions)
        return predictions

    @tf.function
    def valid_step(images, target_vectors):
        predictions = model(images)
        v_loss = loss_object(target_vectors, predictions)

        valid_loss(v_loss)
        valid_accuracy(target_vectors, predictions)

    n_steps = len(train_ds)
    for epoch in range(n_epochs):
        prog_bar = Progbar(n_steps)

        for i_step, (x_train, y_train) in enumerate(train_ds):
            mask = generate_mask(y_train, positive_ratio, ideal_positive_ratio)
            predictions = train_step(x_train, y_train, mask)
            hist.update_histogram(y_train, predictions)
            prog_bar.update(i_step)

        divergence_difference = hist.divergence_difference()
        ideal_positive_ratio *= np.exp(change_rate * divergence_difference)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

        for x_valid, y_valid in valid_ds:
            valid_step(x_valid, y_valid)

        with valid_summary_writer.as_default():
            tf.summary.scalar("loss", valid_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", valid_accuracy.result(), step=epoch)

        prog_bar.update(n_steps, finalize=True)

        with np.printoptions(precision=3, threshold=n_classes):
            print(f"ideal_positive_ratio: {ideal_positive_ratio}")

        hist.reset()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()


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


def _setup_summary_writers() -> Tuple[Any, Any]:
    example_name = "mlrs_net"
    invoked_date = date_string()
    train_summary_writer = tf.summary.create_file_writer(
        logdir=f"./logs/{example_name}/{invoked_date}/train"
    )
    valid_summary_writer = tf.summary.create_file_writer(
        logdir=f"./logs/{example_name}/{invoked_date}/valid"
    )

    return train_summary_writer, valid_summary_writer


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    basicConfig(level=INFO)
    args = _parse_args()
    config = load_config(args.config_file)
    train_model(config)
