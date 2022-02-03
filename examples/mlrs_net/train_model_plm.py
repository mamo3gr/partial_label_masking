from argparse import ArgumentParser
from copy import deepcopy
from logging import INFO, basicConfig, getLogger

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

from examples.config import Config, load_config
from examples.mlrs_net.train_model import (
    _create_model,
    _create_train_valid_set,
    _setup_summary_writers,
)
from examples.utils import get_positive_ratio, set_gpu_memory_growth
from src.plm import ProbabilityHistograms, generate_mask

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
            regularization_loss = tf.math.add_n(model.losses)
            total_loss = prediction_loss + tf.cast(
                regularization_loss, prediction_loss.dtype
            )
            total_loss *= tf.cast(mask, total_loss.dtype)
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
    valid_loss_min = np.inf
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
            for label_i, (label, ratio) in enumerate(zip(labels, ideal_positive_ratio)):
                tf.summary.scalar(
                    f"positive ratio ideal ({label_i}, {label})", ratio, step=epoch
                )

        for x_valid, y_valid in valid_ds:
            valid_step(x_valid, y_valid)

        with valid_summary_writer.as_default():
            tf.summary.scalar("loss", valid_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", valid_accuracy.result(), step=epoch)

        valid_loss_current = valid_loss.result()
        if valid_loss_current < valid_loss_min:
            model.save(conf.model_path)
            logger.info(
                f"Lower validation loss "
                f"({valid_loss_current:.4f} < {valid_loss_min:.4f}). "
                f"Save model to {conf.model_path}"
            )
            valid_loss_min = valid_loss_current

        prog_bar.update(n_steps, finalize=True)

        hist.reset()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

    model.save(conf.model_path)
    logger.info(f"Save model to {conf.model_path}")


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    basicConfig(level=INFO)
    args = _parse_args()
    config = load_config(args.config_file)
    train_model(config)
