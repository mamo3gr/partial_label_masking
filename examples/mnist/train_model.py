from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from examples.utils import get_y_true, set_gpu_memory_growth
from plm import ProbabilityHistograms, generate_mask

eps = tf.keras.backend.epsilon()


def train_model():
    set_gpu_memory_growth()

    train_ds, test_ds = _create_train_and_test_datasets()

    model = MyModel()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_accuracy")

    positive_ratio = _get_positive_ratio(train_ds).astype(np.float32)
    ideal_positive_ratio = np.full_like(positive_ratio, 0.5).astype(np.float32)
    change_rate = 1e-2
    n_bins = 10
    n_classes = 10

    invoked_date = date_string()
    train_summary_writer = tf.summary.create_file_writer(
        logdir=f"./logs/mnist/{invoked_date}/train"
    )
    test_summary_writer = tf.summary.create_file_writer(
        logdir=f"./logs/mnist/{invoked_date}/test"
    )

    @tf.function
    def train_step(images, target_vectors, mask):
        with tf.GradientTape() as tape:
            predictions = model(images)
            prediction_loss = loss_object(target_vectors, predictions)
            prediction_loss *= tf.cast(mask, prediction_loss.dtype)
            regularization_loss = tf.math.add_n(model.losses)
            total_loss = prediction_loss + regularization_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(total_loss)
        train_accuracy(target_vectors, predictions)
        return predictions

    @tf.function
    def test_step(images, target_vectors):
        predictions = model(images)
        t_loss = loss_object(target_vectors, predictions)

        test_loss(t_loss)
        test_accuracy(target_vectors, predictions)

    hist = ProbabilityHistograms(n_classes=n_classes, n_bins=n_bins)

    n_epochs = 10
    for epoch in range(n_epochs):
        for images, target_vectors in train_ds:
            mask = generate_mask(target_vectors, positive_ratio, ideal_positive_ratio)
            predictions = train_step(images, target_vectors, mask)
            hist.update_histogram(target_vectors, predictions)

        divergence_difference = hist.divergence_difference()
        ideal_positive_ratio *= np.exp(change_rate * divergence_difference)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

        for test_images, test_target_vectors in test_ds:
            test_step(test_images, test_target_vectors)

        with test_summary_writer.as_default():
            tf.summary.scalar("loss", test_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)

        print(
            f"Epoch {epoch}, "
            f"Loss: {train_loss.result():.4f}, "
            f"Accuracy: {train_accuracy.result():.4f}, "
            f"Test Loss: {test_loss.result():.4f}, "
            f"Test Accuracy: {test_accuracy.result():.4f}"
        )

        with np.printoptions(precision=3, threshold=n_classes):
            print(f"ideal_positive_ratio: {ideal_positive_ratio}")

        hist.reset()
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(
            32, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(5e-4)
        )
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="sigmoid")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def _get_positive_ratio(ds):
    y = get_y_true(ds)
    n_samples = len(y)
    n_positives = np.sum(y > 0, axis=0)
    positive_ratio = n_positives / n_samples
    return positive_ratio


def _create_train_and_test_datasets() -> Tuple[Any, Any]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    n_classes = 10
    encoder = tf.keras.layers.CategoryEncoding(
        num_tokens=n_classes, output_mode="one_hot"
    )
    y_train = encoder(y_train.astype(np.int64))
    y_test = encoder(y_test.astype(np.int64))

    train_ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(x_train),
            tf.data.Dataset.from_tensor_slices(y_train),
        )
    )
    shuffle_buffer_size = len(train_ds)
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    batch_size = 8192
    train_ds = train_ds.batch(batch_size=batch_size)

    test_ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(x_test),
            tf.data.Dataset.from_tensor_slices(y_test),
        )
    ).batch(batch_size)

    return train_ds, test_ds


if __name__ == "__main__":
    train_model()
