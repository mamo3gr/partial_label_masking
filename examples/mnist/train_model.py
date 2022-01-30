from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from examples.utils import get_y_true, set_gpu_memory_growth


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

    @tf.function
    def test_step(images, target_vectors):
        predictions = model(images)
        t_loss = loss_object(target_vectors, predictions)

        test_loss(t_loss)
        test_accuracy(target_vectors, predictions)

    n_epochs = 5
    for epoch in range(n_epochs):
        for images, target_vectors in train_ds:
            mask = generate_mask(target_vectors, positive_ratio, ideal_positive_ratio)
            train_step(images, target_vectors, mask)

        for test_images, test_target_vectors in test_ds:
            test_step(test_images, test_target_vectors)

        print(
            f"Epoch {epoch}, "
            f"Loss: {train_loss.result():.4f}, "
            f"Accuracy: {train_accuracy.result():.4f}, "
            f"Test Loss: {test_loss.result():.4f}, "
            f"Test Accuracy: {test_accuracy.result():.4f}"
        )

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


def generate_mask(y_true, positive_ratio, positive_ratio_ideal):
    n_samples = y_true.shape[0]

    over_predicted = tf.stack([positive_ratio > positive_ratio_ideal] * n_samples)
    under_predicted = tf.math.logical_not(over_predicted)

    prob_for_over_predicted = tf.stack(
        [positive_ratio_ideal / positive_ratio] * n_samples
    )
    prob_for_under_predicted = 1.0 / prob_for_over_predicted

    ones_for_over_predicted = multi_hot_with_prob(
        prob_for_over_predicted, shape=y_true.shape
    )
    ones_for_under_predicted = multi_hot_with_prob(
        prob_for_under_predicted, shape=y_true.shape
    )

    mask = tf.where((y_true > 0) & over_predicted, ones_for_over_predicted, 1)
    mask = tf.where((y_true == 0) & under_predicted, ones_for_under_predicted, mask)

    return mask


def multi_hot_with_prob(prob, shape):
    """
    Generate tensor where some elements are 1 with a certain probability
    *prob* and the others is 0.

    Args:
        prob: probability. [0, 1]
        shape: output shape.

    Returns:
        ones_with_probability: tensor whose shape is *shape*.
    """
    return tf.where(
        tf.random.uniform(shape=shape, minval=0.0, maxval=1.0) <= prob, 1, 0
    )


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
    y_train = y_train[..., tf.newaxis]
    y_test = y_test[..., tf.newaxis]

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
