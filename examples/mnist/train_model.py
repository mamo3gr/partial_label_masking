from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from examples.utils import set_gpu_memory_growth


def train_model():
    set_gpu_memory_growth()

    train_ds, test_ds = _create_train_and_test_datasets()

    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    n_epochs = 5
    for epoch in range(n_epochs):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

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
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def _create_train_and_test_datasets() -> Tuple[Any, Any]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    shuffle_buffer_size = len(train_ds)
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    batch_size = 1024
    train_ds = train_ds.batch(batch_size=batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, test_ds


if __name__ == "__main__":
    train_model()
