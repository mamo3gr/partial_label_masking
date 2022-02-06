from logging import getLogger
from typing import Optional

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class DatasetGenerator:
    def __init__(
        self,
        image_height: int,
        image_width: int,
        batch_size: int,
        drop_reminder=False,
        shuffle=False,
        random_seed: Optional[int] = None,
        shuffle_buffer_size=None,
        preprocess_func=None,
        augment=None,
        logger=None,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.drop_reminder = drop_reminder
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment = augment

        def no_operation(x):
            return x

        if preprocess_func is None:
            preprocess_func = no_operation
        self.preprocess_func = preprocess_func

        if logger is None:
            logger = getLogger(__name__)
        self._logger = logger

    def generate(self, paths, y):
        ds = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(paths),
                tf.data.Dataset.from_tensor_slices(tf.cast(y, tf.int64)),
            )
        )

        if self.shuffle:
            shuffle_buffer_size = self.shuffle_buffer_size
            if shuffle_buffer_size is None:
                shuffle_buffer_size = len(ds)

            self._logger.info(
                f"Would be shuffled with buffer whose size is {shuffle_buffer_size}"
            )
            ds = ds.shuffle(
                buffer_size=shuffle_buffer_size,
                reshuffle_each_iteration=True,
                seed=self.random_seed,
            )

        ds = ds.map(
            lambda path, y: (self.load_and_preprocess_image(path), y),
            num_parallel_calls=AUTOTUNE,
        )

        if self.augment:
            ds = ds.map(lambda x, y: (self.augment(x), y), num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=self.drop_reminder)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def load_and_preprocess_image(self, path):
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = self.preprocess_image(img)
        return img

    def preprocess_image(self, img):
        img = tf.image.resize(img, (self.image_height, self.image_width))
        img = self.preprocess_func(img)
        return img
