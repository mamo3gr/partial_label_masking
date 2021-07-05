import tensorflow as tf


class PartialLabelMaskingLoss(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, positive_ratio, **kwargs):
        super(PartialLabelMaskingLoss, self).__init__(**kwargs)
        self.positive_ratio = tf.convert_to_tensor(positive_ratio, dtype=tf.float32)
        self.positive_ratio_ideal = tf.convert_to_tensor(
            positive_ratio, dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        raise NotImplementedError

    def generate_mask(self, y_true):
        n_samples = y_true.shape[0]
        over_predicted = tf.stack(
            [self.positive_ratio > self.positive_ratio_ideal] * n_samples
        )
        under_predicted = tf.math.logical_not(over_predicted)

        prob_for_over_predicted = tf.stack(
            [self.positive_ratio_ideal / self.positive_ratio] * n_samples
        )
        prob_for_under_predicted = 1.0 / prob_for_over_predicted

        ones_for_over_predicted = self.multi_hot_with_prob(
            prob_for_over_predicted, shape=y_true.shape
        )
        ones_for_under_predicted = self.multi_hot_with_prob(
            prob_for_under_predicted, shape=y_true.shape
        )

        mask = tf.where((y_true > 0) & over_predicted, ones_for_over_predicted, 1)
        mask = tf.where((y_true == 0) & under_predicted, ones_for_under_predicted, mask)

        return mask

    @staticmethod
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
