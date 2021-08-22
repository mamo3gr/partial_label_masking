import numpy as np
import tensorflow as tf


class PartialLabelMaskingLoss(tf.keras.losses.Loss):
    def __init__(self, positive_ratio, change_rate, n_bins=10, **kwargs):
        super(PartialLabelMaskingLoss, self).__init__(**kwargs)
        self._eps = tf.keras.backend.epsilon()

        self.positive_ratio = tf.convert_to_tensor(positive_ratio, dtype=tf.float32)
        self.positive_ratio_ideal = tf.convert_to_tensor(
            positive_ratio, dtype=tf.float32
        )
        self.change_rate = change_rate
        self.n_bins = n_bins
        self.edges = tf.linspace(start=0.0, stop=1.0 + self._eps, num=n_bins + 1)

    def call(self, y_true, y_pred):
        # sample- and element-(class-) wise binary cross entropy
        y_true = tf.cast(y_true, tf.float32)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        # mask it
        mask = self.generate_mask(y_true)
        bce *= tf.cast(mask, tf.float32)

        return tf.reduce_sum(bce)

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

    def update_ratio(self):
        prob_diff = self._compute_probabilities_difference()
        self.positive_ratio_ideal *= tf.exp(self.change_rate * prob_diff)

    def _compute_probabilities_difference(self):
        # normalize histogram
        self.prob_hist_true /= tf.reduce_sum(self.prob_hist_true)
        self.prob_hist_pred /= tf.reduce_sum(self.prob_hist_pred)

        kl_div = self._kullback_leibler_divergence(
            self.prob_hist_pred, self.prob_hist_true
        )

    def _kullback_leibler_divergence(self, p, q):
        """
        Kullback-Leibler divergence.

        Args:
            p, q: discrete probability distributions, whose shape is (n_bins, n_classes)

        Returns:
            kl_div: Kullback-Leibler divergence (relative entropy from q to p)
        """
        # FIXME: affirm calculation of KL divergence when q contains zero(s).
        q = tf.where(q > 0, q, self._eps)
        kl_div = tf.reduce_sum(p * tf.math.log(p / q), axis=0)
        return kl_div

    def _compute_histogram(self, y_true, y_pred):
        n_classes = y_true.shape[1]
        n_bins = self.n_bins
        value_range = [0.0, 1.0]

        hist_pos_true = np.zeros((n_bins, n_classes), np.int)
        hist_neg_true = np.zeros((n_bins, n_classes), np.int)
        hist_pos_pred = np.zeros((n_bins, n_classes), np.int)
        hist_neg_pred = np.zeros((n_bins, n_classes), np.int)

        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

        for class_i in range(n_classes):
            y_true_class = y_true[:, class_i]
            y_pred_class = y_pred[:, class_i]

            pos_indices = y_true_class > 0
            neg_indices = ~pos_indices

            # NOTE: np.histogram returns *hist* and *bin_edges*
            hist_pos_true[:, class_i], _ = np.histogram(
                y_true_class[pos_indices], n_bins, value_range
            )
            hist_neg_true[:, class_i], _ = np.histogram(
                y_true_class[neg_indices], n_bins, value_range
            )
            hist_pos_pred[:, class_i], _ = np.histogram(
                y_pred_class[pos_indices], n_bins, value_range
            )
            hist_neg_pred[:, class_i], _ = np.histogram(
                y_pred_class[neg_indices], n_bins, value_range
            )

        return hist_pos_true, hist_neg_true, hist_pos_pred, hist_neg_pred
