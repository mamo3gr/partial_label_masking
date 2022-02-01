import numpy as np

eps = np.finfo(float).eps


class ProbabilityHistograms:
    def __init__(self, n_classes: int, n_bins: int = 10):
        self.n_classes = n_classes
        self.n_bins = n_bins

        dtype = np.int
        self.ground_truth_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.ground_truth_negative = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_negative = np.zeros((self.n_bins, self.n_classes), dtype)

    def reset(self):
        dtype = np.int
        self.ground_truth_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.ground_truth_negative = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_negative = np.zeros((self.n_bins, self.n_classes), dtype)

    def update_histogram(self, y_true: np.ndarray, y_pred: np.ndarray):
        value_range = [0.0, 1.0]

        for class_i in range(self.n_classes):
            y_true_class = y_true[:, class_i]
            y_pred_class = y_pred[:, class_i]

            pos_indices = y_true_class > 0
            neg_indices = ~pos_indices

            # NOTE: np.histogram returns *hist* and *bin_edges*
            ground_truth_positive, _ = np.histogram(
                y_true_class[pos_indices], self.n_bins, value_range
            )
            self.ground_truth_positive[:, class_i] += ground_truth_positive

            ground_truth_negative, _ = np.histogram(
                y_true_class[neg_indices], self.n_bins, value_range
            )
            self.ground_truth_negative[:, class_i] += ground_truth_negative

            prediction_positive, _ = np.histogram(
                y_pred_class[pos_indices], self.n_bins, value_range
            )
            self.prediction_positive[:, class_i] += prediction_positive

            prediction_negative, _ = np.histogram(
                y_pred_class[neg_indices], self.n_bins, value_range
            )
            self.prediction_negative[:, class_i] += prediction_negative

    def divergence_difference(self):
        divergence_positive = self._divergence_between_histograms(
            self.prediction_positive, self.ground_truth_positive
        )
        divergence_negative = self._divergence_between_histograms(
            self.prediction_negative, self.ground_truth_negative
        )

        divergence_positive = self._standardize_among_classes(divergence_positive)
        divergence_negative = self._standardize_among_classes(divergence_negative)

        return divergence_positive - divergence_negative

    @staticmethod
    def _divergence_between_histograms(hist_pred, hist_true):
        # normalize histogram
        hist_true = hist_true / np.sum(hist_true, axis=0)
        hist_pred = hist_pred / np.sum(hist_pred, axis=0)

        kl_div = kullback_leibler_divergence(hist_pred, hist_true)
        return kl_div

    @staticmethod
    def _standardize_among_classes(x):
        return (x - np.mean(x)) / (np.std(x) + eps)


def kullback_leibler_divergence(p, q):
    """
    Kullback-Leibler divergence.

    Args:
        p, q: discrete probability distributions, whose shape is (n_bins, n_classes)

    Returns:
        kl_div: Kullback-Leibler divergence (relative entropy from q to p)
    """
    q = np.where(q > 0, q, eps)
    kl_div = np.sum(p * np.log(p / q + eps), axis=0)
    return kl_div
