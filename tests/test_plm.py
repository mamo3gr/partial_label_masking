import numpy as np
from numpy.testing import assert_allclose

from plm import ProbabilityHistograms


class TestProbabilityHistograms:
    def test_reset(self):
        n_classes = 7
        n_bins = 10

        hist = ProbabilityHistograms(n_classes=n_classes, n_bins=n_bins)
        hist.reset()

        all_zero_histogram = np.zeros((n_bins, n_classes))
        assert_allclose(hist.prediction_positive, all_zero_histogram)
        assert_allclose(hist.prediction_negative, all_zero_histogram)
        assert_allclose(hist.ground_truth_positive, all_zero_histogram)
        assert_allclose(hist.ground_truth_negative, all_zero_histogram)

    def test_update_histogram(self):
        n_classes = 2
        n_bins = 3

        # fmt: off
        y_true = np.array(
            [
                [0, 1],
                [1, 0],
                [1, 0]
            ], np.int)
        y_pred = np.array(
            [
                [0.0, 0.9],
                [0.5, 0.2],
                [0.7, 0.8]
            ], np.float)

        hist_expect_pos_true = np.array(
            [
                [0, 0],
                [0, 0],
                [2, 1]
            ], np.int)
        hist_expect_neg_true = np.array(
            [
                [1, 2],
                [0, 0],
                [0, 0]
            ], np.int)
        hist_expect_pos_pred = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1]
            ], np.int)
        hist_expect_neg_pred = np.array(
            [
                [1, 1],
                [0, 0],
                [0, 1]
            ], np.int)
        # fmt: on

        hist = ProbabilityHistograms(n_classes=n_classes, n_bins=n_bins)
        hist.update_histogram(y_true=y_true, y_pred=y_pred)

        assert_allclose(hist.ground_truth_positive, hist_expect_pos_true)
        assert_allclose(hist.ground_truth_negative, hist_expect_neg_true)
        assert_allclose(hist.prediction_positive, hist_expect_pos_pred)
        assert_allclose(hist.prediction_negative, hist_expect_neg_pred)

    def test_divergence_difference(self):
        n_classes = 2
        n_bins = 3
        # n_samples = 10

        # class[0] is under-predicted while class[1] is over-predicted
        # fmt: off
        hist_pos_true = np.array([
            [1, 6],
            [0, 0],
            [9, 4]
        ])
        hist_pos_pred = np.array([
            [5, 1],
            [4, 1],
            [1, 8]
        ])
        hist_neg_true = np.array([
            [9, 2],
            [0, 0],
            [1, 8]
        ])
        hist_neg_pred = np.array([
            [1, 5],
            [3, 3],
            [6, 2]
        ])
        # fmt: on

        hist = ProbabilityHistograms(n_classes=n_classes, n_bins=n_bins)
        hist.ground_truth_positive = hist_pos_true
        hist.ground_truth_negative = hist_neg_true
        hist.prediction_positive = hist_pos_pred
        hist.prediction_negative = hist_neg_pred

        divergence_difference = hist.divergence_difference()
        assert_allclose(divergence_difference > 0, np.array([True, False]))
