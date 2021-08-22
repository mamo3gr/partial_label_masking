import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
from pytest_mock import MockerFixture

from plm import PartialLabelMaskingLoss


# FIXME: This test would occasionally fail due to randomness
def test_generate_mask():
    n_samples = 1_000_000
    n_positives = np.array([800000, 10000, 500000])  # for each class
    n_negatives = np.array([200000, 990000, 500000])
    y_true = multi_hot_with_n_positives(n_positives, n_samples)

    positive_ratio = n_positives / n_negatives
    positive_ratio_ideal = [0.5, 1.2, 1.0]
    relative_tolerance = 0.05
    change_rate = 1e-2

    loss = PartialLabelMaskingLoss(
        positive_ratio=positive_ratio, change_rate=change_rate
    )
    loss.positive_ratio_ideal = positive_ratio_ideal
    mask_actual = loss.generate_mask(y_true).numpy()

    n_selected_positives = np.sum((y_true > 0) & (mask_actual > 0), axis=0)
    n_selected_negatives = np.sum((y_true == 0) & (mask_actual > 0), axis=0)
    positive_ratio_actual = n_selected_positives / n_selected_negatives

    np.testing.assert_allclose(
        positive_ratio_actual,
        positive_ratio_ideal,
        rtol=relative_tolerance,
    )


def multi_hot_with_n_positives(
    n_positives_for_each_class: np.ndarray, n_samples: int
) -> np.ndarray:
    """
    Generate multi-hot vectors (matrix) that contains specified number of positives
    for each class.

    Args:
        n_positives_for_each_class: 1-D array where each element indicates
                                    number of samples for each class.
        n_samples: number of samples to be generated.

    Returns:
        ones: 2-D array whose shape is (n_samples, len(n_positives_for_each_class)).
    """
    n_negatives_for_each_class = n_samples - n_positives_for_each_class
    return np.stack(
        [
            np.random.permutation(np.array([1] * n_positives + [0] * n_negatives))
            for n_positives, n_negatives in zip(
                n_positives_for_each_class, n_negatives_for_each_class
            )
        ]
    ).T


def test_call(mocker: MockerFixture):
    batch_size = 32
    n_classes = 9

    shape = (batch_size, n_classes)
    mask = np.where(np.random.rand(*shape) > 0.5, 1, 0).astype(np.int)
    y_true = np.where(np.random.rand(*shape) > 0.5, 1, 0).astype(np.int)
    y_pred = np.random.rand(*shape)

    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    bce_expect = np.sum(bce * mask)

    positive_ratio = np.random.rand(n_classes)
    change_rate = 1e-2
    loss = PartialLabelMaskingLoss(
        positive_ratio=positive_ratio, change_rate=change_rate
    )

    mock_generate_mask = mocker.patch.object(
        loss,
        "generate_mask",
        return_value=tf.convert_to_tensor(mask, dtype=tf.int32, name="mask"),
    )

    bce_actual = loss.call(
        tf.convert_to_tensor(y_true, dtype=tf.int32, name="y_true"),
        tf.convert_to_tensor(y_pred, dtype=tf.float32, name="y_pred"),
    )

    mock_generate_mask.assert_called_once()
    np.testing.assert_allclose(bce_actual, bce_expect, rtol=1e-4)


def test_update_ratio(mocker: MockerFixture):
    n_classes = 100
    positive_ratio = np.random.rand(n_classes)
    change_rate = 1e-2

    loss = PartialLabelMaskingLoss(
        positive_ratio=positive_ratio, change_rate=change_rate
    )

    # positive_ratio_ideal should be initialized by positive_ratio
    np.testing.assert_allclose(loss.positive_ratio_ideal.numpy(), positive_ratio)

    probabilities_difference = np.random.rand(n_classes)
    mock_compute_probabilities_difference = mocker.patch.object(
        loss,
        "_compute_probabilities_difference",
        return_value=tf.convert_to_tensor(
            probabilities_difference, dtype=tf.float32, name="prob_diff"
        ),
    )

    loss.update_ratio()
    mock_compute_probabilities_difference.assert_called_once()
    np.testing.assert_allclose(
        loss.positive_ratio_ideal.numpy(),
        positive_ratio * np.exp(change_rate * probabilities_difference),
        rtol=1e-4,
    )


def test__compute_histogram():
    positive_ratio = [0.1, 0.2]
    change_rate = 1e-2
    # n_classes = 2
    n_bins = 3

    loss = PartialLabelMaskingLoss(
        positive_ratio=positive_ratio, change_rate=change_rate, n_bins=n_bins
    )

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

    (
        hist_actual_pos_true,
        hist_actual_neg_true,
        hist_actual_pos_pred,
        hist_actual_neg_pred,
    ) = loss._compute_histogram(
        tf.convert_to_tensor(y_true, tf.int32), tf.convert_to_tensor(y_pred, tf.float32)
    )

    assert_allclose(hist_actual_pos_true, hist_expect_pos_true)
    assert_allclose(hist_actual_neg_true, hist_expect_neg_true)
    assert_allclose(hist_actual_pos_pred, hist_expect_pos_pred)
    assert_allclose(hist_actual_neg_pred, hist_expect_neg_pred)
