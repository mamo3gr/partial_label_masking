import numpy as np

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

    loss = PartialLabelMaskingLoss(positive_ratio=positive_ratio)
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
