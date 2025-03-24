import numpy as np


def expected_calibration_error(correct_probs: list, incorrect_probs: list, n_bins: int = 10) -> float:
    """
    More efficient ECE computation using vectorized bin counting.
    """
    y_true = np.array([1] * len(correct_probs) + [0] * len(incorrect_probs), dtype=float)
    y_prob = np.array(correct_probs + incorrect_probs, dtype=float)
    n_samples = y_prob.size

    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    sum_of_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    sum_of_positives = np.bincount(bin_indices, weights=y_true, minlength=n_bins)

    nonzero = bin_counts > 0
    fraction_of_positives = sum_of_positives[nonzero] / bin_counts[nonzero]
    mean_predicted_value = sum_of_probs[nonzero] / bin_counts[nonzero]

    bin_weights = bin_counts[nonzero] / n_samples
    ece = np.sum(bin_weights * np.abs(fraction_of_positives - mean_predicted_value))

    return ece
