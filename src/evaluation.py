"""Evaluation metrics for the classifier."""
import numpy as np


def accuracy(Y_hat, Y):
    """Fraction of examples whose predicted class equals the true class.

    Args:
        Y_hat (np.ndarray): predicted class indices, shape (m,), e.g. Z.argmax(axis=1).
        Y (np.ndarray): true class indices, shape (m,), e.g. Y_one_hot.argmax(axis=1).

    Returns:
        float: accuracy in [0, 1].

    """
    return np.mean(Y_hat == Y)
