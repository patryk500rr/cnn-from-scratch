"""Weight initialization schemes. Each function rescales weights that were sampled from N(0, 1)."""
import numpy as np


def he(W):
    """Scale standard normal weights for ReLU: Var(W) = 2 / n_in.

    The factor 2 compensates for ReLU, which passes only the positive half of the signal,
    so the variance of the activations would otherwise shrink from layer to layer.

    The fan-in n_in is the product of all dimensions except the last one: n_in for Dense weights
    (n_in, n_out), and f * f * n_C_prev for conv weights (f, f, n_C_prev, n_C).

    Args:
        W (np.ndarray): weights sampled from N(0, 1), e.g. rng.standard_normal(shape).

    Returns:
        np.ndarray: W * sqrt(2 / n_in), same shape as W.
    """
    n = np.prod(W.shape[:-1])

    return W * np.sqrt(2 / n)


def xavier(W):
    """Scale standard normal weights for tanh or sigmoid: Var(W) = 1 / n_in.

    Keeps the variance of the activations roughly constant for activations that are
    close to linear around zero. Uses only the fan-in; the variant 2 / (n_in + n_out)
    is not implemented here.

    The fan-in n_in is the product of all dimensions except the last one, as in he().

    Args:
        W (np.ndarray): weights sampled from N(0, 1), e.g. rng.standard_normal(shape).

    Returns:
        np.ndarray: W * sqrt(1 / n_in), same shape as W.
    """
    n = np.prod(W.shape[:-1])

    return W * np.sqrt(1 / n)


def scaled_normal(W, scale):
    """Multiply standard normal weights by a fixed constant, giving standard deviation `scale`.

    Use it for fixed-scale experiments, e.g. N(0, 0.01^2) by passing scale=0.01 to weights
    sampled from N(0, 1). The scale does not depend on the layer size.

    Args:
        W (np.ndarray): weights sampled from N(0, 1), e.g. rng.standard_normal(shape).
        scale (float): standard deviation of the result.

    Returns:
        np.ndarray: W * scale, same shape as W.
    """
    return W * scale
