"""Model definitions and training runs for the Fashion-MNIST experiments."""
import numpy as np
from layers import Dense, ReLU, Flatten, SoftmaxCrossEntropy, conv_naive
from model import Sequential
from optimizers import step_SGD
from train import fit


def run_MLP(X_train, Y_train, X_dev, Y_dev, rng, lr, batch_size, max_epochs):
    """Train the baseline MLP: Flatten -> Dense(128) -> ReLU -> Dense(10) -> softmax cross-entropy.

    Args:
        X_train (np.ndarray): training inputs, shape (m_train, 28, 28, 1).
        Y_train (np.ndarray): one-hot training labels, shape (m_train, 10).
        X_dev (np.ndarray): dev inputs, shape (m_dev, 28, 28, 1).
        Y_dev (np.ndarray): one-hot dev labels, shape (m_dev, 10).
        rng (np.random.Generator): generator for the weight initialization and the shuffling.
        lr (float): learning rate.
        batch_size (int): mini-batch size.
        max_epochs (int): number of epochs.

    Returns:
        Sequential: the trained model.
    """

    model = Sequential([
        Flatten(),                                                              # (m, 28, 28, 1) -> (m, 784)
        Dense(rng.standard_normal((784, 128)) * np.sqrt(2 / 784), np.zeros(128)),  # (m, 784) -> (m, 128)
        ReLU(),                                                                 # (m, 128)
        Dense(rng.standard_normal((128, 10)) * np.sqrt(2 / 128), np.zeros(10)),    # (m, 128) -> (m, 10)
    ])

    for J_train, J_dev, acc_train, acc_dev in fit(
        model, SoftmaxCrossEntropy(), step_SGD, 
        X_train, Y_train, X_dev, Y_dev, 
        max_epochs=max_epochs, batch_size=batch_size, lr=lr, rng=rng): pass
    
    return model

def run_conv_naive(X_train, Y_train, X_dev, Y_dev, rng, lr, batch_size, max_epochs, n_filters=8, f=3):
    """Train a small CNN: Conv -> ReLU -> Flatten -> Dense -> softmax cross-entropy.

    The convolution uses stride 1 and padding (f - 1) // 2, so the spatial size stays 28x28.
    Requires a Conv layer with the interface Conv(W, b, stride, pad), forward(A), backward(dA),
    and a grads dict with 'dW' and 'db'.

    Args:
        X_train (np.ndarray): training inputs, shape (m_train, 28, 28, 1).
        Y_train (np.ndarray): one-hot training labels, shape (m_train, 10).
        X_dev (np.ndarray): dev inputs, shape (m_dev, 28, 28, 1).
        Y_dev (np.ndarray): one-hot dev labels, shape (m_dev, 10).
        rng (np.random.Generator): generator for weight initialization and shuffling.
        lr (float): learning rate.
        batch_size (int): mini-batch size.
        max_epochs (int): number of epochs.
        n_filters (int): number of convolution filters (channels after Conv).
        f (int): filter side length.

    Returns:
        Sequential: the trained model.
    """
    pad = (f - 1) // 2
    flat_size = 28 * 28 * n_filters

    model = Sequential([
        conv_naive(rng.standard_normal((f, f, 1, n_filters)) * np.sqrt(2 / (f * f)), np.zeros(n_filters), s=1, p=pad),  # (m, 28, 28, 1) -> (m, 28, 28, n_filters)
        ReLU(),                                                                                                          # (m, 28, 28, n_filters)
        Flatten(),                                                                                                       # (m, 28, 28, n_filters) -> (m, flat_size)
        Dense(rng.standard_normal((flat_size, 10)) * np.sqrt(2 / flat_size), np.zeros(10)),                              # (m, flat_size) -> (m, 10)
    ])

    for J_train, J_dev, acc_train, acc_dev in fit(
        model, SoftmaxCrossEntropy(), step_SGD,
        X_train, Y_train, X_dev, Y_dev,
        max_epochs=max_epochs, batch_size=batch_size, lr=lr, rng=rng): pass

    return model

def predict_mlp(X):
    """Return predicted class indices for inputs X. NOT WORKING YET.

    Intended use: model = run_MLP(...), then predict_mlp(model, X). The current body
    calls Sequential(X), which is wrong, and argmax without an axis, so it returns one number.

    Args:
        X (np.ndarray): inputs, shape (m, 28, 28, 1).

    Returns:
        np.ndarray: predicted class indices, shape (m,), once implemented.
    """
    Z = Sequential(X)
    return np.argmax(Z)