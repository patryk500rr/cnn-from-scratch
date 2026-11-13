"""Evaluation metrics for the classifier."""
import numpy as np
from data import get_minibatch


def accuracy(Y_hat, Y):
    """Fraction of examples whose predicted class equals the true class.

    Args:
        Y_hat (np.ndarray): predicted class indices, shape (m,), e.g. Z.argmax(axis=1).
        Y (np.ndarray): true class indices, shape (m,), e.g. Y_one_hot.argmax(axis=1).

    Returns:
        float: accuracy in [0, 1].

    """
    return np.mean(Y_hat == Y)

def evaluate(model, loss, X, Y, batch_size=None):
    """Compute loss and accuracy on a dataset without updating the model.

    Runs forward only: no backward and no parameter changes. The current code has
    two branches that return different types, see the Returns section.

    Args:
        model (Sequential): network with a forward method.
        loss (SoftmaxCrossEntropy): loss layer, used as loss.forward(Z, Y).
        X (np.ndarray): inputs, shape (m, 28, 28, 1).
        Y (np.ndarray): one-hot labels, shape (m, 10).
        batch_size (int or None): if given, the data is processed in mini-batches of this size.
            If None, the whole dataset goes through one forward pass.

    Returns:
        if batch_size is given: tuple (J_list, acc_list), one loss and one accuracy per batch.
        if batch_size is None: tuple (J, acc), a single loss and a single accuracy.
            The intended result is the mean over all examples, not per batch.

    Note:
        Because the forward pass overwrites the layers' cache, do not call evaluate between
        the forward and backward pass of a training batch.
    """
    if batch_size is not None:
        batch_loader = get_minibatch(X, Y, batch_size)
        J_list = []
        acc_list = []
        while True:
            try:
                X_minibatch, Y_minibatch = next(batch_loader)
            except StopIteration:
                break

            Z = model.forward(X_minibatch)
            J = loss.forward(Z, Y_minibatch)
            J_list.append(J)

            Y_hat = np.argmax(Z, axis=1)

            acc = accuracy(Y_hat, Y_minibatch.argmax(axis=1))
            acc_list.append(acc)

        return J_list, acc_list
    
    else:
        Z = model.forward(X)
        J = loss.forward(Z, Y)

        Y_hat = np.argmax(Z, axis=1)
        acc = accuracy(Y_hat, Y.argmax(axis=1))

        return J, acc
