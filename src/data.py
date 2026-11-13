"""Loading, splitting, normalizing and mini-batching of the Fashion-MNIST data.

Only functions that really need randomness take an rng argument. load_data
takes a seed and creates the generator itself, because it is called once.
"""
import numpy as np
import pandas as pd


def shuffle(X, Y, rng):
    """Shuffle X and Y with the same random permutation.

    Args:
        X (np.ndarray): inputs, shape (m, ...).
        Y (np.ndarray): labels, shape (m, ...).
        rng (np.random.Generator): generator used to draw the permutation.

    Returns:
        tuple: (X_shuffled, Y_shuffled), same shapes as the inputs.
    """
    perm = rng.permutation(len(X))  # (m,) random order of indices
    return X[perm], Y[perm]


def compute_miu_std(X):
    """Compute one mean and one standard deviation over ALL elements of X.

    Args:
        X (np.ndarray): inputs of any shape, e.g. flattened images (m, 784).

    Returns:
        tuple: (miu, std), both numpy scalars.
    """
    miu = np.mean(X)  # scalar: mean over all elements
    std = np.sqrt(np.mean((X - miu) ** 2))  # scalar: same as sqrt(sum / X.size)
    return miu, std


def load_data(path_train, path_test, dev_size=5000, seed=123):
    """Load Fashion-MNIST from two CSV files, split, normalize and one-hot encode.

    The training CSV is shuffled and its last dev_size examples become the dev set.
    The test CSV is used unchanged as the test set. Mean and std for normalization
    are computed on the training split only and applied to dev and test.

    Args:
        path_train (pathlib.Path or str): training CSV with a 'label' column and 784 pixel columns.
        path_test (pathlib.Path or str): test CSV in the same format.
        dev_size (int): number of training examples held out as the dev set.
        seed (int): seed for the random generator used to shuffle the training data.

    Returns:
        tuple: six arrays (X_train, X_dev, X_test, Y_train, Y_dev, Y_test), where
            X_* have shape (m, 28, 28, 1) and dtype float32,
            Y_* are one-hot with shape (m, 10) and dtype float32.
    """
    rng = np.random.default_rng(seed)

    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    labels_train = train_df.pop('label').to_numpy()  # (60000,)
    labels_test = test_df.pop('label').to_numpy()  # (10000,)
    train_arr = train_df.to_numpy() / 255.  # (60000, 784) scaled to [0, 1]
    test_arr = test_df.to_numpy() / 255.  # (10000, 784) scaled to [0, 1]

    m_train = len(train_arr) - dev_size  # 60000 - dev_size
    n = len(np.unique(labels_train))  # number of classes, 10
    f = int(np.sqrt(train_arr.shape[1]))  # image side length, 28

    X_shuffled, Y_shuffled = shuffle(train_arr, labels_train, rng)  # (60000, 784), (60000,)
    X_train_s, Y_train_s = X_shuffled[:m_train], Y_shuffled[:m_train]  # (m_train, 784), (m_train,)
    X_dev_s, Y_dev_s = X_shuffled[m_train:], Y_shuffled[m_train:]  # (dev_size, 784), (dev_size,)

    miu, std = compute_miu_std(X_train_s)  # scalars, computed on train only
    assert np.isfinite(std) and std > 0

    X_train = ((X_train_s - miu) / std).astype('float32').reshape(-1, f, f, 1)  # (m_train, 28, 28, 1)
    X_dev = ((X_dev_s - miu) / std).astype('float32').reshape(-1, f, f, 1)  # (dev_size, 28, 28, 1)
    X_test = ((test_arr - miu) / std).astype('float32').reshape(-1, f, f, 1)  # (10000, 28, 28, 1)

    Y_train = np.eye(n, dtype='float32')[Y_train_s]  # (m_train, 10) one-hot
    Y_dev = np.eye(n, dtype='float32')[Y_dev_s]  # (dev_size, 10) one-hot
    Y_test = np.eye(n, dtype='float32')[labels_test]  # (10000, 10) one-hot

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def get_minibatch(X, Y, size, rng=None):
    """Generator yielding mini-batches of (X, Y). One generator call is one epoch.

    Args:
        X (np.ndarray): inputs, shape (m, ...).
        Y (np.ndarray): labels, shape (m, ...).
        size (int): batch size.
        rng (np.random.Generator or None): if given, the examples are shuffled once when
            the generator is created, so each epoch gets a new order. Pass the same
            generator object every epoch, not a seed, otherwise each epoch gets the same order.
            If None, the data is not shuffled (use this for dev and test).

    Yields:
        tuple: (X_batch, Y_batch) with shapes (size, ...) and (size, ...).
            The last batch can be smaller if m is not divisible by size.
    """
    if rng is not None:
        X, Y = shuffle(X, Y, rng)
    for start in range(0, len(X), size):
        yield X[start:start + size], Y[start:start + size]
