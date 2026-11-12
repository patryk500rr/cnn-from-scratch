import pandas as pd
import numpy as np


def shuffle(X, Y, rng):
    """Shuffle X and Y with the same random permutation.

    Args:
        X (np.ndarray): inputs, shape (m, ...).
        Y (np.ndarray): labels, shape (m, ...).
        rng (np.random.Generator): random generator used for the permutation.

    Returns:
        tuple: (X_shuffled, Y_shuffled), same shapes as the inputs.
    """
    perm = rng.permutation(len(X))  # (m,) random order of indices
    return X[perm], Y[perm]


def compute_miu_std(X):
    """Compute a single mean and a single standard deviation over ALL elements.

    Args:
        X (np.ndarray): flattened images, shape (m, 784).

    Returns:
        tuple: (miu, std), both scalars.
    """
    m = len(X)

    miu = np.mean(X)  # scalar: mean over all m * 784 values
    std = np.sqrt(1/(m * 784) * np.sum((X - miu)**2))  # scalar: divides by m * 784 = number of values

    return miu, std


def get_batch(X, Y, size, rng, do_shuffle: bool):
    """Generator yielding mini-batches of (X, Y). One generator is one epoch.

    Args:
        X (np.ndarray): inputs, shape (m, ...).
        Y (np.ndarray): labels, shape (m, ...).
        size (int): batch size.
        rng (np.random.Generator): random generator, used only if do_shuffle is True.
        do_shuffle (bool): if True, shuffle the whole dataset when the generator is created.

    Yields:
        tuple: (X_batch, Y_batch) with shapes (size, ...) and (size, ...).
               The last batch can be smaller if m is not divisible by size.
    """
    m = len(X)
    if do_shuffle:
        perm = rng.permutation(m)  # (m,) new order of examples for this epoch
        X = X[perm]  # (m, ...)
        Y = Y[perm]  # (m, ...)
    n_batches = np.ceil(m / size).astype(int)  # number of batches, the last may be partial
    for i in range(n_batches):
        yield X[size * i  : size * (i + 1)], Y[size * i  : size * (i + 1)]  # (size, ...) each


def load_data(path_train, path_test, batch_size, dev_size=5000, print_distribution=False, return_batch=True):
    """Load Fashion-MNIST from CSV, split, normalize and one-hot encode the labels.

    The dev set is taken from the training CSV. The test CSV is used as the test set.
    Normalization statistics (mean and std) are computed on the training split only.

    Args:
        path_train (pathlib.Path): path to the training CSV (label column + 784 pixel columns).
        path_test (pathlib.Path): path to the test CSV, same format.
        batch_size (int): batch size used by the generators.
        dev_size (int): number of training examples held out as the dev set.
        print_distribution (bool): if True, print the number of examples in each split.
        return_batch (bool): if True, return generators of mini-batches.
                             if False, return the full arrays.

    Returns:
        if return_batch is True:
            tuple of three generators (train, dev, test). The train generator shuffles
            on creation. Each yields (X_batch, Y_batch) with X of shape
            (batch_size, 28, 28, 1) and Y of shape (batch_size, 10).
        if return_batch is False:
            tuple of three (X, Y) pairs (train, dev, test) with full arrays:
            X_train (m_train, 28, 28, 1), X_dev (dev_size, 28, 28, 1), X_test (m_test, 28, 28, 1),
            Y_* one-hot of shape (m, 10), float32.
    """
    rng = np.random.default_rng(123)

    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)



    labels_train, labels_test = train_df.pop('label').to_numpy(), test_df.pop('label').to_numpy()  # (60000,), (10000,)
    train_arr, test_arr = train_df.to_numpy() / 255., test_df.to_numpy() / 255.  # (60000, 784), (10000, 784) scaled to [0, 1]

    m_train = len(train_arr) - dev_size  # 60000 - dev_size
    m_dev = dev_size
    m_test = len(test_arr)  # 10000
    n = len(np.unique(labels_train, axis=0))  # number of classes, 10
    f = int(np.sqrt(len(train_arr[0])))  # image side length, 28
    if print_distribution:
        print(f'Number of train examples: {m_train},\nNumber of dev examples: {m_dev}, \nNumber of test examples: {m_test}, \nSize of images: ({f}px, {f}px)')

    X_test_shuffle, Y_test_shuffle = shuffle(test_arr, labels_test, rng)  # (10000, 784), (10000,)
    X_temp_shuffle, Y_temp_shuffle = shuffle(train_arr, labels_train, rng)  # (60000, 784), (60000,)

    X_train_shuffle, Y_train_shuffle = X_temp_shuffle[:m_train], Y_temp_shuffle[:m_train]  # (m_train, 784), (m_train,)
    X_dev_shuffle, Y_dev_shuffle = X_temp_shuffle[m_train:], Y_temp_shuffle[m_train:]  # (dev_size, 784), (dev_size,)

    assert len(X_dev_shuffle) == m_dev and len(X_train_shuffle) == m_train


    miu, std = compute_miu_std(X_train_shuffle)  # scalars, computed on train only
    assert np.isfinite(std)

    X_train_norm = ((X_train_shuffle - miu) / std).astype('float32')  # (m_train, 784)
    X_dev_norm = ((X_dev_shuffle - miu) / std).astype('float32')  # (dev_size, 784)
    X_test_norm = ((X_test_shuffle - miu) / std).astype('float32')  # (10000, 784)


    Y_train = np.eye(n)[Y_train_shuffle].astype('float32')  # (m_train, 10) one-hot
    Y_dev = np.eye(n)[Y_dev_shuffle].astype('float32')  # (dev_size, 10) one-hot
    Y_test = np.eye(n)[Y_test_shuffle].astype('float32')  # (10000, 10) one-hot

    # (m, 784) -> (1, m, 784) -> (m, 28, 28, 1): add a batch axis, then reshape to images with 1 channel
    X_train = np.reshape(np.expand_dims(X_train_norm, axis=0), (-1, f, f, 1))
    X_dev = np.reshape(np.expand_dims(X_dev_norm, axis=0), (-1, f, f, 1))
    X_test = np.reshape(np.expand_dims(X_test_norm, axis=0), (-1, f, f, 1))

    if return_batch:
        return get_batch(X_train, Y_train, batch_size, rng, do_shuffle=True), get_batch(X_dev, Y_dev, batch_size, rng, do_shuffle=False), get_batch(X_test, Y_test, batch_size, rng, do_shuffle=False)
    else:
        return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)


def check_norm(data):
    """Return the mean and standard deviation of the training inputs.

    After normalization they should be close to 0 and 1.

    Args:
        data (tuple): (X_train, Y_train) as returned by load_data(..., return_batch=False).

    Returns:
        tuple: (mean, std), both scalars.
    """
    X_train, _ = data
    return X_train.mean(), X_train.std()


def plot_distribution(train, dev, test, ax):
    """Plot histograms of class labels for the train, dev and test splits.

    Args:
        train (tuple): (X, Y) with one-hot Y of shape (m, 10).
        dev (tuple): (X, Y) with one-hot Y of shape (dev_size, 10).
        test (tuple): (X, Y) with one-hot Y of shape (m_test, 10).
        ax (array-like): three matplotlib axes, one per split.
    """
    _, Y_train = train
    _, Y_dev = dev
    _, Y_test = test

    ax[0].hist(np.argmax(Y_train, axis=1), edgecolor='black', linewidth=0.5)  # labels in 0..9
    ax[0].set_title('Train')
    ax[1].hist(np.argmax(Y_dev, axis=1), edgecolor='black', linewidth=0.5, color='orange')
    ax[1].set_title('Dev')
    ax[2].hist(np.argmax(Y_test, axis=1), edgecolor='black', linewidth=0.5, color='green')
    ax[2].set_title('Test')
    for ax_i in ax:
        ax_i.set_xticks(np.arange(0, 10, 1))


def check_minibatch_len(data_train, data_dev, data_test, m_train, m_dev, m_test):
    """Check that the mini-batches of each split add up to the expected number of examples.

    Consumes the generators completely, so pass freshly created ones.
    Asserts that X and Y in each batch have the same length and that no batch is empty.

    Args:
        data_train, data_dev, data_test: generators yielding (X_batch, Y_batch).
        m_train (int): expected number of train examples.
        m_dev (int): expected number of dev examples.
        m_test (int): expected number of test examples.

    Raises:
        AssertionError: if a batch is malformed or a total does not match.
    """
    m_minibatch_train = 0  # running total of train examples seen
    m_minibatch_dev = 0
    m_minibatch_test = 0

    while True:
        try:
            X_train, Y_train = next(data_train)
            assert len(X_train) == len(Y_train)
            assert len(X_train) != 0

            m_minibatch_train += len(X_train)
        except StopIteration:
            break

    while True:
        try:
            X_dev, Y_dev = next(data_dev)
            assert len(X_dev) == len(Y_dev)
            assert len(X_dev) != 0

            m_minibatch_dev += len(X_dev)
        except StopIteration:
            break

    while True:
        try:
            X_test, Y_test = next(data_test)
            assert len(X_test) == len(Y_test)
            assert len(Y_test) != 0

            m_minibatch_test += len(X_test)
        except StopIteration:
            break

    assert m_minibatch_train == m_train
    assert m_minibatch_dev == m_dev
    assert m_minibatch_test == m_test

    print('All test passed')
