"""Data loading, splitting, normalization and mini-batching for Fashion-MNIST."""
import pandas as pd
import numpy as np


class Data:
    """Loader for the Fashion-MNIST CSV files.

    Attributes:
        path_train (pathlib.Path or str): training CSV with a 'label' column and 784 pixel columns.
        path_test (pathlib.Path or str): test CSV in the same format.
        rng (np.random.Generator): seeded generator used for all shuffling.
    """

    def __init__(self, path_train, path_test):
        """Store the CSV paths and create the seeded random generator.

        Args:
            path_train (pathlib.Path or str): path to the training CSV.
            path_test (pathlib.Path or str): path to the test CSV.
        """
        self.path_train = path_train
        self.path_test = path_test
        self.rng = np.random.default_rng(123)

    def shuffle(self, X, Y):
        """Shuffle X and Y with the same random permutation, using self.rng.

        Args:
            X (np.ndarray): inputs, shape (m, ...).
            Y (np.ndarray): labels, shape (m, ...).

        Returns:
            tuple: (X_shuffled, Y_shuffled), same shapes as the inputs.
        """
        perm = self.rng.permutation(len(X))  # (m,) random order of indices
        return X[perm], Y[perm]

    def compute_miu_std(self, X):
        """Compute one mean and one standard deviation over ALL elements of X.

        Args:
            X (np.ndarray): flattened images, shape (m, 784).

        Returns:
            tuple: (miu, std), both scalars.
        """
        m = len(X)

        miu = np.mean(X)  # scalar: mean over all m * 784 values
        std = np.sqrt(1/(X.size) * np.sum((X - miu)**2))  # scalar: divides by X.size = m * 784

        return miu, std

    def load_data(self, dev_size=5000):
        """Load Fashion-MNIST from the two CSV files, split, normalize and one-hot encode.

        The dev set is the last dev_size examples of the shuffled training CSV.
        The test CSV is used as the test set. Mean and std for normalization are
        computed on the training split only and then applied to dev and test.

        Args:
            dev_size (int): number of training examples held out as the dev set.

        Returns:
            tuple: six arrays (X_train, X_dev, X_test, Y_train, Y_dev, Y_test), where
                X_* have shape (m, 28, 28, 1) and dtype float32,
                Y_* are one-hot with shape (m, 10) and dtype float32.
        """
        train_df = pd.read_csv(self.path_train)
        test_df = pd.read_csv(self.path_test)

        labels_train, labels_test = train_df.pop('label').to_numpy(), test_df.pop('label').to_numpy()  # (60000,), (10000,)
        train_arr, test_arr = train_df.to_numpy() / 255., test_df.to_numpy() / 255.  # (60000, 784), (10000, 784) scaled to [0, 1]

        m_train = len(train_arr) - dev_size  # 60000 - dev_size
        m_dev = dev_size
        m_test = len(test_arr)  # 10000
        n = len(np.unique(labels_train, axis=0))  # number of classes, 10
        f = int(np.sqrt(len(train_arr[0])))  # image side length, 28

        X_test_shuffle, Y_test_shuffle = self.shuffle(test_arr, labels_test)  # (10000, 784), (10000,)
        X_temp_shuffle, Y_temp_shuffle = self.shuffle(train_arr, labels_train)  # (60000, 784), (60000,)

        X_train_shuffle, Y_train_shuffle = X_temp_shuffle[:m_train], Y_temp_shuffle[:m_train]  # (m_train, 784), (m_train,)
        X_dev_shuffle, Y_dev_shuffle = X_temp_shuffle[m_train:], Y_temp_shuffle[m_train:]  # (dev_size, 784), (dev_size,)

        assert len(X_dev_shuffle) == m_dev and len(X_train_shuffle) == m_train

        miu, std = self.compute_miu_std(X_train_shuffle)  # scalars, computed on train only
        assert np.isfinite(std)

        X_train_norm = ((X_train_shuffle - miu) / std).astype('float32')  # (m_train, 784)
        X_dev_norm = ((X_dev_shuffle - miu) / std).astype('float32')  # (dev_size, 784)
        X_test_norm = ((X_test_shuffle - miu) / std).astype('float32')  # (10000, 784)

        Y_train = np.eye(n)[Y_train_shuffle].astype('float32')  # (m_train, 10) one-hot
        Y_dev = np.eye(n)[Y_dev_shuffle].astype('float32')  # (dev_size, 10) one-hot
        Y_test = np.eye(n)[Y_test_shuffle].astype('float32')  # (10000, 10) one-hot

        # (m, 784) -> (m, 28, 28, 1): reshape each flat image into a 28x28 image with 1 channel
        X_train = np.reshape(X_train_norm, (-1, f, f, 1))
        X_dev = np.reshape(X_dev_norm, (-1, f, f, 1))
        X_test = np.reshape(X_test_norm, (-1, f, f, 1))

        return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

    def get_minibatch(self, X, Y, size, do_shuffle: bool):
        """Generator yielding mini-batches of (X, Y). One generator is one epoch.

        Args:
            X (np.ndarray): inputs, shape (m, ...).
            Y (np.ndarray): labels, shape (m, ...).
            size (int): batch size.
            do_shuffle (bool): if True, shuffle the whole dataset with self.rng when the generator is created.

        Yields:
            tuple: (X_batch, Y_batch) with shapes (size, ...) and (size, ...).
                The last batch can be smaller if m is not divisible by size.
        """
        m = len(X)
        if do_shuffle:
            perm = self.rng.permutation(m)  # (m,) new order of examples for this epoch
            X = X[perm]  # (m, ...)
            Y = Y[perm]  # (m, ...)
        n_batches = np.ceil(m / size).astype(int)  # number of batches, the last may be partial
        for i in range(n_batches):
            yield X[size * i  : size * (i + 1)], Y[size * i  : size * (i + 1)]  # (size, ...) each
