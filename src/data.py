import pandas as pd
import numpy as np


def shuffle(X, Y, rng):
    perm = rng.permutation(len(X))
    return X[perm], Y[perm]

def compute_miu_std(X):
    m = len(X)

    miu = np.mean(X)
    std = np.sqrt(1/(m * 784) * np.sum((X - miu)**2))

    return miu, std

def get_batch(X, Y, size, rng, do_shuffle:bool):
    m = len(X)
    if do_shuffle:
        perm = rng.permutation(m)
        X = X[perm]
        Y = Y[perm]
    n_batches = np.ceil(m / size).astype(int)
    for i in range(n_batches):
        yield X[size * i  : size * (i + 1)], Y[size * i  : size * (i + 1)]



def load_data(path_train, path_test, batch_size, dev_size=5000, print_distribution=False, return_batch=True):
    """
        Loads data and preprocesses it

        args:
            -path_train: pathlib path
            -path_test: pathlib path

        returns:
            -train: generator object, tuple (X_train Y_train)
            -dev: generator object, tuple (X_dev, Y_dev)
            -test generator object, typle (X_test, Y_dev)
    """
    rng = np.random.default_rng(123)

    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    

    labels_train, labels_test = train_df.pop('label').to_numpy(), test_df.pop('label').to_numpy()
    train_arr, test_arr = train_df.to_numpy() / 255., test_df.to_numpy() / 255.

    m_train = len(train_arr) - dev_size
    m_dev = dev_size
    m_test = len(test_arr)
    n = len(np.unique(labels_train, axis=0))
    f = int(np.sqrt(len(train_arr[0])))
    if print_distribution:
        print(f'Number of train examples: {m_train - dev_size}, \nNumber of dev examples: {m_dev}, \nNumber of test examples: {m_test}, \nSize of images: ({f}px, {f}px)')

    X_test_shuffle, Y_test_shuffle = shuffle(test_arr, labels_test, rng)
    X_temp_shuffle, Y_temp_shuffle = shuffle(train_arr, labels_train, rng)

    X_train_shuffle, Y_train_shuffle = X_temp_shuffle[:m_train], Y_temp_shuffle[:m_train]
    X_dev_shuffle, Y_dev_shuffle = X_temp_shuffle[m_train:], Y_temp_shuffle[m_train:]

    assert len(X_dev_shuffle) == m_dev and len(X_train_shuffle) == m_train


    miu, std = compute_miu_std(X_train_shuffle)
    assert np.isfinite(std)

    X_train_norm = ((X_train_shuffle - miu) / std).astype('float32')
    X_dev_norm = ((X_dev_shuffle - miu) / std).astype('float32')
    X_test_norm = ((X_test_shuffle - miu) / std).astype('float32')


    Y_train = np.eye(n)[Y_train_shuffle].astype('float32')
    Y_dev = np.eye(n)[Y_dev_shuffle].astype('float32')
    Y_test = np.eye(n)[Y_test_shuffle].astype('float32')

    X_train = np.reshape(np.expand_dims(X_train_norm, axis=0), (-1, f, f, 1))
    X_dev = np.reshape(np.expand_dims(X_dev_norm, axis=0), (-1, f, f, 1))
    X_test = np.reshape(np.expand_dims(X_test_norm, axis=0), (-1, f, f, 1))
    
    if return_batch:
        return get_batch(X_train, Y_train, batch_size, rng, do_shuffle=True), get_batch(X_dev, Y_dev, batch_size, rng, do_shuffle=False), get_batch(X_test, Y_test, batch_size, rng, do_shuffle=False)
    else:
        return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)


def check_norm(data):
    """
        args:
            -data: iterable
        returns:
            avg, std
    """
    X_train, _ = data
    return X_train.mean(), X_train.std()

def plot_distribution(train, dev, test, ax):
    _, Y_train = train
    _, Y_dev = dev
    _, Y_test = test
    
    ax[0].hist(np.argmax(Y_train, axis=1), edgecolor='black', linewidth=0.5)
    ax[0].set_title('Train')
    ax[1].hist(np.argmax(Y_dev, axis=1), edgecolor='black', linewidth=0.5, color='orange')
    ax[1].set_title('Dev')
    ax[2].hist(np.argmax(Y_test, axis=1), edgecolor='black', linewidth=0.5, color='green')
    ax[2].set_title('Test')
    for ax_i in ax:
        ax_i.set_xticks(np.arange(0, 10, 1))

def check_minibatch_len(data_train, data_dev, data_test, m_train, m_dev, m_test):
    m_minibatch_train = 0
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