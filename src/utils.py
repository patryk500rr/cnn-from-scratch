"""Helper functions for checking the data pipeline and plotting class distributions."""
import numpy as np


def check_norm(X):
    """Print the mean and standard deviation of an array of inputs.

    After normalization on the training set, they should be close to 0 and 1.

    Args:
        X (np.ndarray): training inputs, e.g. X_train from load_data.

    Returns:
        None. The values are printed, rounded to 5 decimal places.
    """
    print(f'mean: {round(X.mean(), 5)}, std: {round(X.std(), 5)}')


def plot_distribution(Y_train, Y_dev, Y_test, ax):
    """Plot histograms of the class labels for the train, dev and test splits.

    Args:
        Y_train (np.ndarray): one-hot labels, shape (m_train, 10).
        Y_dev (np.ndarray): one-hot labels, shape (dev_size, 10).
        Y_test (np.ndarray): one-hot labels, shape (m_test, 10).
        ax (array-like): three matplotlib axes, one per split.
    """
    ax[0].hist(np.argmax(Y_train, axis=1), edgecolor='black', linewidth=0.5)  # labels in 0..9
    ax[0].set_title('Train')
    ax[1].hist(np.argmax(Y_dev, axis=1), edgecolor='black', linewidth=0.5, color='orange')
    ax[1].set_title('Dev')
    ax[2].hist(np.argmax(Y_test, axis=1), edgecolor='black', linewidth=0.5, color='green')
    ax[2].set_title('Test')
    for ax_i in ax:
        ax_i.set_xticks(np.arange(0, 10, 1))


def check_minibatch_len(mini_batch_loader, m):
    """Check that the mini-batches from one generator add up to the expected number of examples.

    Consumes the generator completely, so pass a freshly created one.
    Asserts that X and Y in each batch have the same length and that no batch is empty.
    Prints 'All tests passed' when all checks succeed.

    Args:
        mini_batch_loader (generator): yields (X_batch, Y_batch) tuples.
        m (int): expected total number of examples across all batches.

    Raises:
        AssertionError: if a batch is malformed or the total does not match m.
    """
    m_minibatch = 0

    while True:
        try:
            X, Y = next(mini_batch_loader)
            assert len(X) == len(Y)
            assert len(X) != 0

            m_minibatch += len(X)
        except StopIteration:
            break

    assert m_minibatch == m

    print('All tests passed')

def ewa(series, beta, bias_correction=False):
    """Compute the exponentially weighted average (EWA) of a one-dimensional series.

    Each output value is a weighted average of the current and all previous inputs,
    with weights that decay geometrically. The state v starts at 0 and is updated as
    v = beta * v + (1 - beta) * value. The bias correction only changes the returned
    value, not the state v.

    Args:
        series (array-like): one-dimensional sequence of numbers in time order, e.g. the
            mini-batch losses from step 9. Converted to a float array.
        beta (float): decay factor in (0, 1). Larger beta means a longer memory, roughly
            the last 1 / (1 - beta) values.
        bias_correction (bool): if True, each returned value is divided by (1 - beta**t),
            where t starts at 1. This removes the underestimate caused by v starting at 0.
            If False, the raw v is returned.

    Returns:
        np.ndarray: EWA values, same length as series. Element t (counted from 1) is the
            average after t inputs.

    Raises:
        AssertionError: if beta is not in (0, 1), if series is empty, or if series is not
            one-dimensional.
    """
    series = np.asarray(series, dtype='float')

    assert beta > 0 and beta < 1
    assert len(series) > 0
    assert series.ndim == 1

    v = 0
    t = 1
    series_out = []

    for value in series:
        v = beta * v + (1 - beta) * value
        if bias_correction:
            v_hat = v / (1 - beta**t)
            series_out.append(v_hat)
        else:
            series_out.append(v)
        
        t += 1

    return np.array(series_out)
        


