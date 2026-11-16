"""Random search over hyperparameters of a small CNN.

Each run samples one configuration, trains the model for a few epochs on a subset of the
training set, evaluates on the dev set and appends the result as one row to
outputs/tuning.csv. The test set is never used here.

Each finished run is written to disk right away, so a crash keeps the finished runs.
Call main() from a notebook, or run it from a terminal:
    python3 src/tuning.py
"""
import csv
import time
from pathlib import Path

import numpy as np

from data import load_data
from init import he, xavier
from layers import BatchNorm2d, Conv2d, Dense, Dropout, Flatten, ReLU, SoftmaxCrossEntropy
from model import Sequential
from optimizers import Adam

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'outputs'

FILTER_CHOICES = [8, 16, 32]


def sample_config(rng, n_conv_layers):
    """Draw one random hyperparameter configuration.

    Learning rate, L2 strength and 1 - Beta1 are drawn on a log scale, so every decade gets
    the same number of samples. keep_prob is drawn uniformly from [0.5, 1.0], where 1.0 means
    no dropout. The number of filters is drawn separately for each convolutional block.

    Args:
        rng (np.random.Generator): generator used for all random choices.
        n_conv_layers (int): number of convolutional blocks, each Conv2d -> BN -> ReLU.

    Returns:
        dict: configuration with plain Python types, so it can be saved as JSON. Keys:
            seed (int): seed for the model weights and the training order.
            lr (float): learning rate, log-uniform in [1e-4, 1e-1].
            lambda (float): L2 strength, log-uniform in [1e-5, 1e-2].
            Beta1 (float): Adam decay factor, drawn as 1 - 10**u with u in [-3, -1].
            n_filters (list of int): filters of each convolutional block, drawn from [8, 16, 32].
            keep_prob (float): dropout keep probability, uniform in [0.5, 1.0].
            augment (bool): whether training batches are randomly shifted.
    """
    return {
        'seed': int(rng.integers(0, 10**6)),
        'lr': float(10 ** rng.uniform(-4, -1)),
        'lambda': float(10 ** rng.uniform(-5, -2)),
        'Beta1': float(1 - 10 ** rng.uniform(-3, -1)),
        'n_filters': [int(rng.choice(FILTER_CHOICES)) for _ in range(n_conv_layers)],
        'keep_prob': float(rng.uniform(0.5, 1.0)),
        'augment': bool(rng.choice([True, False])),
    }


def build_model(config, rng):
    """Build the network described by a configuration.

    The network is: for each block Conv2d (no bias) -> BatchNorm2d -> ReLU, then Dropout if
    keep_prob < 1, then Flatten and Dense(10). The L2 strength is passed to the model.

    Args:
        config (dict): configuration from sample_config().
        rng (np.random.Generator): generator used to initialize the weights.

    Returns:
        Sequential: built model, ready for fit().
    """
    layers = []
    for n_C in config['n_filters']:
        layers += [Conv2d(n_C, f=3, s=1, p='same', bias=False, init=he),
                   BatchNorm2d(axes=(0, 1, 2)),
                   ReLU()]
    if config['keep_prob'] < 1.0:
        layers.append(Dropout(config['keep_prob']))
    layers += [Flatten(), Dense(10, bias=True, init=xavier)]

    model = Sequential(layers, input_shape=(28, 28, 1), rng=rng, l2_lambda=config['lambda'])
    model.build()
    return model


def run_experiment(config, X_train, Y_train, X_dev, Y_dev, max_epochs=3, batch_size=64):
    """Train one configuration and return its metrics.

    Args:
        config (dict): configuration from sample_config().
        X_train, Y_train (np.ndarray): training subset, one-hot labels.
        X_dev, Y_dev (np.ndarray): dev set, one-hot labels.
        max_epochs (int): number of epochs. Keep it small for a search.
        batch_size (int): mini-batch size.

    Returns:
        dict: the configuration plus metrics: n_params, best_dev_acc and best_dev_epoch (the
            highest dev accuracy and the epoch where it occurred), final_train_acc,
            final_dev_acc and minutes (wall-clock time of the run).
    """
    rng = np.random.default_rng(config['seed'])
    model = build_model(config, rng)
    n_params = model.summary()[0]['Total Parameters']

    start = time.perf_counter()
    output = model.fit(SoftmaxCrossEntropy(), X_train, Y_train, X_dev, Y_dev,
                       optimizer=Adam(Beta1=config['Beta1']), max_epochs=max_epochs,
                       batch_size=batch_size, lr=config['lr'], augment=config['augment'],
                       return_=['all'], print_epochs=0, seed=config['seed'])
    minutes = (time.perf_counter() - start) / 60

    (_, _), (acc_train, acc_dev), _ = output[0]
    best_epoch = max(acc_dev, key=acc_dev.get)
    last_epoch = max_epochs - 1

    return {
        'config': config,
        'n_params': int(n_params),
        'best_dev_acc': float(acc_dev[best_epoch]),
        'best_dev_epoch': int(best_epoch),
        'final_train_acc': float(acc_train[last_epoch]),
        'final_dev_acc': float(acc_dev[last_epoch]),
        'minutes': float(minutes),
    }


CSV_COLUMNS = ['run', 'seed', 'lr', 'lambda', 'Beta1', 'n_filters', 'keep_prob', 'augment',
               'n_params', 'best_dev_acc', 'best_dev_epoch', 'final_train_acc', 'final_dev_acc',
               'minutes']


def flatten_result(result):
    """Turn one result dict from run_experiment() into one flat row for the CSV file.

    The config is unpacked into its own columns. n_filters becomes a string such as '8-16',
    so it fits in one cell.

    Args:
        result (dict): result from run_experiment(), with the key 'run' added by main().

    Returns:
        dict: one value per name in CSV_COLUMNS.
    """
    config = result['config']
    row = {'run': result['run'], 'seed': config['seed'], 'lr': config['lr'],
           'lambda': config['lambda'], 'Beta1': config['Beta1'],
           'n_filters': '-'.join(str(n) for n in config['n_filters']),
           'keep_prob': config['keep_prob'], 'augment': config['augment']}
    for key in CSV_COLUMNS[len(row):]:
        row[key] = result[key]
    return row


def main(n_runs=10, n_train=10000, seed=0, max_epochs=3, n_conv_layers=2, data=None, out_path=None):
    """Run the random search, append every result to a CSV file and return them.

    Can be called from a notebook: pass the arrays you already loaded, so the CSV files are
    not read again. Nothing is loaded at import time. The header is written once, when the
    file is created or empty; later runs append rows, so one file can collect several calls.

    Args:
        n_runs (int): number of random configurations to try.
        n_train (int): size of the training subset used by every run.
        seed (int): seed of the generator that draws the configurations.
        max_epochs (int): number of epochs per run.
        data (tuple or None): the six arrays returned by load_data(), in the same order:
            (X_train, X_dev, X_test, Y_train, Y_dev, Y_test). X_test is not used. If None,
            the data is loaded from the CSV files in data/.
        out_path (str, pathlib.Path or None): CSV file the results are appended to. If None,
            outputs/tuning.csv is used.

    Returns:
        list of dict: one result per run, as returned by run_experiment(), plus the key 'run'.
    """
    if data is None:
        data = load_data(DATA_DIR / 'fashion-mnist_train.csv',
                         DATA_DIR / 'fashion-mnist_test.csv',
                         dtype='float64')
    X_train, X_dev, _, Y_train, Y_dev, _ = data
    X_train, Y_train = X_train[:n_train], Y_train[:n_train]

    rng = np.random.default_rng(seed)
    out_path = OUT_DIR / 'tuning.csv' if out_path is None else Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    results = []
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for run in range(n_runs):
            config = sample_config(rng, n_conv_layers)
            result = run_experiment(config, X_train, Y_train, X_dev, Y_dev, max_epochs=max_epochs)
            result['run'] = run
            results.append(result)
            writer.writerow(flatten_result(result))
            f.flush()
            print(f"run {run}: best dev {result['best_dev_acc']:.4f} "
                  f"(epoch {result['best_dev_epoch']}), params {result['n_params']}, "
                  f"{result['minutes']:.1f} min")
    return results


if __name__ == '__main__':
    main()
