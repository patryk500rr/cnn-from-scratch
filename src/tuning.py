"""Random search over hyperparameters of a network that you describe with a list of layers.

You pass the layers in the order you want them, an optimizer class and (optionally) a learning
rate schedule. The module works out which hyperparameters make sense to search for exactly that
network (list_hyperparameters), samples one configuration per run, trains for a few epochs on a
subset of the training set, evaluates on the dev set and appends the result as one row to a CSV
file. The test set is never used here.

Each finished run is written to disk right away, so a crash keeps the finished runs.

Example (notebook):
    layers = [Conv2d(16, f=3, s=1, p='same', bias=False, init=he),
              BatchNorm2d(axes=(0, 1, 2)), ReLU(), Dropout(0.8),
              Flatten(), Dense(10, init=xavier)]
    list_hyperparameters(layers, Adam)                 # print what can be searched
    results = main(layers, Adam, n_runs=20, data=(X_train, X_dev, X_test, Y_train, Y_dev, Y_test))

Terminal (runs the default layers below):
    python3 src/tuning.py
"""
import copy
import ast
import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np

from data import load_data
from init import he, xavier
from layers import (BatchNorm2d, Conv2d, Dense, Dropout, Flatten, GlobalAvgPool, MaxPool2d, ReLU,
                    ResidualBlock, SoftmaxCrossEntropy)
from model import Sequential
from optimizers import Adam

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'outputs'

DTYPE = 'float32'  # data and weights; about 2x less memory traffic than float64

# Names that are not layer attributes and not optimizer arguments.
RESERVED = ('seed', 'lr', 'lambda', 'batch_size', 'augment', 'gamma', 'keepprob')

# Search ranges used by default_space(). See describe_spec() for how a spec is read.
FILTER_CHOICES = [8, 16, 32]
KERNEL_CHOICES = [3, 5]
UNIT_CHOICES = [32, 64, 128, 256]
BATCH_CHOICES = [32, 64, 128]


# ----------------------------------------------------------------------------------------------
# Search space
# ----------------------------------------------------------------------------------------------
# A spec is how one hyperparameter is sampled. It is one of:
#   ('log', low, high)            log-uniform in [low, high]
#   ('one_minus_log', low, high)  1 - x, where x is log-uniform in [low, high] (for betas)
#   ('uniform', low, high)        uniform in [low, high]
#   ('choice', [a, b, ...])       one of the listed values
#   any other value               fixed: the hyperparameter is not searched, it always has this value
# A space is a dict {name: spec}. Names are the keys of the CSV columns.

def _walk_layers(layers, prefix=''):
    """Yield (path, layer) for every layer, going inside ResidualBlocks.

    Args:
        layers (list): layer objects.
        prefix (str): path of the parent block, e.g. '2.' for the layers inside layers[2].

    Yields:
        tuple: (path, layer), where path is '3' for layers[3] and '2.1' for the second layer
            of the block at position 2.
    """
    for i, layer in enumerate(layers):
        yield f'{prefix}{i}', layer
        if isinstance(layer, ResidualBlock):
            yield from _walk_layers(layer.layers, prefix=f'{prefix}{i}.')


def _find_layer(layers, path):
    """Return the layer at a path such as '3' or '2.1' (see _walk_layers)."""
    indices = [int(x) for x in path.split('.')]
    layer = layers[indices[0]]
    for j in indices[1:]:
        layer = layer.layers[j]
    return layer


def default_space(layers, optimizer=Adam, scheduler=None, share_keepprob=True):
    """Return the hyperparameters worth searching for this network, with default ranges.

    What is included depends on the layers and the optimizer:
        lr, lambda, batch_size, augment    always.
        Beta / Beta1 / Beta2               if the optimizer's constructor has an argument of that name.
        gamma                              if a scheduler is given (decay factor per epoch).
        layer<path>.n_C, layer<path>.f     for every Conv2d (number of filters, kernel size).
        layer<path>.n_out                  for every Dense except the last one (the output layer
                                           must keep its number of classes).
        keepprob                           ONE value used by every Dropout (share_keepprob=True), or
        layer<path>.keepprob               one per Dropout (share_keepprob=False).
        layer<path>.scale                  for Conv2d / Dense built with init=scaled_normal.
    The path is the position of the layer in the list, e.g. layer0.n_C, or layer2.1.n_C for the
    second layer inside the ResidualBlock at position 2.

    Args:
        layers (list): layer objects, in the order of the network.
        optimizer (type): optimizer class, e.g. Adam. It is instantiated for each run.
        scheduler (type or None): learning rate schedule class, e.g. ExponentialDecay, or None.
        share_keepprob (bool): if True (default), all Dropout layers get the same keep probability
            and the space has one entry 'keepprob'. If False, each Dropout is searched separately.

    Returns:
        dict: name -> spec, in a fixed order.
    """
    space = {
        'lr': ('log', 1e-4, 1e-1),
        'lambda': ('log', 1e-5, 1e-2),
        'batch_size': ('choice', BATCH_CHOICES),
        'augment': ('choice', [True, False]),
    }
    params = inspect.signature(optimizer.__init__).parameters
    for name in ('Beta', 'Beta1'):
        if name in params:
            space[name] = ('one_minus_log', 1e-3, 1e-1)  # window 1 / (1 - Beta) from 10 to 1000 steps
    if 'Beta2' in params:
        space['Beta2'] = ('one_minus_log', 1e-4, 1e-2) if optimizer is Adam else ('one_minus_log', 1e-3, 1e-1)
    if scheduler is not None:
        space['gamma'] = ('uniform', 0.8, 1.0)

    last_dense = max((i for i, layer in enumerate(layers) if isinstance(layer, Dense)), default=None)
    for path, layer in _walk_layers(layers):
        if isinstance(layer, Conv2d):
            space[f'layer{path}.n_C'] = ('choice', FILTER_CHOICES)
            space[f'layer{path}.f'] = ('choice', KERNEL_CHOICES)
        elif isinstance(layer, Dense) and path != str(last_dense):
            space[f'layer{path}.n_out'] = ('choice', UNIT_CHOICES)
        elif isinstance(layer, Dropout) and not share_keepprob:
            space[f'layer{path}.keepprob'] = ('uniform', 0.5, 1.0)
        if isinstance(layer, (Conv2d, Dense)) and layer.scale is not None:
            space[f'layer{path}.scale'] = ('log', 1e-3, 1e-1)
    if share_keepprob and any(isinstance(layer, Dropout) for _, layer in _walk_layers(layers)):
        space['keepprob'] = ('uniform', 0.5, 1.0)
    return space


def describe_spec(spec):
    """Return a short human-readable description of a spec."""
    if not isinstance(spec, tuple):
        return f'fixed = {spec}'
    kind = spec[0]
    if kind == 'log':
        return f'log-uniform in [{spec[1]:g}, {spec[2]:g}]'
    if kind == 'one_minus_log':
        return f'1 - log-uniform in [{spec[1]:g}, {spec[2]:g}]'
    if kind == 'uniform':
        return f'uniform in [{spec[1]:g}, {spec[2]:g}]'
    if kind == 'choice':
        return f'one of {list(spec[1])}'
    raise ValueError(f'unknown spec {spec!r}')


MEANING = {
    'lr': 'learning rate (alpha0 if a scheduler is used)',
    'lambda': 'L2 strength',
    'batch_size': 'mini-batch size',
    'augment': 'random shifts of training images',
    'Beta': 'momentum decay',
    'Beta1': 'Adam: decay of the first moment (momentum)',
    'Beta2': 'decay of the second moment (squared gradients)',
    'gamma': 'learning rate decay factor per epoch',
    'n_C': 'number of filters',
    'f': 'kernel size',
    'n_out': 'number of neurons',
    'keepprob': 'dropout keep probability (the same for all Dropout layers when it has no layer prefix)',
    'scale': 'std of the initial weights (init=scaled_normal)',
}


def describe_space(space):
    """Print the search space as a table: name, how it is sampled, what it means.

    Args:
        space (dict): name -> spec, e.g. from default_space().

    Returns:
        None.
    """
    width = max(len(name) for name in space)
    for name, spec in space.items():
        attr = name.split('.')[-1] if name.startswith('layer') else name
        print(f'  {name:<{width}}  {describe_spec(spec):<34}  {MEANING.get(attr, "")}')


def list_hyperparameters(layers, optimizer=Adam, scheduler=None):
    """Print the hyperparameters that make sense to search for this network and return them.

    Also prints what is NOT searched by default but can be added through the space argument of
    main(): any layer attribute as 'layer<path>.<attr>' (e.g. 'layer1.momentum' for BatchNorm2d),
    and the optimizer arguments eps and clip_norm.

    Args:
        layers (list): layer objects, in the order of the network.
        optimizer (type): optimizer class, e.g. Adam.
        scheduler (type or None): learning rate schedule class, or None.

    Returns:
        dict: the default space, name -> spec.
    """
    space = default_space(layers, optimizer, scheduler)
    print(f'Hyperparameters searched by default ({len(space)}):')
    describe_space(space)
    extras = []
    params = inspect.signature(optimizer.__init__).parameters
    for name in ('eps', 'clip_norm'):
        if name in params:
            extras.append(f'{name} (optimizer)')
    if any(isinstance(layer, BatchNorm2d) for _, layer in _walk_layers(layers)):
        extras.append('layer<i>.momentum (BatchNorm2d running averages)')
    print('Can be added through space={...}, not searched by default: ' + ', '.join(extras + [
        'a fixed value instead of a range, e.g. space={"lr": 1e-3}',
        'architecture (number and order of layers): change the layers list instead']))
    return space


def sample_value(spec, rng):
    """Draw one value from a spec (see the top of this section). A fixed value is returned as is."""
    if not isinstance(spec, tuple):
        return spec
    kind = spec[0]
    if kind == 'log':
        return float(10 ** rng.uniform(np.log10(spec[1]), np.log10(spec[2])))
    if kind == 'one_minus_log':
        return float(1 - 10 ** rng.uniform(np.log10(spec[1]), np.log10(spec[2])))
    if kind == 'uniform':
        return float(rng.uniform(spec[1], spec[2]))
    if kind == 'choice':
        options = spec[1]
        return options[int(rng.integers(len(options)))]
    raise ValueError(f'unknown spec {spec!r}')


def sample_config(rng, space):
    """Draw one random configuration from the search space.

    Args:
        rng (np.random.Generator): generator used for all random choices.
        space (dict): name -> spec.

    Returns:
        dict: 'seed' (int, for the weights and the training order) plus one plain Python value
            per name in space, in the order of space.
    """
    config = {'seed': int(rng.integers(0, 10**6))}
    for name, spec in space.items():
        config[name] = sample_value(spec, rng)
    return config


# Values used for a hyperparameter that is not searched, when it is not set by the layers.
DEFAULTS = {'lr': 1e-3, 'lambda': None, 'batch_size': 64, 'augment': False, 'gamma': 0.95}


def default_value(name, layers, optimizer):
    """Return the value a hyperparameter takes when it is not searched.

    Layer attributes keep the value from the layer you passed (e.g. the n_C of your Conv2d),
    optimizer arguments keep the default of the optimizer's constructor, and lr, lambda,
    batch_size, augment and gamma use DEFAULTS.

    Args:
        name (str): name from the search space.
        layers (list): layer objects, as passed to main().
        optimizer (type): optimizer class.

    Returns:
        the fixed value.
    """
    if name in DEFAULTS:
        return DEFAULTS[name]
    if name == 'keepprob':  # shared: the value of the first Dropout in the list
        return next(layer.keepprob for _, layer in _walk_layers(layers) if isinstance(layer, Dropout))
    opt_params = inspect.signature(optimizer.__init__).parameters
    if name in opt_params:
        return opt_params[name].default
    path, attr = name[len('layer'):].rsplit('.', 1)
    return getattr(_find_layer(layers, path), attr)


def check_space(layers, optimizer, scheduler, space):
    """Raise ValueError if a name in space does not match anything in the network.

    Args:
        layers (list): layer objects.
        optimizer (type): optimizer class.
        scheduler (type or None): schedule class or None.
        space (dict): name -> spec.

    Raises:
        ValueError: for a name that is not a known setting, not an argument of the optimizer
            and not an existing attribute of the layer it points to.
    """
    opt_params = inspect.signature(optimizer.__init__).parameters
    for name in space:
        if name == 'seed':
            raise ValueError("'seed' is drawn automatically and cannot be in space")
        if name == 'gamma' and scheduler is None:
            raise ValueError("'gamma' needs a scheduler, e.g. main(..., scheduler=ExponentialDecay)")
        if name in RESERVED or name in opt_params:
            continue
        if not name.startswith('layer') or '.' not in name:
            raise ValueError(f'unknown hyperparameter {name!r}')
        path, attr = name[len('layer'):].rsplit('.', 1)
        try:
            layer = _find_layer(layers, path)
        except (ValueError, IndexError, AttributeError):
            raise ValueError(f'{name!r}: there is no layer at position {path}') from None
        if not hasattr(layer, attr):
            raise ValueError(f'{name!r}: {type(layer).__name__} has no attribute {attr!r}')


# ----------------------------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------------------------

def build_model(layers, config, rng, input_shape=(28, 28, 1)):
    """Build a fresh network from the layer templates and a configuration.

    The template layers are copied, so they are never modified. Every key 'layer<path>.<attr>'
    of the config is written into the copy before build(), then the weights are created and cast
    to DTYPE.

    Args:
        layers (list): layer objects, as passed to main().
        config (dict): configuration from sample_config().
        rng (np.random.Generator): generator used to initialize the weights.
        input_shape (tuple): shape of one example without the batch axis.

    Returns:
        Sequential: built model, ready for fit().
    """
    layers = copy.deepcopy(layers)
    for name, value in config.items():
        if name.startswith('layer'):
            path, attr = name[len('layer'):].rsplit('.', 1)
            setattr(_find_layer(layers, path), attr, value)
    if 'keepprob' in config:  # one value for every Dropout layer
        for _, layer in _walk_layers(layers):
            if isinstance(layer, Dropout):
                layer.keepprob = config['keepprob']
    model = Sequential(layers, input_shape=input_shape, rng=rng, l2_lambda=config.get('lambda'))
    model.build()
    model.astype(DTYPE)  # build() makes float64 weights; cast them so float32 data stays float32
    return model


def make_optimizer(config, optimizer, scheduler):
    """Create the optimizer (and the schedule) for one configuration.

    Args:
        config (dict): configuration; keys that are arguments of the optimizer's constructor
            (Beta1, Beta2, Beta, eps, clip_norm) are passed to it, and 'gamma' to the schedule.
        optimizer (type): optimizer class.
        scheduler (type or None): schedule class or None.

    Returns:
        tuple: (optimizer instance, schedule instance or None).
    """
    opt_params = inspect.signature(optimizer.__init__).parameters
    opt = optimizer(**{k: v for k, v in config.items() if k in opt_params})
    lr_scheduler = scheduler(gamma=config['gamma']) if scheduler is not None else None
    return opt, lr_scheduler


def run_experiment(config, layers, optimizer, scheduler, X_train, Y_train, X_dev, Y_dev,
                   max_epochs=3, input_shape=(28, 28, 1)):
    """Train one configuration and return its metrics.

    Args:
        config (dict): configuration from sample_config().
        layers (list): layer templates, as passed to main().
        optimizer (type): optimizer class. Arguments of its constructor found in config
            (e.g. Beta1, Beta2, clip_norm) are passed to it.
        scheduler (type or None): schedule class, built with gamma=config['gamma'], or None.
        X_train, Y_train (np.ndarray): training subset, one-hot labels.
        X_dev, Y_dev (np.ndarray): dev set, one-hot labels.
        max_epochs (int): number of epochs. Keep it small for a search.
        input_shape (tuple): shape of one example without the batch axis.

    Returns:
        dict: the configuration plus metrics: n_params, best_dev_acc and best_dev_epoch (the
            highest dev accuracy and the epoch where it occurred), final_train_acc,
            final_dev_acc and minutes (wall-clock time of the run).
    """
    rng = np.random.default_rng(config['seed'])
    model = build_model(layers, config, rng, input_shape)
    n_params = model.summary()[0]['Total Parameters']

    opt, lr_scheduler = make_optimizer(config, optimizer, scheduler)

    start = time.perf_counter()
    output = model.fit(SoftmaxCrossEntropy(), X_train, Y_train, X_dev, Y_dev,
                       optimizer=opt, max_epochs=max_epochs,
                       batch_size=config.get('batch_size', 64), lr=config['lr'],
                       lr_scheduler=lr_scheduler, augment=config.get('augment', False),
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


METRIC_COLUMNS = ['n_params', 'best_dev_acc', 'best_dev_epoch', 'final_train_acc', 'final_dev_acc',
                  'minutes']


def csv_columns(space):
    """Return the CSV header: run, seed, one column per hyperparameter, then the metrics."""
    return ['run', 'seed'] + list(space) + METRIC_COLUMNS


def flatten_result(result, columns):
    """Turn one result dict from run_experiment() into one flat row for the CSV file.

    Args:
        result (dict): result from run_experiment(), with the key 'run' added by main().
        columns (list of str): header from csv_columns().

    Returns:
        dict: one value per name in columns.
    """
    row = {'run': result['run']}
    row.update(result['config'])
    for key in METRIC_COLUMNS:
        row[key] = result[key]
    return {c: row[c] for c in columns}


# ----------------------------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------------------------

def main(layers, optimizer=Adam, scheduler=None, space=None, search=None, share_keepprob=True, name=None,
         n_runs=10, n_train=10000, seed=0,
         max_epochs=3, input_shape=(28, 28, 1), data=None, out_path=None):
    """Run the random search, append every result to a CSV file and return them.

    Prints the search space first. Can be called from a notebook: pass the arrays you already
    loaded, so the CSV files are not read again. The header is written once, when the file is
    created or empty; later calls with the same space append rows. If the file already has a
    different header (another network or space), a ValueError asks for a different out_path.

    Args:
        layers (list): layer objects in the order of the network, e.g. Conv2d, BatchNorm2d, ReLU,
            Dropout, Flatten, Dense(10). The last Dense is the output layer. They are templates:
            each run builds a fresh copy, and the hyperparameters in the space overwrite
            attributes of the copy (n_C, f, n_out, keepprob, ...).
        optimizer (type): optimizer class, e.g. Adam, Momentum, RMSprop, SGD (not an instance).
        scheduler (type or None): schedule class, e.g. ExponentialDecay, or None for a constant
            learning rate.
        space (dict or None): changes to the default space. Each entry replaces the default one
            or adds a new name: a spec to search it (see the top of the search space section) or
            a plain value to fix it, e.g. {'lr': 1e-3, 'layer0.n_C': ('choice', [4, 8])}.
            None uses default_space() unchanged.
        search (list of str or None): names of the hyperparameters to search, e.g.
            ['lr', 'lambda']. All the others are fixed: layer attributes keep the value from
            your layers, optimizer arguments keep the optimizer's default, and lr, lambda,
            batch_size, augment, gamma use DEFAULTS. None searches everything in the space.
        share_keepprob (bool): True (default): all Dropout layers use the same keep probability,
            searched as one hyperparameter 'keepprob'. False: one per Dropout layer.
        name (str or None): file name of the CSV inside outputs/, e.g. 'tuning_mlp' or
            'tuning_mlp.csv' (.csv is added if missing). Ignored if out_path is given.
            None uses 'tuning.csv'.
        n_runs (int): number of random configurations to try.
        n_train (int): size of the training subset used by every run.
        seed (int): seed of the generator that draws the configurations.
        max_epochs (int): number of epochs per run.
        input_shape (tuple): shape of one example without the batch axis.
        data (tuple or None): the six arrays returned by load_data(), in the same order:
            (X_train, X_dev, X_test, Y_train, Y_dev, Y_test). X_test is not used. If None,
            the data is loaded from the CSV files in data/.
        out_path (str, pathlib.Path or None): CSV file the results are appended to. If None,
            outputs/<name> is used (outputs/tuning.csv if name is None).

    Returns:
        list of dict: one result per run, as returned by run_experiment(), plus the key 'run'.
    """
    full_space = default_space(layers, optimizer, scheduler, share_keepprob)
    full_space.update(space or {})
    check_space(layers, optimizer, scheduler, full_space)
    if search is not None:
        unknown = [hp for hp in search if hp not in full_space]
        if unknown:
            raise ValueError(f'search has names that are not in the space: {unknown}. '
                             f'Available: {list(full_space)}')
        for hp, spec in full_space.items():  # not `name`: that is the argument with the CSV name
            if hp not in search and isinstance(spec, tuple):
                full_space[hp] = default_value(hp, layers, optimizer)
    n_searched = sum(isinstance(spec, tuple) for spec in full_space.values())
    print(f'Searched: {n_searched}, fixed: {len(full_space) - n_searched}')
    describe_space(full_space)

    if data is None:
        data = load_data(DATA_DIR / 'fashion-mnist_train.csv',
                         DATA_DIR / 'fashion-mnist_test.csv',
                         dtype=DTYPE)
    X_train, X_dev, _, Y_train, Y_dev, _ = data
    X_train, Y_train = X_train[:n_train], Y_train[:n_train]

    columns = csv_columns(full_space)
    rng = np.random.default_rng(seed)
    if out_path is None:
        file_name = 'tuning.csv' if name is None else (name if name.endswith('.csv') else name + '.csv')
        out_path = OUT_DIR / file_name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    if not write_header:
        with open(out_path, newline='', encoding='utf-8') as f:
            header = next(csv.reader(f))
        if header != columns:
            raise ValueError(f'{out_path} has different columns than this search '
                             f'(another network or space). Use another out_path.')

    results = []
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for run in range(n_runs):
            config = sample_config(rng, full_space)
            result = run_experiment(config, layers, optimizer, scheduler, X_train, Y_train,
                                    X_dev, Y_dev, max_epochs=max_epochs, input_shape=input_shape)
            result['run'] = run
            results.append(result)
            writer.writerow(flatten_result(result, columns))
            f.flush()
            print(f"run {run}: best dev {result['best_dev_acc']:.4f} "
                  f"(epoch {result['best_dev_epoch']}), params {result['n_params']}, "
                  f"{result['minutes']:.1f} min")
    return results


# ----------------------------------------------------------------------------------------------
# Best configuration, final training, saving and restoring
# ----------------------------------------------------------------------------------------------

def _parse(text):
    """Turn a CSV cell back into a Python value: '' -> None, '8' -> 8, '0.1' -> 0.1, 'True' -> True."""
    if text == '':
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def best_config(csv_path, key='best_dev_acc'):
    """Read a tuning CSV and return the configuration of the best run.

    Args:
        csv_path (str or pathlib.Path): file written by main().
        key (str): metric column to maximize. Default 'best_dev_acc'.

    Returns:
        tuple: (config, metrics). config is a dict that can be passed to build_model() and
            make_optimizer(); metrics is a dict with run, n_params, best_dev_acc, ... of that run.
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    row = max(rows, key=lambda r: float(r[key]))
    metrics = {k: _parse(row[k]) for k in ['run'] + METRIC_COLUMNS}
    config = {k: _parse(v) for k, v in row.items() if k != 'run' and k not in METRIC_COLUMNS}
    return config, metrics


def layer_code(layer, indent=0):
    """Return Python source that re-creates a layer, e.g. "Conv2d(32, f=3, s=1, p='same', ...)".

    Args:
        layer: a layer object (Conv2d, Dense, Dropout, BatchNorm2d, MaxPool2d, ReLU, Flatten,
            GlobalAvgPool or ResidualBlock).
        indent (int): number of spaces in front of the lines inside a ResidualBlock.

    Returns:
        str: one line, or several lines for a ResidualBlock.
    """
    if isinstance(layer, ResidualBlock):
        pad = ' ' * (indent + 4)
        inner = ',\n'.join(pad + layer_code(inner_layer, indent + 4) for inner_layer in layer.layers)
        return f'ResidualBlock([\n{inner}])'
    if isinstance(layer, (Conv2d, Dense)):
        scale = f', scale={layer.scale}' if layer.scale is not None else ''
        if isinstance(layer, Conv2d):
            return (f"Conv2d({layer.n_C}, f={layer.f}, s={layer.s}, p={layer.p!r}, bias={layer.bias}, "
                    f"init={layer.init.__name__}{scale})")
        return f"Dense({layer.n_out}, bias={layer.bias}, init={layer.init.__name__}{scale})"
    if isinstance(layer, Dropout):
        return f'Dropout({layer.keepprob})'
    if isinstance(layer, BatchNorm2d):
        return f'BatchNorm2d(axes={layer.axes}, momentum={layer.momentum})'
    if isinstance(layer, MaxPool2d):
        return f'MaxPool2d(f={layer.f}, s={layer.s})'
    return f'{type(layer).__name__}()'


def architecture_code(model):
    """Return the whole network as Python source (one layer per line), for the architecture docs.

    The result is a list literal that you can paste into a notebook to rebuild the exact
    network, then call model.load_weights() on the saved .npz file.

    Args:
        model (Sequential): built model.

    Returns:
        str: source of the layer list.
    """
    return '[' + ',\n '.join(layer_code(layer, indent=1) for layer in model.layers) + ']'


def train_final(layers, config, name, X_train, Y_train, X_dev, Y_dev, optimizer=Adam, scheduler=None,
                epochs=8, input_shape=(28, 28, 1), out_dir=None):
    """Train one configuration on the given data, then save the weights and a description.

    Writes two files into out_dir (default: ROOT / 'trained_models'):
        <name>.npz   weights, written with Sequential.save()
        <name>.json  everything needed to restore and describe the model: the configuration, the
                     architecture as Python source, n_params, final metrics, training time and the
                     per-epoch history (loss and accuracy on train and dev).

    Args:
        layers (list): layer templates, as passed to main().
        config (dict): hyperparameters, e.g. from best_config().
        name (str): file name without extension, e.g. 'conv_deep'.
        X_train, Y_train, X_dev, Y_dev (np.ndarray): data, one-hot labels.
        optimizer (type): optimizer class.
        scheduler (type or None): schedule class or None.
        epochs (int): number of epochs.
        input_shape (tuple): shape of one example without the batch axis.
        out_dir (str, pathlib.Path or None): folder for the files, created if missing.

    Returns:
        dict: the content of the JSON file (history keys are epochs as ints).
    """
    out_dir = ROOT / 'trained_models' if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config['seed'])
    model = build_model(layers, config, rng, input_shape)
    n_params = int(model.summary()[0]['Total Parameters'])
    opt, lr_scheduler = make_optimizer(config, optimizer, scheduler)

    start = time.perf_counter()
    output = model.fit(SoftmaxCrossEntropy(), X_train, Y_train, X_dev, Y_dev,
                       optimizer=opt, max_epochs=epochs,
                       batch_size=config.get('batch_size', 64), lr=config['lr'],
                       lr_scheduler=lr_scheduler, augment=config.get('augment', False),
                       return_=['all'], print_epochs=1, seed=config['seed'])
    minutes = (time.perf_counter() - start) / 60
    (J_train, J_dev), (acc_train, acc_dev), _ = output[0]

    model.save(str(out_dir / name))
    best_epoch = max(acc_dev, key=acc_dev.get)
    record = {
        'name': name, 'config': config, 'optimizer': optimizer.__name__,
        'scheduler': None if scheduler is None else scheduler.__name__, 'epochs': epochs,
        'input_shape': list(input_shape), 'n_params': n_params, 'minutes': minutes,
        'best_dev_acc': float(acc_dev[best_epoch]), 'best_dev_epoch': int(best_epoch),
        'final_train_acc': float(acc_train[epochs - 1]), 'final_dev_acc': float(acc_dev[epochs - 1]),
        'architecture': architecture_code(model),
        'history': {'J_train': {int(k): float(v) for k, v in J_train.items()},
                    'J_dev': {int(k): float(v) for k, v in J_dev.items()},
                    'acc_train': {int(k): float(v) for k, v in acc_train.items()},
                    'acc_dev': {int(k): float(v) for k, v in acc_dev.items()}},
    }
    with open(out_dir / f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return record


def load_record(name, out_dir=None):
    """Read <name>.json written by train_final(); history keys are turned back into ints."""
    out_dir = ROOT / 'trained_models' if out_dir is None else Path(out_dir)
    with open(out_dir / f'{name}.json', encoding='utf-8') as f:
        record = json.load(f)
    record['history'] = {k: {int(e): v for e, v in h.items()} for k, h in record['history'].items()}
    return record


def load_trained(layers, name, out_dir=None):
    """Rebuild a model saved by train_final() and load its weights.

    Args:
        layers (list): the same layer templates that were used for training (the dict entry
            from your notebook). The saved configuration is applied to a copy of them.
        name (str): file name without extension, e.g. 'conv_deep'.
        out_dir (str, pathlib.Path or None): folder with <name>.npz and <name>.json.

    Returns:
        Sequential: model ready for evaluate() / predict(). It is in test mode by default.
    """
    out_dir = ROOT / 'trained_models' if out_dir is None else Path(out_dir)
    record = load_record(name, out_dir)
    model = build_model(layers, record['config'], np.random.default_rng(record['config']['seed']),
                        tuple(record['input_shape']))
    model.load_weights(str(out_dir / f'{name}.npz'))
    return model


def default_layers():
    """Layers used when the module is run from a terminal: 2 x (Conv -> BN -> ReLU), Dropout, Dense."""
    return [Conv2d(16, f=3, s=1, p='same', bias=False, init=he),
            BatchNorm2d(axes=(0, 1, 2)),
            ReLU(),
            Conv2d(16, f=3, s=1, p='same', bias=False, init=he),
            BatchNorm2d(axes=(0, 1, 2)),
            ReLU(),
            Dropout(0.8),
            Flatten(),
            Dense(10, bias=True, init=xavier)]


if __name__ == '__main__':
    main(default_layers(), Adam)
