"""Evaluation metrics and plots of the training history."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import ewa
from layers import SoftmaxCrossEntropy


def accuracy(Y_hat, Y):
    """Fraction of examples whose predicted class equals the true class.

    Args:
        Y_hat (np.ndarray): predicted class indices, shape (m,), e.g. Z.argmax(axis=1).
        Y (np.ndarray): true class indices, shape (m,), e.g. Y_one_hot.argmax(axis=1).

    Returns:
        float: accuracy in [0, 1].
    """
    return np.mean(Y_hat == Y)


def _history_arrays(history):
    """Turn a {epoch: value} dictionary into two arrays sorted by epoch."""
    epochs = np.array(sorted(history.keys()))
    values = np.array([history[e] for e in epochs])
    return epochs, values


def compare_models(outputs, names=None, lrs=None, smooth_beta=0.8, title='Porównanie modeli', return_fig=False):
    """Plot the training histories of several models in one 2x2 figure and report the best dev accuracy.

    Panels: loss on train, loss on dev, accuracy on train, accuracy on dev. Each model has
    its own colour. On the dev panels a thin line shows the raw values and a thick line shows
    the smoothed trend (exponentially weighted average, see ewa in utils.py). The best dev
    accuracy of each model is marked with a star and listed in a box on the accuracy dev panel.
    The best model overall is printed to the console.

    Args:
        outputs (list): one return value of Sequential.fit with return_=['all'] per model.
        names (list of str or None): model names for the legend, same length as outputs.
            None gives 'Model 1', 'Model 2', ...
        lrs (list of float or None): learning rate of each model, same length as outputs.
            Shown in the legend. None omits the learning rates.
        smooth_beta (float): beta of the exponentially weighted average used for the smoothed
            dev curves, in (0, 1). Larger values give a smoother curve. Default 0.8.
        title (str): title of the whole figure. Default 'Porównanie modeli'.
        return_fig (bool): if True, the function returns the matplotlib Figure, so it can be
            saved with fig.savefig(). If False (default), it returns None.

    Returns:
        matplotlib.figure.Figure if return_fig is True, otherwise None. The figure is shown
        with plt.show() in both cases.
    """
    n_models = len(outputs)
    if names is None:
        names = [f'Model {i + 1}' for i in range(n_models)]
    if lrs is None:
        lrs = [None] * n_models
    assert len(names) == n_models, 'names must have one entry per model'
    assert len(lrs) == n_models, 'lrs must have one entry per model'

    colors = plt.cm.tab10(np.arange(n_models) % 10)

    histories = []
    for output in outputs:
        J_history, accuracy_history = output[0][0], output[0][1]
        J_train, J_dev = J_history
        acc_train, acc_dev = accuracy_history
        histories.append({'J_train': J_train, 'J_dev': J_dev, 'acc_train': acc_train, 'acc_dev': acc_dev})

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = [
        (ax[0, 0], 'J_train', 'Loss, train', 'Loss', False),
        (ax[0, 1], 'J_dev', 'Loss, dev', 'Loss', True),
        (ax[1, 0], 'acc_train', 'Accuracy, train', 'Accuracy', False),
        (ax[1, 1], 'acc_dev', 'Accuracy, dev', 'Accuracy', True),
    ]

    best_per_model = []
    for i, (history, name, lr, color) in enumerate(zip(histories, names, lrs, colors)):
        label = f'{name} (lr={lr})' if lr is not None else name

        for panel_ax, key, panel_title, ylabel, smoothed in panels:
            epochs, values = _history_arrays(history[key])
            if smoothed:
                panel_ax.plot(epochs, values, color=color, linewidth=1, alpha=0.3)
                panel_ax.plot(epochs, ewa(values, smooth_beta, bias_correction=True),
                              color=color, linewidth=2.5, label=label)
            else:
                panel_ax.plot(epochs, values, color=color, linewidth=2, label=label)
            panel_ax.set_title(panel_title, fontsize=12)
            panel_ax.set_ylabel(ylabel)

        epochs_acc, values_acc = _history_arrays(history['acc_dev'])
        best_idx = int(np.argmax(values_acc))
        best_epoch, best_acc = int(epochs_acc[best_idx]), float(values_acc[best_idx])
        best_per_model.append((name, lr, best_epoch, best_acc))

        ax[1, 1].scatter(best_epoch, best_acc, color=color, marker='*', s=220,
                         edgecolor='black', linewidth=0.6, zorder=5)

    for panel_ax in ax.flat:
        panel_ax.grid(alpha=0.3)
        panel_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        panel_ax.set_xlabel('Epoch')

    summary_lines = ['Max dev accuracy:']
    for name, lr, epoch, acc in best_per_model:
        summary_lines.append(f'{name}: {acc:.2%} (epoch {epoch})')
    ax[1, 1].text(0.98, 0.03, '\n'.join(summary_lines), transform=ax[1, 1].transAxes,
                  fontsize=9, va='bottom', ha='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='lightgray'))

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.045),
               ncol=min(n_models, 4), frameon=True)

    fig.text(0.5, 0.005, f'Thin line: raw values, thick line: smoothed trend (EWA, beta={smooth_beta}). '
                         'Star: best dev accuracy.',
             ha='center', fontsize=9, color='dimgray')

    best_name, best_lr, best_epoch, best_acc = max(best_per_model, key=lambda row: row[3])
    best_label = f'{best_name} (lr={best_lr})' if best_lr is not None else best_name
    print(f'Najlepszy dev accuracy: {best_label}, {best_acc:.4f} (epoka {best_epoch})')

    fig.suptitle(title, fontsize=15, y=0.99)
    plt.tight_layout(rect=(0, 0.1, 1, 0.96))
    plt.show()

    if return_fig:
        return fig


def plot_history(model_output, model_name=None, lr=None, smooth_beta=0.8, title='Training history', return_fig=False):
    """Plot the training history of one model in two panels and print a summary.

    Left: loss on train (solid) and dev (thin raw line, thick smoothed line). The lowest dev
    loss is marked with a star. Right: the same for accuracy, with a dashed line at the best
    dev accuracy and a star at its epoch. A box on the right lists the key numbers: best and
    final dev accuracy, final train accuracy, the train-dev gap, lowest and final dev loss,
    and the total training time if it was recorded. The same summary is printed.

    Args:
        model_output (list): return value of Sequential.fit with return_=['all'].
        model_name (str or None): shown in the title. None gives a title without a name.
        lr (float or None): learning rate, shown in the title. None omits it.
        smooth_beta (float): beta of the exponentially weighted average for the smoothed dev
            curves, in (0, 1). Default 0.8.
        title (str): base title of the figure. Default 'Training history'. The model name and
            the learning rate, if given, are appended to it.
        return_fig (bool): if True, the function returns the matplotlib Figure, so it can be
            saved with fig.savefig(). If False (default), it returns None.

    Returns:
        matplotlib.figure.Figure if return_fig is True, otherwise None. The figure is shown
        with plt.show() in both cases.
    """
    J_history, accuracy_history, time_dict = model_output[0][0], model_output[0][1], model_output[0][2]
    epochs, J_train = _history_arrays(J_history[0])
    _, J_dev = _history_arrays(J_history[1])
    _, acc_train = _history_arrays(accuracy_history[0])
    _, acc_dev = _history_arrays(accuracy_history[1])

    color_train, color_dev = 'tab:blue', 'tab:orange'

    best_idx = int(np.argmax(acc_dev))
    best_epoch, best_acc = int(epochs[best_idx]), float(acc_dev[best_idx])
    min_loss_idx = int(np.argmin(J_dev))
    min_loss_epoch, min_loss = int(epochs[min_loss_idx]), float(J_dev[min_loss_idx])
    total_minutes = sum(value[0] for value in time_dict.values()) if time_dict else None

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(epochs, J_train, color=color_train, linewidth=2, label='Train')
    ax[0].plot(epochs, J_dev, color=color_dev, linewidth=1, alpha=0.3)
    ax[0].plot(epochs, ewa(J_dev, smooth_beta, bias_correction=True), color=color_dev,
               linewidth=2.5, label='Dev (smoothed)')
    ax[0].scatter(min_loss_epoch, min_loss, color=color_dev, marker='*', s=220,
                  edgecolor='black', linewidth=0.6, zorder=5, label='Lowest dev loss')
    ax[0].set_title('Loss', fontsize=12)
    ax[0].set_ylabel('Loss')

    ax[1].plot(epochs, acc_train, color=color_train, linewidth=2, label='Train')
    ax[1].plot(epochs, acc_dev, color=color_dev, linewidth=1, alpha=0.3)
    ax[1].plot(epochs, ewa(acc_dev, smooth_beta, bias_correction=True), color=color_dev,
               linewidth=2.5, label='Dev (smoothed)')
    ax[1].axhline(best_acc, color='green', linestyle='--', linewidth=1.2, alpha=0.8,
                  label=f'Best dev accuracy: {best_acc:.2%}')
    ax[1].scatter(best_epoch, best_acc, color=color_dev, marker='*', s=220,
                  edgecolor='black', linewidth=0.6, zorder=5, label='Best dev epoch')
    ax[1].set_title('Accuracy', fontsize=12)
    ax[1].set_ylabel('Accuracy')

    for panel_ax in ax:
        panel_ax.grid(alpha=0.3)
        panel_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        panel_ax.set_xlabel('Epoch')

    ax[0].legend(loc='upper right', frameon=True)
    ax[1].legend(loc='lower right', frameon=True)

    summary_lines = [
        f'Best dev accuracy: {best_acc:.2%} (epoch {best_epoch})',
        f'Final dev accuracy: {acc_dev[-1]:.2%}',
        f'Final train accuracy: {acc_train[-1]:.2%}',
        f'Train-dev gap: {(acc_train[-1] - acc_dev[-1]) * 100:.1f} pp',
        f'Lowest dev loss: {min_loss:.4f} (epoch {min_loss_epoch})',
        f'Final dev loss: {J_dev[-1]:.4f}',
        f'Epochs: {len(epochs)}',
    ]
    if total_minutes is not None:
        summary_lines.append(f'Training time: {total_minutes:.1f} min')
    ax[0].text(0.98, 0.45, '\n'.join(summary_lines), transform=ax[0].transAxes,
               fontsize=9, va='center', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='lightgray'))

    full_title = title
    if model_name is not None:
        full_title += f': {model_name}'
    if lr is not None:
        full_title += f', lr={lr}'
    fig.suptitle(full_title, fontsize=14)
    fig.text(0.5, 0.005, f'Thin line: raw dev values, thick line: smoothed trend (EWA, beta={smooth_beta}).',
             ha='center', fontsize=9, color='dimgray')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

    print(f'Best dev accuracy: {best_acc:.4f} (epoch {best_epoch}); '
          f'final dev accuracy: {acc_dev[-1]:.4f}; lowest dev loss: {min_loss:.4f} (epoch {min_loss_epoch})')

    if return_fig:
        return fig


def plot_gradients(model, X, Y, title='Gradienty w warstwach', return_fig=False):
    """Plot the gradient norm of every layer with weights on a log scale, labelled with model.summary().

    One forward and one backward pass are run on the batch (X, Y) in training mode, so the
    gradients are the ones fit() would compute. For each layer with a weight matrix (Dense, Conv2d)
    the norm ||dW|| is divided by sqrt(number of elements), so layers of different sizes are
    comparable. The x-axis labels come from model.summary(), in layer order, and the title shows
    the total number of parameters.

    Args:
        model (Sequential): built model; build() must have been called.
        X (np.ndarray): batch of inputs, e.g. X_train[:256].
        Y (np.ndarray): one-hot labels for X, shape (m, n_classes).
        title (str): title of the figure. Default 'Gradienty w warstwach'.
        return_fig (bool): if True, the function returns the matplotlib Figure. Default False.

    Returns:
        matplotlib.figure.Figure if return_fig is True, otherwise None. The figure is shown
        with plt.show() in both cases.

    Raises:
        AssertionError: if the number of layers with gradients does not match the summary.
    """
    params_dict, shapes_dict = model.summary()
    total_params = params_dict['Total Parameters']

    loss = SoftmaxCrossEntropy()
    Z = model.forward(X, training=True)
    loss.forward(Z, Y)
    model.backward(loss.backward())

    weight_layers = [layer for layer in model.layers if 'dW' in layer.grads]
    labels = [key for key in shapes_dict if key.split(' ')[0] in ('Dense', 'Conv2d')]
    assert len(labels) == len(weight_layers), 'summary and model layers do not match'

    norms = np.array([np.linalg.norm(layer.grads['dW']) / np.sqrt(layer.grads['dW'].size)
                      for layer in weight_layers])

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(norms) + 2), 4.5))
    x = np.arange(len(norms))
    ax.semilogy(x, np.where(norms > 0, norms, np.nan), marker='o', color='tab:blue')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('||dW|| / sqrt(liczba elementow)  [log]')
    ax.grid(alpha=0.3, which='both')
    ax.set_title(f'{title}, parametry: {total_params:,}')
    plt.tight_layout()
    plt.show()

    if return_fig:
        return fig
