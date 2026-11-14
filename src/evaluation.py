"""Evaluation metrics and plots of the training history."""
import numpy as np
import matplotlib.pyplot as plt


def accuracy(Y_hat, Y):
    """Fraction of examples whose predicted class equals the true class.

    Args:
        Y_hat (np.ndarray): predicted class indices, shape (m,), e.g. Z.argmax(axis=1).
        Y (np.ndarray): true class indices, shape (m,), e.g. Y_one_hot.argmax(axis=1).

    Returns:
        float: accuracy in [0, 1].
    """
    return np.mean(Y_hat == Y)


def compare_models(model1_output, model2_output, model1_lr, model2_lr, model1_name='Model 1', model2_name='Model 2'):
    """Plot the loss and accuracy histories of two models in one 2x2 figure.

    Top row: loss on train and dev. Bottom row: accuracy on train and dev.
    Both models are drawn in each subplot, with a shared legend at the bottom.

    Args:
        model1_output (list): return value of Sequential.fit with return_=['all'].
        model2_output (list): return value of Sequential.fit with return_=['all'].
        model1_lr (float): learning rate of model 1, shown in the figure title.
        model2_lr (float): learning rate of model 2, shown in the figure title.
        model1_name (str): label of model 1 in the legend.
        model2_name (str): label of model 2 in the legend.

    Returns:
        matplotlib.figure.Figure: the figure (also shown with plt.show()).
    """
    J_history_1, accuracy_history_1 = model1_output[0][0], model1_output[0][1]
    J_history_2, accuracy_history_2 = model2_output[0][0], model2_output[0][1]

    J_history_train_1, J_history_dev_1 = J_history_1
    J_history_train_2, J_history_dev_2 = J_history_2

    accuracy_history_train_1, accuracy_history_dev_1 = accuracy_history_1
    accuracy_history_train_2, accuracy_history_dev_2 = accuracy_history_2

    epochs_1 = max(J_history_train_1.keys())
    epochs_2 = max(J_history_train_2.keys())

    max_epochs = max(epochs_1, epochs_2)

    fig, ax = plt.subplots(2, 2, sharey='row', figsize=(10, 7))

    ax[0, 0].plot(list(J_history_train_1.keys()), list(J_history_train_1.values()), label=model1_name)
    ax[0, 0].plot(list(J_history_train_2.keys()), list(J_history_train_2.values()), label=model2_name)
    ax[0, 0].set_title('Loss Train')

    ax[0, 1].plot(list(J_history_dev_1.keys()), list(J_history_dev_1.values()), label=model1_name)
    ax[0, 1].plot(list(J_history_dev_2.keys()), list(J_history_dev_2.values()), label=model2_name)
    ax[0, 1].set_title('Loss Dev')

    ax[1, 0].plot(list(accuracy_history_train_1.keys()), list(accuracy_history_train_1.values()), label=model1_name)
    ax[1, 0].plot(list(accuracy_history_train_2.keys()), list(accuracy_history_train_2.values()), label=model2_name)
    ax[1, 0].set_title('Accuracy Train')

    ax[1, 1].plot(list(accuracy_history_dev_1.keys()), list(accuracy_history_dev_1.values()), label=model1_name)
    ax[1, 1].plot(list(accuracy_history_dev_2.keys()), list(accuracy_history_dev_2.values()), label=model2_name)
    ax[1, 1].set_title('Accuracy Dev')

    for axs_loss in ax[0]:
        axs_loss.set_xlabel('Epoch Num')
        axs_loss.set_ylabel('Loss')

    for axs_accuracy in ax[1]:
        axs_accuracy.set_xlabel('Epoch Num')
        axs_accuracy.set_ylabel('Accuracy')

    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row, col].grid(alpha=0.3)
            ax[row, col].set_xticks(np.arange(0, max_epochs + 10, 10))

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(f'Model Comparison, {model1_name} lr= {model1_lr}, {model2_name} lr= {model2_lr}')
    plt.tight_layout()
    plt.show()

    return fig


def plot_history(model_output, model_name=None):
    """Plot the loss and accuracy history of one model in one figure with two subplots.

    The accuracy plot also shows a dashed line at the best dev accuracy.

    Args:
        model_output (list): return value of Sequential.fit with return_=['all'].
        model_name (str or None): shown in the title. None gives a title without a name.

    Returns:
        matplotlib.figure.Figure: the figure (also shown with plt.show()).
    """
    J_history, accuracy_history = model_output[0][0], model_output[0][1]

    J_history_train, J_history_dev = J_history
    accuracy_history_train, accuracy_history_dev = accuracy_history

    max_epochs = max(J_history_train.keys())
    max_dev_accuracy = max(accuracy_history_dev.values())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(list(J_history_train.keys()), list(J_history_train.values()), label='Train')
    ax[0].plot(list(J_history_dev.keys()), list(J_history_dev.values()), label='Dev')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_xticks(np.arange(0, max_epochs + 10, 10))
    ax[0].grid(alpha=0.3)
    ax[0].set_title('Loss')

    ax[1].plot(list(accuracy_history_train.keys()), list(accuracy_history_train.values()), label='Train')
    ax[1].plot(list(accuracy_history_dev.keys()), list(accuracy_history_dev.values()), label='Dev')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xticks(np.arange(0, max_epochs + 10, 10))
    ax[1].grid(alpha=0.3)
    ax[1].axhline(y=max_dev_accuracy, alpha=0.8, linestyle='--', color='green', label=f'Max dev accuracy: {round(max_dev_accuracy * 100, 1)}%')
    ax[1].set_title('Accuracy')

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))

    title = 'Model History' if model_name is None else f'Model History, {model_name}'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig
