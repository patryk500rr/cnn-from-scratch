"""Training loop: mini-batch SGD with per-epoch evaluation on train and dev."""
from data import get_minibatch
from evaluation import evaluate
import numpy as np


def fit(model, loss, optimizer, X_train, Y_train, X_dev, Y_dev, max_epochs, batch_size, lr, rng):
    """Train the model with mini-batch SGD and evaluate after every epoch.

    Each epoch shuffles the training set with rng, runs forward, loss, backward and
    the optimizer step for every mini-batch, and then evaluates the whole train and
    dev sets. Progress is printed once per epoch.

    Args:
        model (Sequential): network with forward, backward and parameters().
        loss (SoftmaxCrossEntropy): loss layer, used as loss.forward(Z, Y) and loss.backward().
        optimizer (callable): update function, called as optimizer(model.parameters(), lr).
        X_train (np.ndarray): training inputs, shape (m_train, 28, 28, 1).
        Y_train (np.ndarray): one-hot training labels, shape (m_train, 10).
        X_dev (np.ndarray): dev inputs, shape (m_dev, 28, 28, 1).
        Y_dev (np.ndarray): one-hot dev labels, shape (m_dev, 10).
        max_epochs (int): number of epochs to run.
        batch_size (int): mini-batch size for training and evaluation.
        lr (float): learning rate passed to the optimizer.
        rng (np.random.Generator): generator used to shuffle the training data each epoch.
            Pass the same object every epoch.

    Yields:
        tuple: (J_dict_train, J_dict_dev, acc_train_dict, acc_dev_dict) after each epoch.
            Each dict maps the epoch index to a value. The dicts are shared between
            yields and keep growing, so copy them if you need a snapshot.
    """

    J_dict_train = {}
    J_dict_dev = {}
    acc_train_dict = {}
    acc_dev_dict = {}
    for i in range(max_epochs):
        batch_loader_train = get_minibatch(X_train, Y_train, batch_size, rng)
        while True:
            try:
                X_mb_train, Y_mb_train = next(batch_loader_train)
            except StopIteration: break

            Z = model.forward(X_mb_train)
            loss.forward(Z, Y_mb_train)

            dZ = loss.backward()
            model.backward(dZ)

            optimizer(model.parameters(), lr)

        J_list_train, acc_list_train = evaluate(model, loss, X_train, Y_train, batch_size)
        J_dev, acc_dev = evaluate(model, loss, X_dev, Y_dev)

        J_dict_train[i] = np.mean(J_list_train)
        acc_train_dict[i] = np.mean(acc_list_train)

        J_dict_dev[i] = J_dev
        acc_dev_dict[i] = acc_dev

        print(f'Epoch: {i}, loss: {round(np.mean(J_list_train), 6)}, accuracy train: {round(np.mean(acc_list_train), 3)}, accuracy dev: {round(acc_dev, 3)}')

        yield J_dict_train, J_dict_dev, acc_train_dict, acc_dev_dict


        


