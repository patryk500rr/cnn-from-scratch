import numpy as np
from layers import Dense, ReLU, Flatten, SoftmaxCrossEntropy
from model import Sequential
from optimizers import step_SGD
from train import fit
from data import load_data
from pathlib import Path

data_train_path = Path('../data/fashion-mnist_train.csv')
data_test_path = Path('../data/fashion-mnist_test.csv')

X_train, X_dev, _, Y_train, Y_dev, _ = load_data(data_train_path, data_test_path)

rng = np.random.default_rng(0)
model = Sequential([
    Flatten(),                                                              # (m, 28, 28, 1) -> (m, 784)
    Dense(rng.standard_normal((784, 128)) * np.sqrt(2 / 784), np.zeros(128)),  # (m, 784) -> (m, 128)
    ReLU(),                                                                 # (m, 128)
    Dense(rng.standard_normal((128, 10)) * np.sqrt(2 / 128), np.zeros(10)),    # (m, 128) -> (m, 10)
])

for J_train, J_dev, acc_train, acc_dev in fit(model, SoftmaxCrossEntropy(), step_SGD, X_train, Y_train, X_dev, Y_dev, max_epochs=50, batch_size=64, lr=1e-2, rng=rng): pass