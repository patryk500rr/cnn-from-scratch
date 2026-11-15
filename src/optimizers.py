"""Parameter update rules: SGD, Momentum, RMSprop and Adam.

All optimizers share the same interface: an object is created once with its
hyperparameters, and step(parameters, lr) is called after every model.backward().
Optimizers with memory (Momentum, RMSprop, Adam) keep their state in dictionaries,
keyed by id(param), so the state of each parameter survives between calls.
"""
from utils import ewa
import numpy as np


class SGD:
    """Plain stochastic gradient descent, without memory between steps."""

    def __init__(self):
        """Create the optimizer. SGD has no hyperparameters and no state."""
        pass

    def step(self, parameters, lr):
        """Apply one gradient descent update in place: param -= lr * grad.

        Call this after model.backward(), so that the gradients are up to date.

        Args:
            parameters (list): (param, grad) pairs, as returned by model.parameters().
                The arrays are updated in place, so the layers see the new values.
            lr (float): learning rate.

        Returns:
            None.
        """
        for param, d_param in parameters:
            param -= lr * d_param


class Momentum:
    """Gradient descent with momentum: the step follows an exponentially weighted average of gradients."""

    def __init__(self, Beta=0.9):
        """Store the decay factor and create an empty state dictionary.

        Args:
            Beta (float): decay factor in (0, 1). Larger values average over more past gradients,
                roughly the last 1 / (1 - Beta) steps. Default 0.9.
        """
        self.Beta = Beta
        self.v = {}

    def step(self, parameters, lr):
        """Update the velocity v and move each parameter against it.

        For every parameter, with g the gradient and v its velocity:
            v = Beta * v + (1 - Beta) * g
            param -= lr * v

        The velocity has the shape of the parameter and starts at zero the first time the
        parameter is seen. No bias correction is applied.

        Args:
            parameters (list): (param, grad) pairs, as returned by model.parameters().
                The arrays are updated in place.
            lr (float): learning rate.

        Returns:
            None.
        """
        for param, grad in parameters:
            key = id(param)

            if key not in self.v:
                self.v[key] = np.zeros_like(param)

            v = self.Beta * self.v[key] + (1 - self.Beta) * grad
            self.v[key] = v
            param -= lr * v


class RMSprop:
    """RMSprop: each parameter's step is divided by the root of its recent mean squared gradient."""

    def __init__(self, Beta2=0.9, eps=1e-8):
        """Store the decay factor and the stabilizing constant, and create an empty state dictionary.

        Args:
            Beta2 (float): decay factor in (0, 1) for the mean squared gradient. Default 0.9.
                Common values are 0.9 to 0.99. Values near 0.999 average over about one epoch.
            eps (float): small constant added outside the square root to avoid division by zero.
        """
        self.Beta2 = Beta2
        self.eps = eps
        self.s = {}

    def step(self, parameters, lr):
        """Update the mean squared gradient s and move each parameter with a scaled step.

        For every parameter, with g the gradient and s its mean squared gradient:
            s = Beta2 * s + (1 - Beta2) * g**2
            param -= lr * g / (sqrt(s) + eps)

        The state has the shape of the parameter and starts at zero. No bias correction is
        applied, so the first steps are smaller than the ones that Adam would take.

        Args:
            parameters (list): (param, grad) pairs, as returned by model.parameters().
                The arrays are updated in place.
            lr (float): learning rate.

        Returns:
            None.
        """
        for param, grad in parameters:
            key = id(param)

            if key not in self.s:
                self.s[key] = np.zeros_like(param)

            s = self.Beta2 * self.s[key] + (1 - self.Beta2) * grad**2
            self.s[key] = s
            param -= lr * grad / (np.sqrt(s) + self.eps)


class Adam:
    """Adam: momentum on the gradients and RMSprop-style scaling, both with bias correction."""

    def __init__(self, Beta1=0.9, Beta2=0.999, eps=1e-8):
        """Store the decay factors and the stabilizing constant, and create empty state dictionaries.

        Args:
            Beta1 (float): decay factor for the gradient average v. Default 0.9.
            Beta2 (float): decay factor for the mean squared gradient s. Default 0.999.
            eps (float): small constant added outside the square root. Default 1e-8.
        """
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.eps = eps
        self.s = {}
        self.v = {}
        self.t = 1

    def step(self, parameters, lr):
        """Update v and s for each parameter, correct their bias, and move the parameter.

        For every parameter, with g the gradient:
            v = Beta1 * v + (1 - Beta1) * g
            s = Beta2 * s + (1 - Beta2) * g**2
            v_hat = v / (1 - Beta1**t)
            s_hat = s / (1 - Beta2**t)
            param -= lr * v_hat / (sqrt(s_hat) + eps)

        The counter t counts calls to step(), not parameters, and is increased once per call
        after all parameters are updated. It is not reset between epochs. Dictionaries v and s
        keep the state of each parameter, keyed by id(param).

        Args:
            parameters (list): (param, grad) pairs, as returned by model.parameters().
                The arrays are updated in place.
            lr (float): learning rate.

        Returns:
            None.
        """
        for param, grad in parameters:
            key = id(param)

            if key not in self.s and key not in self.v:
                self.s[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)

            v = self.Beta1 * self.v[key] + (1 - self.Beta1) * grad
            s = self.Beta2 * self.s[key] + (1 - self.Beta2) * grad**2
            self.v[key] = v
            self.s[key] = s
            v_corrected = v / (1 - self.Beta1**self.t)
            s_corrected = s / (1 - self.Beta2**self.t)

            param -= lr * v_corrected / (np.sqrt(s_corrected) + self.eps)

        self.t += 1

class ExponentialDecay:
    """Learning rate schedule with exponential decay: alpha = alpha0 * gamma**epoch.

    This is not an optimizer. It only computes the learning rate for a given epoch, and
    the training loop passes that value to optimizer.step(parameters, lr).
    """

    def __init__(self, alpha0, gamma=0.95):
        """Store the initial learning rate and the decay factor.

        Args:
            alpha0 (float): learning rate in epoch 0.
            gamma (float): decay factor in (0, 1). Each epoch multiplies the learning rate
                by gamma. Default 0.95.
        """
        self.alpha0 = alpha0
        self.gamma = gamma

    def get_alpha(self, epoch):
        """Return the learning rate for the given epoch.

        Args:
            epoch (int): epoch index counted from 0, the same index as in the loop over epochs
                in fit(). Epoch 0 returns alpha0.

        Returns:
            float: learning rate alpha0 * gamma**epoch.
        """
        alpha = self.alpha0 * self.gamma**epoch
        return alpha
