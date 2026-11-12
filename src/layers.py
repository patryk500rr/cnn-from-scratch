import numpy as np


class Dense:
    """Fully connected layer: Z = A @ W + b.

    Shapes:
        W: (n_in, n_out)
        b: (n_out,)
        A: (m, n_in)   input batch, saved in cache for backward
        Z: (m, n_out)  output
    """

    def __init__(self, W, b):
        """Store the parameters and create empty cache and gradient dicts.

        Args:
            W (np.ndarray): weights, shape (n_in, n_out).
            b (np.ndarray): biases, shape (n_out,).
        """
        self.W = W  # (n_in, n_out)
        self.b = b  # (n_out,)
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}  # parameter gradients, filled in backward

    def forward(self, A):
        """Compute Z = A @ W + b and save A for the backward pass.

        Args:
            A (np.ndarray): input batch, shape (m, n_in).

        Returns:
            np.ndarray: Z, shape (m, n_out).
        """
        Z = np.dot(A, self.W) + self.b  # (m, n_in) @ (n_in, n_out) + (n_out,) -> (m, n_out)
        self.cache['A'] = A  # (m, n_in), used later for dW = A.T @ dZ
        return Z

    def backward(self, dZ):
        """Compute parameter gradients and the gradient w.r.t. the input.

        Args:
            dZ (np.ndarray): gradient of the loss w.r.t. Z, shape (m, n_out).

        Returns:
            np.ndarray: dA, shape (m, n_in), passed to the previous layer.
        """
        dW = np.dot(self.cache['A'].T, dZ)  # (n_in, m) @ (m, n_out) -> (n_in, n_out)
        db = np.sum(dZ, axis=0)  # sum over the m examples: (m, n_out) -> (n_out,)
        dA = np.dot(dZ, self.W.T)  # (m, n_out) @ (n_out, n_in) -> (m, n_in)

        assert dW.shape == self.W.shape
        assert db.shape == self.b.shape
        assert dA.shape == self.cache['A'].shape

        self.grads['dW'] = dW
        self.grads['db'] = db

        return dA


class ReLU:
    """Element-wise ReLU activation: A = max(0, Z).

    Shapes:
        Z: (m, n) input, same shape as the output.
    """

    def __init__(self):
        """Create empty cache and gradient dicts (ReLU has no parameters)."""
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}

    def forward(self, Z):
        """Apply ReLU element-wise and save Z for the backward pass.

        Args:
            Z (np.ndarray): input, any shape, e.g. (m, n).

        Returns:
            np.ndarray: A with the same shape as Z.
        """
        self.cache['Z'] = Z  # same shape as Z, used for the mask Z > 0
        return np.maximum(0, Z)

    def backward(self, dA):
        """Pass the gradient only where Z > 0.

        Args:
            dA (np.ndarray): gradient w.r.t. the output A, same shape as Z.

        Returns:
            np.ndarray: dZ, same shape as Z.
        """
        dZ = dA * (self.cache['Z'] > 0).astype('float32')  # mask has the same shape as Z
        self.grads['dZ'] = dZ
        return dZ


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy loss, used as the last layer.

    WARNING: backward() returns the gradient of the softmax COMBINED with
    cross-entropy loss, i.e. dZ = (Y_hat - Y) / m. It is not the gradient of
    softmax alone. It must only be used as the final layer with cross-entropy
    as the loss, and it takes the one-hot labels Y instead of dA.

    Shapes:
        Z:     (m, n_classes) logits
        Y_hat: (m, n_classes) probabilities, each row sums to 1
    """

    def __init__(self):
        """Create empty cache and gradient dicts (softmax has no parameters)."""
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}

    def forward(self, Z, Y):
        """Compute softmax (numerically stable) and the mean cross-entropy loss.

        Args:
            Z (np.ndarray): logits, shape (m, n_classes).
            Y (np.ndarray): one-hot labels, shape (m, n_classes).

        Returns:
            float: loss J, the mean cross-entropy over the m examples.
                   Y_hat is saved in cache for backward.
        """
        Z = Z - np.max(Z, axis=1, keepdims=True)  # (m, n_classes), max per row subtracted for stability
        Y_hat = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)  # (m, n_classes), rows sum to 1
        self.cache['Y_hat'] = Y_hat  # (m, n_classes), used in backward
        self.cache['Y'] = Y

        log_Y_hat = Z - np.max(Z) - np.log(np.sum(np.exp(Z - np.max(Z)), axis=1, keepdims=True))
        loss = -(1/len(Y)) * np.sum(Y * log_Y_hat)
        return loss

    def backward(self, Y):
        """Gradient of the cross-entropy loss w.r.t. Z (see class WARNING).

        Args:
            Y (np.ndarray): one-hot labels, shape (m, n_classes).

        Returns:
            np.ndarray: dZ, shape (m, n_classes).
        """
        dZ = (self.cache['Y_hat'] - self.cache['Y']) / len(self.cache['Y'])  # (m, n_classes) - (m, n_classes) -> (m, n_classes), / m
        self.grads['dZ'] = dZ
        return dZ
    
class Tanh:
    """Element-wise tanh activation: A = tanh(Z).

    Shapes:
        Z: (m, n) input, same shape as the output.
    """

    def __init__(self):
        """Create empty cache and gradient dicts (tanh has no parameters)."""
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}

    def forward(self, Z):
        """Apply tanh element-wise and save A for the backward pass.

        Args:
            Z (np.ndarray): input, any shape, e.g. (m, n).

        Returns:
            np.ndarray: A with the same shape as Z.
        """
        A = np.tanh(Z)
        self.cache['A'] = A
        return A
    
    def backward(self, dA):
        """Multiply the incoming gradient by the derivative 1 - A**2.

        Args:
            dA (np.ndarray): gradient w.r.t. the output A, same shape as Z.

        Returns:
            np.ndarray: dZ, same shape as Z.
        """
        dZ = dA * (1 - self.cache['A']**2)
        self.grads['dZ'] = dZ
        return dZ
    
class Flatten:
    """Reshape each example to a vector: (m, H, W, C) -> (m, H*W*C).

    Has no parameters. Stores the input shape so backward can restore it.
    """

    def __init__(self):
        """Create empty cache and gradient dicts (flatten has no parameters)."""
        self.cache = {}  # stores 'A_shape', the original input shape
        self.grads = {}

    def forward(self, A):
        """Flatten the input while keeping the batch axis.

        Args:
            A (np.ndarray): input, shape (m, ...), e.g. (m, H, W, C).

        Returns:
            np.ndarray: A_flat, shape (m, H*W*C).
        """
        A_flat = A.reshape(A.shape[0], -1)
        self.cache['A_shape'] = A.shape
        return A_flat
    
    def backward(self, A_flat):
        """Restore the gradient to the original input shape.

        Args:
            A_flat (np.ndarray): gradient, shape (m, H*W*C).

        Returns:
            np.ndarray: gradient reshaped to the input shape, e.g. (m, H, W, C).
        """
        dA_flat = A_flat.reshape(self.cache['A_shape'])
        self.grads['dA_flat'] = dA_flat
        return dA_flat
    
class Sigmoid:
    """Element-wise sigmoid activation: A = 1 / (1 + exp(-Z)).

    Shapes:
        Z: (m, n) input, same shape as the output.
    """

    def __init__(self):
        """Create empty cache and gradient dicts (sigmoid has no parameters)."""
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}

    def forward(self, Z):
        """Apply sigmoid element-wise and save A for the backward pass.

        Args:
            Z (np.ndarray): input, any shape, e.g. (m, n).

        Returns:
            np.ndarray: A with the same shape as Z.
        """
        A = 1 / (1 + np.exp(-Z))
        self.cache['A'] = A
        return A

    def backward(self, dA):
        """Multiply the incoming gradient by the derivative A * (1 - A).

        Args:
            dA (np.ndarray): gradient w.r.t. the output A, same shape as Z.

        Returns:
            np.ndarray: dZ, same shape as Z.
        """
        dZ = dA * self.cache['A'] * (1 - self.cache['A'])
        self.grads['dZ'] = dZ
        return dZ

