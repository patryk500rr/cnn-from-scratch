"""Layers and loss for a numpy neural network.

Each layer implements forward(x) and backward(dout). Layers with parameters
store their gradients in self.grads. Shapes use channels-last (m, ...) batches.
"""
import numpy as np


class Dense:
    """Fully connected layer: Z = A @ W + b.

    Shapes:
        W: (n_in, n_out)
        b: (n_out,)
        A: (m, n_in)   input batch, saved in cache for backward
        Z: (m, n_out)  output
    """

    def __init__(self, n_out):
        """Store the parameters and create empty cache and gradient dicts.

        Args:
            W (np.ndarray): weights, shape (n_in, n_out).
            b (np.ndarray): biases, shape (n_out,).
        """
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}  # parameter gradients, filled in backward
        self.n_out = n_out

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
    
    def build(self, input_shape, rng, return_nparams=False):
        """Create W and b from the input size and the number of outputs.

        Args:
            input_shape (tuple): shape of one example, e.g. (n_in,), without the batch axis.
            rng (np.random.Generator): generator used to initialize W.

        Returns:
            tuple: output shape (n_out,).
        """
        n_in = input_shape[0]
        self.W = rng.standard_normal((n_in, self.n_out)) * 0.1
        self.b = np.zeros(self.n_out)

        if return_nparams:
            n_params = n_in * self.n_out + self.n_out
            return (self.n_out, ), n_params
        else:
            return (self.n_out, )
        
    def name(self):
        return 'Dense'
    
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
    
    def build(self, input_shape, return_nparams=False):
        """Return the output shape. The activation is element-wise and has no parameters.

        Args:
            input_shape (tuple): shape of one example, without the batch axis.

        Returns:
            tuple: output shape, the same as input_shape.
        """
        if return_nparams:
            return input_shape, 0
        else:
            return input_shape
        
    def name(self):
        return 'ReLU'

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

        log_Y_hat = Z - np.log(np.sum(np.exp(Z), axis=1, keepdims=True))
        loss = -(1/len(Y)) * np.sum(Y * log_Y_hat)
        return loss

    def backward(self):
        """Gradient of the cross-entropy loss w.r.t. Z (see class WARNING).

        Args:
            None

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
    
    def build(self, input_shape, return_nparams=False):
        """Return the output shape. The activation is element-wise and has no parameters.

        Args:
            input_shape (tuple): shape of one example, without the batch axis.

        Returns:
            tuple: output shape, the same as input_shape.
        """
        if return_nparams:
            return input_shape, 0
        else:
            return input_shape

    def name(self):
        return 'Tanh'
        
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
        dA = A_flat.reshape(self.cache['A_shape'])
        self.grads['dA_flat'] = dA
        return dA

    def build(self, input_shape, return_nparams=False):
        """Return the output shape: all dimensions of one example multiplied together.

        Args:
            input_shape (tuple): shape of one example, e.g. (H, W, C), without the batch axis.

        Returns:
            tuple: output shape (H * W * C,).
        """
        if return_nparams:
            return (int(np.prod(input_shape)),), 0
        else:
            return (int(np.prod(input_shape)),)
        
    def name(self):
        return 'Flatten'
    
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
    def build(self, input_shape, return_nparams=False):
        """Return the output shape. The activation is element-wise and has no parameters.

        Args:
            input_shape (tuple): shape of one example, without the batch axis.

        Returns:
            tuple: output shape, the same as input_shape.
        """
        if return_nparams:
            return input_shape, 0
        else:
            return input_shape
        
    def name(self):
        return 'Sigmoid'

class conv_naive:
    """Convolution layer computed with explicit Python loops (reference implementation).

    Slow, but easy to check by hand. Used for gradient checks and as the baseline
    for the vectorized version.

    Shapes:
        A_prev: (m, n_H_prev, n_H_prev, n_C_prev)
        W:      (f, f, n_C_prev, n_C)
        b:      (n_C,)
        Z:      (m, n_H, n_H, n_C), with n_H = (n_H_prev + 2p - f) / s + 1
    """

    def __init__(self, W, b, s, p):
        """Store the filters, the bias, the stride and the padding.

        Args:
            W (np.ndarray): filters, shape (f, f, n_C_prev, n_C).
            b (np.ndarray): biases, shape (n_C,).
            s (int): stride.
            p (int or 'same'): padding. 'same' is converted to an integer when forward runs.
        """
        self.W = W # (f, f, n_C_prev, n_C)
        self.b = b # (n_C, )
        self.cache = {}
        self.grads = {}
        self.s = s
        self.p = p

    def forward(self, A_prev):
        """Convolve every example with every filter and save what backward needs.

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_H_prev, n_C_prev).

        Returns:
            np.ndarray: Z, shape (m, n_H, n_H, n_C).

        Raises:
            AssertionError: if (n_H_prev + 2p - f) is not divisible by the stride.
        """
        n_H_prev = A_prev.shape[1]
        f = self.W.shape[0]
        s = self.s
        p = self.p
        if p == 'same':
            p_numenator = (n_H_prev * (s - 1) + f - s)
            assert p_numenator % 2 == 0
            p = p_numenator // 2

        m = A_prev.shape[0]
        numerator = A_prev.shape[1] + 2 * p - self.W.shape[0]
        denominator = s
        assert numerator % denominator == 0
        n_H = int(numerator / denominator + 1)
        self.cache['A_prev'] = A_prev # (m, n_H_prev, n_H_prev, n_C_prev)
        self.cache['s'] = s
        self.cache['p'] = p

        A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0))) # (m, n_H_prev + 2p, n_H_prev + 2p, n_C_prev)

        Z = np.zeros((m, n_H, n_H, self.W.shape[-1]))

        for i in range(m):
            for row in range(n_H):
                for col in range(n_H):
                    A_prev_slice = A_prev_pad[i, row * s:(row * s + f), col * s:(col * s + f)] # (1, f, f, n_C_prev)
                    for c in range(self.W.shape[-1]):
                        Z_slice = np.sum(A_prev_slice * self.W[:, :, :, c]) + self.b[c]
                        Z[i, row, col, c] = Z_slice

        return Z # (m, n_H, n_H, n_C)
    
    def backward(self, dZ): # dZ (m, n_H, n_H, n_C_)
        """Compute the gradients of the filters, the biases and the input.

        Gradients for W and b are summed over all positions and examples, and the
        gradient for the input is accumulated over all windows that contain a pixel.
        The padding is removed at the end, so dA has the shape of A_prev.

        Args:
            dZ (np.ndarray): gradient w.r.t. Z, shape (m, n_H, n_H, n_C).

        Returns:
            np.ndarray: dA, shape (m, n_H_prev, n_H_prev, n_C_prev).
                dW and db are stored in self.grads.
        """
        p = self.cache['p']
        s = self.cache['s']
        A_pad = np.pad(self.cache['A_prev'], ((0, 0), (p, p), (p, p), (0, 0)))

        dW = np.zeros_like(self.W, dtype='float64') # (f, f, n_C_prev, n_C)
        db = np.zeros_like(self.b, dtype='float64') # (n_C)
        dA_pad = np.pad(np.zeros_like(self.cache['A_prev'], dtype='float64'), ((0, 0), (p, p), (p, p), (0, 0)))


        m = dZ.shape[0]
        f = dW.shape[0]

        for i in range(m):
            for row in range(dZ.shape[1]):
                for col in range(dZ.shape[2]):
                    A_pad_slice = A_pad[i, row * s:(row * s + f), col * s:(col * s + f), :]
                    for c in range(dZ.shape[-1]):
                        dW[:, :, :, c] += A_pad_slice * dZ[i, row, col, c]
                        db[c] += dZ[i, row, col, c]
                        dA_pad[i, row * s:(row * s + f), col * s:(col * s + f), :] += self.W[:, :, :, c] * dZ[i, row, col, c]
     
        if p == 0:
            dA = dA_pad
        else:
            dA = dA_pad[:, p:-p, p:-p, :]
        self.grads['dW'] = dW
        self.grads['db'] = db
        self.grads['dA'] = dA
        return dA    

class MaxPool2d:
    """Max pooling: each f x f window (moved with stride s) is replaced by its maximum.

    Shapes:
        A_prev: (m, n_H_prev, n_H_prev, n_C)
        A:      (m, n_H, n_H, n_C), with n_H = floor((n_H_prev - f) / s) + 1

    Has no parameters. Each channel is pooled separately.
    """

    def __init__(self, f=2, s=2):
        """Store the window size and stride.

        Args:
            f (int): side length of the pooling window.
            s (int): stride, the step between neighbouring windows.
        """
        self.f = f
        self.s = s
        self.cache = {}
        self.grads = {}

    def forward(self, A_prev):
        """Take the maximum of each window and save A_prev for the backward pass.

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_H_prev, n_C).

        Returns:
            np.ndarray: A, shape (m, n_H, n_H, n_C).
        """
        m = A_prev.shape[0]
        n_H_prev = A_prev.shape[1]
        n_C = A_prev.shape[3]

        n_H = int(np.floor((n_H_prev - self.f) / self.s) + 1)
        A = np.zeros((m, n_H, n_H, n_C))

        for i in range(m):
            for row in range(n_H):
                for col in range(n_H):
                    for c in range(n_C):
                        A_prev_slice = A_prev[i, row * self.s:row * self.s + self.f, col * self.s: col * self.s + self.f, c]
                        A[i, row, col, c] = np.max(A_prev_slice)

        self.cache['A_prev'] = A_prev
        return A

    def backward(self, dA):
        """Send each output gradient back to the position of the window maximum.

        Work in progress: the body is not written yet.

        Args:
            dA (np.ndarray): gradient w.r.t. A, shape (m, n_H, n_H, n_C).

        Returns:
            np.ndarray: dA_prev, shape (m, n_H_prev, n_H_prev, n_C).
        """
        A_prev = self.cache['A_prev']
        dA_prev = np.zeros_like(A_prev).astype('float64')
        s = self.s
        f = self.f

        for i in range(dA.shape[0]):
            for row in range(dA.shape[1]):
                for col in range(dA.shape[2]):
                    for c in range(dA.shape[3]):
                        slice = A_prev[i, s*row:s*row + f, s*col:s*col + f, c].copy()
                        mask = np.zeros_like(slice)
                        idx_slice = np.unravel_index(np.argmax(slice), slice.shape)
                        mask[idx_slice] = dA[i, row, col, c]
                        dA_prev[i, s*row:s*row + f, s*col:s*col + f, c] += mask

        self.grads['dA_prev'] = dA_prev
        return dA_prev
    
    def build(self, input_shape, return_nparams=False):
        """Return the output shape of the pooling. No parameters are created.

        Args:
            input_shape (tuple): shape of one example (n_H_prev, n_W_prev, n_C), without the batch axis.

        Returns:
            tuple: output shape (n_H, n_W, n_C) with integer sizes.
        """
        n_H_prev, n_W_prev, n_C = input_shape
        n_H = (n_H_prev - self.f) // self.s + 1
        n_W = (n_W_prev - self.f) // self.s + 1
        if return_nparams:
            return (int(n_H), int(n_W), int(n_C)), 0
        else:
            return (int(n_H), int(n_W), int(n_C))
        
    def name(self):
        return 'MaxPool2d'

class AvgPool2d:
    """Average pooling: each f x f window (moved with stride s) is replaced by its mean.

    Shapes:
        A_prev: (m, n_H_prev, n_H_prev, n_C)
        A:      (m, n_H, n_H, n_C), with n_H = floor((n_H_prev - f) / s) + 1

    Has no parameters. Each channel is pooled separately. Every input in a window
    contributes equally, with weight 1 / (f * f).
    """

    def __init__(self, f=2, s=2):
        """Store the window size and stride.

        Args:
            f (int): side length of the pooling window.
            s (int): stride, the step between neighbouring windows.
        """
        self.f = f
        self.s = s
        self.cache = {}
        self.grads = {}

    def forward(self, A_prev):
        """Take the mean of each window and save A_prev for the backward pass.

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_H_prev, n_C).

        Returns:
            np.ndarray: A, shape (m, n_H, n_H, n_C), the mean of each f x f window.
        """
        m = A_prev.shape[0]
        n_H_prev = A_prev.shape[1]
        n_C = A_prev.shape[3]

        n_H = int(np.floor((n_H_prev - self.f) / self.s) + 1)
        A = np.zeros((m, n_H, n_H, n_C))

        for i in range(m):
            for row in range(n_H):
                for col in range(n_H):
                    for c in range(n_C):
                        A_prev_slice = A_prev[i, row * self.s:row * self.s + self.f, col * self.s: col * self.s + self.f, c]
                        A[i, row, col, c] = np.average(A_prev_slice)

        self.cache['A_prev'] = A_prev
        return A

    def backward(self, dA):
        """Spread each output gradient evenly over the f * f inputs of its window.

        Each input in a window contributes 1 / (f * f) of the output, so the gradient
        dA[i, row, col, c] is divided by f * f and added to every position of its window.
        Overlapping windows accumulate with +=.

        Args:
            dA (np.ndarray): gradient w.r.t. A, shape (m, n_H, n_H, n_C).

        Returns:
            np.ndarray: dA_prev, shape (m, n_H_prev, n_H_prev, n_C).
        """
        A_prev = self.cache['A_prev']
        dA_prev = np.zeros_like(A_prev).astype('float64')
        s = self.s
        f = self.f

        for i in range(dA.shape[0]):
            for row in range(dA.shape[1]):
                for col in range(dA.shape[2]):
                    for c in range(dA.shape[3]):
                        dA_prev[i, s*row:s*row + f, s*col:s*col + f, c] += dA[i, row, col, c] / f**2

        self.grads['dA_prev'] = dA_prev
        return dA_prev
    
    def build(self, input_shape, return_nparams=False):
        """Return the output shape of the pooling. No parameters are created.

        Args:
            input_shape (tuple): shape of one example (n_H_prev, n_W_prev, n_C), without the batch axis.

        Returns:
            tuple: output shape (n_H, n_W, n_C) with integer sizes.
        """
        n_H_prev, n_W_prev, n_C = input_shape
        n_H = (n_H_prev - self.f) // self.s + 1
        n_W = (n_W_prev - self.f) // self.s + 1
        if return_nparams:
            return (int(n_H), int(n_W), int(n_C)), 0
        else:
            return (int(n_H), int(n_W), int(n_C))
        
    def name(self):
        return 'AvgPool2d'
    
class GlobalAvgPool:
    """Global average pooling: the mean of each whole feature map, one number per channel.

    Shapes:
        A_prev: (m, H, W, n_C)
        A:      (m, n_C)

    Has no parameters. Replaces Flatten followed by a large Dense layer with a much
    smaller classifier input, and does not depend on where in the map a feature is.
    """

    def __init__(self):
        """Create empty cache and gradient dicts (global pooling has no parameters)."""
        self.cache = {}
        self.grads = {}

    def forward(self, A_prev):
        """Average each feature map over all its positions and save A_prev for backward.

        Args:
            A_prev (np.ndarray): input, shape (m, H, W, n_C).

        Returns:
            np.ndarray: A, shape (m, n_C).
        """
        m = A_prev.shape[0]
        n_C = A_prev.shape[-1]
        A = np.zeros((m, n_C))
        
        for i in range(m):
            for c in range(n_C):
                A[i, c] = np.average(A_prev[i, :, :, c])

        self.cache['A_prev'] = A_prev
        return A 
    
    def backward(self, dA):
        """Spread each gradient evenly over all H * W positions of its feature map.

        Each position contributes 1 / (H * W) to the mean, so dA[i, c] is divided by
        H * W and copied to every position of map (i, c).

        Args:
            dA (np.ndarray): gradient w.r.t. A, shape (m, n_C).

        Returns:
            np.ndarray: dA_prev, shape (m, H, W, n_C).
        """
        A_prev = self.cache['A_prev']
        H = A_prev.shape[1]
        W = A_prev.shape[2]
        m = A_prev.shape[0]
        n_C = A_prev.shape[-1]

        dA_prev = np.zeros_like(A_prev).astype('float64')

        for i in range(m):
            for c in range(n_C):
                dA_prev[i, :, :, c] = dA[i, c] / (H * W)

        self.grads['dA_prev'] = dA_prev
        return dA_prev
            
    def build(self, input_shape, return_nparams=False):
        """Return the output shape: one number per channel.

        Args:
            input_shape (tuple): shape of one example (H, W, n_C), without the batch axis.

        Returns:
            tuple: output shape (n_C,).
        """
        if return_nparams:
            return (input_shape[-1],), 0
        else:
            return (input_shape[-1],)
        
    def name(self):
        return 'GlobalAvgPool'

class Conv2d:
    """Convolution layer. The filters and biases are created in build(), not in __init__.

    Shapes:
        A_prev: (m, n_H_prev, n_W_prev, n_C_prev)   input of this layer
        W:      (f, f, n_C_prev, n_C)                created in build()
        b:      (n_C,)                               created in build()
        output: (m, n_H, n_W, n_C)                   before any activation
    """
    def __init__(self, n_C, f, s, p):
        """Store the hyperparameters. The weights are created later, in build().

        Args:
            n_C (int): number of filters, i.e. channels of the output.
            f (int): side length of the square filter.
            s (int): stride.
            p (int or 'same'): padding, or 'same' to keep the size when stride is 1.
        """
        self.s = s
        self.p = p
        self.f = f
        self.n_C = n_C
        self.cache = {}
        self.grads = {}

    def forward(self, A_prev):
        """Convolve A_prev with the filters and save A_prev for the backward pass.

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_W_prev, n_C_prev).

        Returns:
            np.ndarray: output (called A in the code), shape (m, n_H, n_W, n_C), before any activation.
        """
        s = self.s
        p = self.p
        n_H_prev = A_prev.shape[1]
        f = self.W.shape[0]

        if p == 'same':
            p_numenator = (n_H_prev * (s - 1) + f - s)
            assert p_numenator % 2 == 0
            p = p_numenator // 2
        A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)))

        windows_stride1 = np.lib.stride_tricks.sliding_window_view(A_prev_pad, axis=(1, 2), window_shape=(f, f))
        windows = windows_stride1[:, ::s, ::s]

        Z = np.tensordot(windows, self.W, axes=([3, 4, 5], [2, 0, 1])) + self.b

        self.cache['A_prev'] = A_prev
        self.cache['p'] = p
        self.cache['s'] = s

        return Z
    
    def backward(self, dZ): 
        """
            W (f, f, n_C_prev, n_C)
            b (n_C, )
            A_prev (m, n_H_prev, n_H_prev, n_C_prev)
            Z (m, n_H, n_H, n_C)
            dZ (m, n_H, n_H, n_C)
            windows (m, n_H, n_H, n_C_prev, f, f)
        """
        A_prev = self.cache['A_prev'] 
        n_H_prev = A_prev.shape[1]
        n_C_prev = A_prev.shape[-1]
        n_H = dZ.shape[1]
        m = dZ.shape[0]
        p = self.cache['p']
        s = self.s
        f = self.W.shape[0]
        A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0))) 


        windows_stride1 = np.lib.stride_tricks.sliding_window_view(A_prev_pad, axis=(1, 2), window_shape=(f, f))
        windows = windows_stride1[:, ::s, ::s]

        dW = np.tensordot(windows, dZ, ([0, 1, 2], [0, 1, 2])).transpose(1, 2, 0, 3)
        db = np.sum(dZ, axis=(0, 1, 2))

        dA_pad_windows = np.tensordot(self.W, dZ, ([3], [3])) # (f, f, n_C, m, n_H_prev, n_H_prev)
        dA_pad = np.zeros((m, n_H_prev + 2*p, n_H_prev + 2*p, n_C_prev))
        for i in range(m):
            for row in range(n_H):
                for col in range(n_H):
                    contribution = dA_pad_windows[:, :, :, i, row, col]
                    dA_pad[i, s*row:s*row + f, s*col:s*col + f, :] += contribution

        self.grads['dW'] = dW
        self.grads['db'] = db

        if p != 0:
            dA = dA_pad[:, p:-p, p:-p]
        else:
            dA = dA_pad
        self.grads['dA'] = dA

        return dA
    
    def build(self, input_shape, rng, return_nparams=False):
        """Create the filters and biases from the input shape and return the output shape.

        Args:
            input_shape (tuple): shape of one example (n_H_prev, n_W_prev, n_C_prev), without the batch axis.
            rng (np.random.Generator): generator used to initialize the filters.

        Returns:
            tuple: output shape (n_H, n_W, n_C) with integer sizes.
        """
        n_H_prev, n_W_prev, n_C_prev = input_shape
        self.W = rng.standard_normal((self.f, self.f, n_C_prev, self.n_C))
        self.b = np.zeros(self.n_C)
        p = self.p
        if p == 'same':
            p = (n_H_prev * (self.s - 1) + self.f - self.s) // 2
        n_H = (n_H_prev + 2 * p - self.f) // self.s + 1
        n_W = (n_W_prev + 2 * p - self.f) // self.s + 1
        if return_nparams:
            nparams = self.f * self.f * n_C_prev * self.n_C + self.n_C
            return (int(n_H), int(n_W), self.n_C), nparams
        else:
            return (int(n_H), int(n_W), self.n_C)
        
    def name(self):
        return 'Conv2d'

        

        
        
