"""Layers and loss for a numpy neural network.

Each layer implements forward(x) and backward(dout). Layers with parameters
store their gradients in self.grads. Shapes use channels-last (m, ...) batches.
"""
import numpy as np
from init import he, xavier, scaled_normal
import inspect


def _check_init(init, scale):
    """Check that init and scale are compatible.

    scale is needed only for init=scaled_normal, which multiplies standard normal weights by
    a fixed constant. For he and xavier the scale is computed from the layer size, so scale
    must be None.

    Args:
        init (callable): he, xavier or scaled_normal.
        scale (float or None): fixed standard deviation, used only with scaled_normal.

    Raises:
        ValueError: if scale is missing for scaled_normal, or given for another init.
    """
    if init is scaled_normal:
        if scale is None:
            raise ValueError('init=scaled_normal needs a scale, e.g. scale=0.01')
    elif scale is not None:
        raise ValueError('scale is used only with init=scaled_normal')


def _init_weights(shape, rng, init, scale):
    """Sample weights from N(0, 1) and rescale them with the chosen init.

    Args:
        shape (tuple): shape of the weight array, e.g. (n_in, n_out) or (f, f, n_C_prev, n_C).
        rng (np.random.Generator): generator used to draw the standard normal samples.
        init (callable): he, xavier or scaled_normal.
        scale (float or None): fixed standard deviation, used only with scaled_normal.

    Returns:
        np.ndarray: initialized weights with the given shape.
    """
    raw = rng.standard_normal(shape)
    if init is scaled_normal:
        return scaled_normal(raw, scale)
    return init(raw)


class Dense:
    """Fully connected layer: Z = A @ W + b, or Z = A @ W when bias=False.

    Shapes:
        W: (n_in, n_out)
        b: (n_out,), only if bias=True
        A: (m, n_in)   input batch, saved in cache for backward
        Z: (m, n_out)  output
    """

    def __init__(self, n_out, bias=True, init=he, scale=None):
        """Store the parameters and create empty cache and gradient dicts.

        Args:
            n_out (int): number of outputs (neurons). W and b are created later, in build().
            bias (bool): if True (default), the layer has a bias b. Set it to False when the
                layer is followed by batch normalization, which removes the bias anyway.
            init (callable): weight initialization, one of he (default, for ReLU), xavier (for
                tanh or sigmoid) or scaled_normal.
            scale (float or None): standard deviation of the weights. Required only when
                init=scaled_normal, e.g. scale=0.01. Must be None for he and xavier.

        Raises:
            ValueError: if scale is missing for scaled_normal, or given for he or xavier.
        """
        _check_init(init, scale)
        self.cache = {}  # values saved in forward, needed by backward
        self.grads = {}  # parameter gradients, filled in backward
        self.n_out = n_out
        self.bias = bias
        self.init = init
        self.scale = scale

    def forward(self, A):
        """Compute Z = A @ W (+ b if bias=True) and save A for the backward pass.

        Args:
            A (np.ndarray): input batch, shape (m, n_in).

        Returns:
            np.ndarray: Z, shape (m, n_out).
        """
        Z = np.dot(A, self.W)  # (m, n_in) @ (n_in, n_out) -> (m, n_out)
        if self.bias:
            Z = Z + self.b  # (m, n_out) + (n_out,) -> (m, n_out)
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
        dA = np.dot(dZ, self.W.T)  # (m, n_out) @ (n_out, n_in) -> (m, n_in)

        assert dW.shape == self.W.shape
        assert dA.shape == self.cache['A'].shape


        self.grads['dW'] = dW
        if self.bias:
            db = np.sum(dZ, axis=0)  # sum over the m examples: (m, n_out) -> (n_out,)
            assert db.shape == self.b.shape
            self.grads['db'] = db

        return dA
    
    def build(self, input_shape, rng, return_nparams=False):
        """Create W (and b if bias=True) from the input size and the number of outputs.

        Args:
            input_shape (tuple): shape of one example, e.g. (n_in,), without the batch axis.
            rng (np.random.Generator): generator used to initialize W.

        Returns:
            tuple: output shape (n_out,).
        """
        n_in = input_shape[0]
        self.W = _init_weights((n_in, self.n_out), rng, self.init, self.scale)
        if self.bias:
            self.b = np.zeros(self.n_out)

        if return_nparams:
            n_params = n_in * self.n_out + (self.n_out if self.bias else 0)
            return (self.n_out, ), n_params
        else:
            return (self.n_out, )
        
    def name(self):
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
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
        dZ = dA * (self.cache['Z'] > 0)  # boolean mask, same shape as Z
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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
        return 'ReLU'
    
class Dropout:
    """Inverted dropout: randomly zeroes activations during training and rescales the rest.

    During training each element of the input is kept with probability keepprob. The kept
    elements are divided by keepprob, so the expected value of each activation stays the same.
    At test time the layer returns its input unchanged, so no scaling is needed.

    Dropout has no parameters and does not change the shape of the input.
    """

    def __init__(self, keepprob, seed=123):
        """Store the probability of keeping an activation and create an empty cache.

        Args:
            keepprob (float): probability of keeping each activation, in (0, 1].
                1.0 means no dropout. Typical values: about 0.5 for large dense layers,
                0.7 to 0.9 for other hidden layers.
            seed (int): not used by the layer. The random generator is taken from build().
        """
        self.keepprob = keepprob
        self.cache = {}
        self.seed = seed

    def forward(self, A, training:bool):
        """Apply a random mask during training, or return the input unchanged at test time.

        During training the mask d has the same shape as A. Each element is 1 / keepprob with
        probability keepprob and 0 otherwise, so A * d has the same expected value as A. The mask
        is saved in the cache for backward(). The flag training is saved in self.training.

        Args:
            A (np.ndarray): input, any shape, e.g. (m, n) or (m, H, W, C).
            training (bool): True during training, False at test time.

        Returns:
            np.ndarray: A * d during training, A itself at test time. Same shape as A.
        """
        self.training = training
        if training==True:
            rng = self.rng

            d = (rng.random(A.shape) < self.keepprob).astype(A.dtype) / self.keepprob  # same dtype as A
            self.cache['d'] = d
            return A * d
        else:
            return A
        
    def backward(self, dA):
        """Pass the gradient only through the kept activations, scaled by the same mask.

        Valid only after forward(training=True), because it uses the mask from the cache.
        At test time there is no backward pass, so the input gradient is returned unchanged.

        Args:
            dA (np.ndarray): gradient w.r.t. the output of the layer, same shape as the input.

        Returns:
            np.ndarray: gradient w.r.t. the input, dA * d, same shape as the input.
        """
        if self.training == True:
            d = self.cache['d']
            return dA * d
        else: return dA

    def build(self, input_shape, rng, return_nparams=False):
        """Store the random generator used to draw the masks and return the output shape.

        The generator comes from the model, so the masks are reproducible together with the
        rest of the network. Dropout creates no weights.

        Args:
            input_shape (tuple): shape of one example without the batch axis.
            rng (np.random.Generator): generator used to draw the masks.
            return_nparams (bool): if True, also return the number of parameters, which is 0.

        Returns:
            tuple: output shape, equal to input_shape, and optionally the number of parameters.
        """
        self.rng = rng
        if return_nparams:
            return input_shape, 0
        else:
            return input_shape
    def name(self):
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: 'Dropout'.
        """
        return 'Dropout'


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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
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
        self.grads['dA'] = dA
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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
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
                    A_prev_slice = A_prev_pad[i, row * s:(row * s + f), col * s:(col * s + f)] # (f, f, n_C_prev)
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

class MaxPool2d_naive:
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

        Naive version with explicit loops over examples, positions and channels.
        It is the reference for checking the vectorized MaxPool2d.

        Args:
            dA (np.ndarray): gradient w.r.t. A, shape (m, n_H, n_W, n_C).

        Returns:
            np.ndarray: dA_prev, shape (m, n_H_prev, n_W_prev, n_C).
        """
        A_prev = self.cache['A_prev']
        dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)
        s = self.s
        f = self.f

        for i in range(dA.shape[0]):
            for row in range(dA.shape[1]):
                for col in range(dA.shape[2]):
                    for c in range(dA.shape[3]):
                        window = A_prev[i, s*row:s*row + f, s*col:s*col + f, c].copy()
                        mask = np.zeros_like(window)
                        idx_window = np.unravel_index(np.argmax(window), window.shape)
                        mask[idx_window] = dA[i, row, col, c]
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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
        return 'MaxPool2d_naive'

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
        """Take the maximum of each window and cache the position of the maximum for backward().

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_H_prev, n_C).

        Returns:
            np.ndarray: A, shape (m, n_H, n_H, n_C).
        """
        s = self.s
        f = self.f
        windows = np.lib.stride_tricks.sliding_window_view(A_prev, window_shape=(f, f), axis=(1, 2))[:, ::s, ::s]
        A = np.max(windows, axis=(-2, -1))
        flat = windows.reshape(windows.shape[:-2] + (f * f,))
        self.cache['idx'] = np.argmax(flat, axis=-1)  # (m, n_H, n_W, n_C), offset 0..f*f-1 inside the window
        self.cache['A_shape'] = A_prev.shape
        return A

    def backward(self, dA):
        """Send each output gradient back to the position of the window maximum.

        Only the maximum of each window receives a gradient. The cached offsets are used, so no
        windows are built again. Overlapping windows add their contributions with +=.

        Args:
            dA (np.ndarray): gradient w.r.t. A, shape (m, n_H, n_W, n_C).

        Returns:
            np.ndarray: dA_prev, shape (m, n_H_prev, n_W_prev, n_C).
        """
        s = self.s
        f = self.f
        idx = self.cache['idx']
        H_out, W_out = idx.shape[1], idx.shape[2]
        dA_prev = np.zeros(self.cache['A_shape'], dtype=dA.dtype)

        for a in range(f):
            for b in range(f):
                mask = (idx == a * f + b)
                dA_prev[:, a: a + s * H_out: s, b: b + s * W_out: s, :] += dA * mask

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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
        return 'MaxPool2d'

class AvgPool2d_naive:
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
        dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)
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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
        return 'AvgPool2d_naive'
    
class AvgPool2d:
    """Average pooling: each f x f window (moved with stride s) is replaced by its mean.

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
        """Take the mean of each window and save A_prev for the backward pass.

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_W_prev, n_C).

        Returns:
            np.ndarray: A, shape (m, n_H, n_W, n_C), the mean of each f x f window.
        """
        s = self.s
        f = self.f
        windows_stride1 = np.lib.stride_tricks.sliding_window_view(A_prev, window_shape=(f, f), axis=(1, 2))
        windows = windows_stride1[:, ::s, ::s]
        A = np.average(windows, axis=(-2, -1))
        self.cache['A_prev'] = A_prev

        return A
    
    def backward(self, dA):
        """Spread each output gradient evenly over the f * f inputs of its window.

        Every input in a window contributes 1 / (f * f) of the output, so dA is divided by
        f * f and added to all positions of the window. Overlapping windows add with +=.

        Args:
            dA (np.ndarray): gradient w.r.t. A, shape (m, n_H, n_W, n_C).

        Returns:
            np.ndarray: dA_prev, shape (m, n_H_prev, n_W_prev, n_C).
        """
        s = self.s
        f = self.f
        m = dA.shape[0]
        H_out = dA.shape[1]
        n_C_out = dA.shape[-1]
        A_prev = self.cache['A_prev']

        contribution = np.broadcast_to((dA[..., None, None] / f**2), (m, H_out, H_out, n_C_out, f, f))
        
        dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)

        for a in range(f):
            for b in range(f):
                dA_prev[:, a: a + s * H_out : s, b : b + s * H_out : s, :] += contribution[:, :, :, :, a, b]

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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
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
        A = np.average(A_prev, axis=(1, 2))

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
        
        dA_prev = np.broadcast_to(dA[:, None, None, :] / (H*W), A_prev.shape)

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
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
        return 'GlobalAvgPool'

class Conv2d:
    """Convolution layer. The filters and biases are created in build(), not in __init__.

    Assumes square inputs (n_H = n_W), as the vectorized backward uses n_H for both axes.

    Shapes:
        A_prev: (m, n_H_prev, n_W_prev, n_C_prev)   input of this layer
        W:      (f, f, n_C_prev, n_C)                created in build()
        b:      (n_C,)                               created in build(), only if bias=True
        output: (m, n_H, n_W, n_C)                   before any activation
    """
    def __init__(self, n_C, f, s, p, bias=True, init=he, scale=None):
        """Store the hyperparameters. The weights are created later, in build().

        Args:
            n_C (int): number of filters, i.e. channels of the output.
            f (int): side length of the square filter.
            s (int): stride.
            p (int or 'same'): padding, or 'same' to keep the size when stride is 1.
            bias (bool): if True (default), each filter has a bias b. Set it to False when the
                layer is followed by batch normalization, which removes the bias anyway.
            init (callable): weight initialization, one of he (default, for ReLU), xavier (for
                tanh or sigmoid) or scaled_normal. For he and xavier the fan-in is f * f * n_C_prev.
            scale (float or None): standard deviation of the filters. Required only when
                init=scaled_normal, e.g. scale=0.01. Must be None for he and xavier.

        Raises:
            ValueError: if scale is missing for scaled_normal, or given for he or xavier.
        """
        _check_init(init, scale)
        self.s = s
        self.p = p
        self.f = f
        self.n_C = n_C
        self.bias = bias
        self.init = init
        self.scale = scale
        self.cache = {}
        self.grads = {}

    def forward(self, A_prev):
        """Convolve A_prev with the filters and cache the windows for the backward pass.

        The padded input is turned into one matrix of windows, shape (m * n_H * n_W, f * f * n_C_prev).
        This copy is made once here; the matrix is multiplied by the filters reshaped to
        (f * f * n_C_prev, n_C). It is cached and reused by backward(), so the padding and the
        windows are not built twice.

        Args:
            A_prev (np.ndarray): input, shape (m, n_H_prev, n_W_prev, n_C_prev).

        Returns:
            np.ndarray: Z, the output before any activation, shape (m, n_H, n_W, n_C).
        """
        s = self.s
        p = self.p
        n_H_prev = A_prev.shape[1]
        f = self.W.shape[0]
        n_C = self.W.shape[-1]

        if p == 'same':
            p_numenator = (n_H_prev * (s - 1) + f - s)
            assert p_numenator % 2 == 0
            p = p_numenator // 2
        A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)))

        windows = np.lib.stride_tricks.sliding_window_view(A_prev_pad, axis=(1, 2), window_shape=(f, f))[:, ::s, ::s]
        m, n_H, n_W = windows.shape[:3]
        # (m, n_H, n_W, n_C_prev, f, f) -> (m, n_H, n_W, f, f, n_C_prev), then one copy into a matrix
        cols = np.ascontiguousarray(windows.transpose(0, 1, 2, 4, 5, 3)).reshape(m * n_H * n_W, -1)
        W_mat = self.W.reshape(-1, n_C)  # (f * f * n_C_prev, n_C), same order as cols

        Z = (cols @ W_mat).reshape(m, n_H, n_W, n_C)
        if self.bias:
            Z = Z + self.b

        self.cache['cols'] = cols
        self.cache['A_shape'] = A_prev.shape
        self.cache['p'] = p
        self.cache['s'] = s

        return Z

    def backward(self, dZ):
        """Compute the gradients of the filters, the biases and the input.

        dW is computed from the cached matrix of windows, so the input is not padded or windowed
        again. dA is the transposed product of dZ and the filters, followed by a loop over the
        f * f offsets that adds each piece back to its position in the padded input.

        Args:
            dZ (np.ndarray): gradient of the loss w.r.t. this layer's output Z,
                shape (m, n_H, n_W, n_C).

        Returns:
            np.ndarray: dA, gradient w.r.t. the input A_prev, shape (m, n_H_prev, n_W_prev, n_C_prev).
                The padding is removed before returning.
        """
        A_shape = self.cache['A_shape']
        cols = self.cache['cols']
        p = self.cache['p']
        s = self.s
        f = self.W.shape[0]
        m, n_H, n_W, n_C = dZ.shape
        n_C_prev = A_shape[-1]
        n_H_prev, n_W_prev = A_shape[1], A_shape[2]

        dZ_mat = dZ.reshape(m * n_H * n_W, n_C)
        dW = (cols.T @ dZ_mat).reshape(f, f, n_C_prev, n_C)
        self.grads['dW'] = dW
        if self.bias:
            self.grads['db'] = np.sum(dZ, axis=(0, 1, 2))

        dcols = (dZ_mat @ self.W.reshape(-1, n_C).T).reshape(m, n_H, n_W, f, f, n_C_prev)
        dA_pad = np.zeros((m, n_H_prev + 2 * p, n_W_prev + 2 * p, n_C_prev), dtype=dZ.dtype)
        for a in range(f):
            for b in range(f):
                dA_pad[:, a: a + s * n_H: s, b: b + s * n_W: s, :] += dcols[:, :, :, a, b, :]

        if p != 0:
            dA = dA_pad[:, p:-p, p:-p, :]
        else:
            dA = dA_pad
        self.grads['dA'] = dA
        return dA

    def build(self, input_shape, rng, return_nparams=False):
        """Create the filters (and biases, if bias=True) from the input shape and return the output shape.

        Args:
            input_shape (tuple): shape of one example (n_H_prev, n_W_prev, n_C_prev), without the batch axis.
            rng (np.random.Generator): generator used to initialize the filters.

        Returns:
            tuple: output shape (n_H, n_W, n_C) with integer sizes.
        """
        n_H_prev, n_W_prev, n_C_prev = input_shape
        self.W = _init_weights((self.f, self.f, n_C_prev, self.n_C), rng, self.init, self.scale)
        if self.bias:
            self.b = np.zeros(self.n_C)
        p = self.p
        if p == 'same':
            p = (n_H_prev * (self.s - 1) + self.f - self.s) // 2
        n_H = (n_H_prev + 2 * p - self.f) // self.s + 1
        n_W = (n_W_prev + 2 * p - self.f) // self.s + 1
        if return_nparams:
            nparams = self.f * self.f * n_C_prev * self.n_C + (self.n_C if self.bias else 0)
            return (int(n_H), int(n_W), self.n_C), nparams
        else:
            return (int(n_H), int(n_W), self.n_C)
        
    def name(self):
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: class name of the layer.
        """
        return 'Conv2d'
    


        
class BatchNorm2d:
    """Batch normalization: normalizes each neuron or channel over the mini-batch, then rescales it.

    Training (training=True): the mean and variance are computed from the current mini-batch,
    the input is normalized with them, and the running averages miu_run and std2_run are updated.
    Test (training=False): the running averages are used instead, and nothing is updated.

    Output: Z_out = gamma * Z_norm + beta, where Z_norm = (Z - miu) / sqrt(std2 + 1e-8).
    gamma and beta are learned. Both have one value per neuron (Dense) or per channel (conv).

    Shapes:
        Dense: Z (m, n), axes=0, gamma and beta (n,), miu_run and std2_run (n,).
        Conv:  Z (m, H, W, C), axes=(0, 1, 2), gamma and beta (C,), miu_run and std2_run (C,).

    The layer must be built with build() before forward() is called.
    """

    def __init__(self, axes, momentum=0.1):
        """Store the axes over which statistics are computed and the update rate of the running averages.

        Args:
            axes (int or tuple of int): axes that are averaged over. 0 for Dense inputs (m, n),
                (0, 1, 2) for conv inputs (m, H, W, C). An int is converted to a 1-element tuple.
            momentum (float): weight of the new mini-batch statistics in the running averages,
                miu_run = (1 - momentum) * miu_run + momentum * miu. Default 0.1, which corresponds
                to beta = 0.9 in the exponentially weighted average.
        """
        if type(axes) == int:
            self.axes = (axes, )
        else:
            self.axes = tuple(axes)
        self.cache = {}
        self.grads = {}
        self.momentum = momentum


    def forward(self, Z, training:bool):
        """Normalize Z and rescale it with gamma and beta.

        In training mode, statistics come from the current mini-batch and are saved in the cache
        for backward(). The running averages are then updated. In test mode, the running averages
        are used and the layer's state does not change.

        Args:
            Z (np.ndarray): input, shape (m, n) for Dense or (m, H, W, C) for conv.
            training (bool): True during training, False at test time.

        Returns:
            np.ndarray: Z_out with the same shape as Z.
        """
        self.cache['Z'] = Z
        if training == True:
            if len(self.axes) == 1:
                self.N = Z.shape[0]
            else:
                self.N = Z.shape[0] * Z.shape[1] * Z.shape[2] 
            miu = np.mean(Z, axis=self.axes)
            std2 =  np.sum((Z - miu)**2, axis=self.axes) / self.N
            Z_norm = (Z - miu) / np.sqrt(std2 + 1e-8)
            self.cache['miu'] = miu
            self.cache['std2'] = std2
            self.cache['Z_norm'] = Z_norm
            self.miu_run = (1 - self.momentum) * self.miu_run + self.momentum * miu
            self.std2_run = (1 - self.momentum) * self.std2_run + self.momentum * std2

            Z_out = self.gamma * Z_norm + self.beta

            return Z_out
        
        if training == False:
            Z_norm = (Z - self.miu_run) / np.sqrt(self.std2_run + 1e-8)
            self.cache['Z_norm'] = Z_norm
            Z_out = self.gamma * Z_norm + self.beta

            return Z_out
        
    def backward(self, dZ_out):
        """Compute gradients for gamma, beta and the input. Valid only after forward(training=True).

        Gradients of gamma and beta are stored in self.grads['dgamma'] and self.grads['dbeta'].
        The gradient of the mean and variance with respect to the batch is included, because
        both depend on every example in the mini-batch.

        Args:
            dZ_out (np.ndarray): gradient of the loss w.r.t. Z_out, same shape as Z.

        Returns:
            np.ndarray: dZ, gradient w.r.t. the input Z, same shape as Z.
        """
        Z = self.cache['Z']
        miu = self.cache['miu']
        std2 = self.cache['std2']
        Z_norm = self.cache['Z_norm']

        dgamma = np.sum(dZ_out * Z_norm, axis=self.axes)
        dbeta = np.sum(dZ_out, axis=self.axes)
        dZ_norm = dZ_out * self.gamma
        dstd2 = np.sum(dZ_norm * (Z - miu), axis=self.axes) * (-1/2) * (std2 + 1e-8)**(-3/2)
        dmiu = -np.sum(dZ_norm / np.sqrt(std2 + 1e-8), axis=self.axes) + dstd2 * (-2/self.N) * np.sum(Z - miu, axis=self.axes)
        dZ = dZ_norm / np.sqrt(std2 + 1e-8) + dstd2 * 2 * (Z - miu) / self.N + dmiu / self.N

        self.grads['dgamma'] = dgamma
        self.grads['dbeta'] = dbeta
        self.grads['dZ_norm'] = dZ_norm
        self.grads['dstd2'] = dstd2
        self.grads['dmiu'] = dmiu

        return dZ
        
    def build(self, input_shape, rng, return_nparams=False):
        """Create gamma, beta and the running averages, and return the output shape.

        gamma starts at 1, beta at 0 (the identity transform), miu_run at 0 and std2_run at 1.
        The layer has no random weights, so rng is not used. The output shape equals the input shape.

        Args:
            input_shape (tuple): shape of one example without the batch axis, e.g. (n,) for Dense
                or (H, W, C) for conv.
            rng (np.random.Generator): not used, accepted for compatibility with Sequential.build.
            return_nparams (bool): if True, also return the number of parameters, 2 * n_C.

        Returns:
            tuple: output shape (same as input_shape), and optionally the number of parameters.
        """
        if len(input_shape) == 1:
            n_C = input_shape[0]
            self.gamma = np.ones(input_shape, dtype='float64')
            self.beta = np.zeros(input_shape, dtype='float64')

            self.miu_run = np.zeros(n_C, dtype='float64')
            self.std2_run = np.ones(n_C, dtype='float64')
        else:
            n_C = input_shape[-1]
            self.gamma = np.ones(n_C, dtype='float64')
            self.beta = np.zeros(n_C, dtype='float64')

            self.miu_run = np.zeros(n_C, dtype='float64')
            self.std2_run = np.ones(n_C, dtype='float64')

        if return_nparams:
            return input_shape, 2*n_C
        else:
            return input_shape

        
    def name(self):
        """Return the layer name, used as the key in the model summary.

        Returns:
            str: 'BatchNorm2d'.
        """
        return 'BatchNorm2d'
    
class ResidualBlock:
    """Residual block: a sub-network whose output is added to its input (skip connection).

    The block computes S = F(X) + X, where F is the sequence of layers passed to the
    constructor and X is the block input. The skip path is the identity when F(X) has the
    same shape as X. If the number of channels changes (e.g. 4 -> 8), a 1x1 convolution
    (self.projectile) maps X to the new channel count, and the output is F(X) + P(X).

    The block works for convolutional inputs (m, H, W, C) and for Dense inputs (m, n). For
    Dense inputs the projection is a Dense layer (n_in -> n_out) and the main path F can use
    Dense, BatchNorm2d(axes=0), ReLU and Dropout. F may change only the last axis (channels
    or neurons); build() raises ValueError if the height, the width or the number of axes
    changes, because the projection is 1x1 with stride 1. Downsampling should be done
    outside the block.

    Example (Dense):
        block = ResidualBlock([
            Dense(64, bias=False, init=he),
            BatchNorm2d(axes=0),
            ReLU(),
            Dense(64, bias=False, init=he),
            BatchNorm2d(axes=0),
        ])

    Example (conv):
        block = ResidualBlock([
            Conv2d(8, f=3, s=1, p='same', bias=False, init=he),
            BatchNorm2d(axes=(0, 1, 2)),
            ReLU(),
            Conv2d(8, f=3, s=1, p='same', bias=False, init=he),
            BatchNorm2d(axes=(0, 1, 2)),
        ])

    Attributes:
        layers (list): layers of the main path F, in order from input to output. Each must
            implement forward, backward, build and name().
        project (bool): True if the skip path needs a projection (input and output shapes
            differ). Set in build().
        projectile (Conv2d or Dense): projection on the skip path, a 1x1 convolution for conv
            inputs and a Dense layer for (n,) inputs. Exists only if project is True.
    """

    def __init__(self, layers):
        """Store the layers of the main path. Weights are created later, in build().

        Args:
            layers (list): layer objects in order, from input to output.
        """
        self.pairs = {}
        self.layers_W = {}
        self.layers_b = {}
        self.layers_gamma = {}
        self.layers_beta = {}
        self.layers = layers
        # self.layers_names = [layer.__name__ for layer in layers]

    def forward(self, A, training:bool):
        """Compute S = F(X) + X (or F(X) + P(X) if a projection is needed).

        Args:
            A (np.ndarray): block input X, shape (m, H, W, C).
            training (bool): passed to the layers that accept it (BatchNorm2d, Dropout).

        Returns:
            np.ndarray: block output S, shape (m, H, W, C_out), where C_out is the channel
                count of the last layer of the main path.
        """
        identity = A.copy()
        for layer in self.layers:   
            if 'training' in inspect.signature(layer.forward).parameters:
                A = layer.forward(A, training)
            else:
                A = layer.forward(A)
        
        if self.project:
            identity = self.projectile.forward(identity)
        S = A + identity
        return S
    
    def backward(self, dA):
        """Backpropagate through the main path and the skip path, and add the two gradients.

        dX = dF/dX applied to dS, plus dS itself on the identity skip path, or plus
        P.backward(dS) if a projection is used. Must be called after forward(), because
        the layers keep the values they need in their caches.

        Args:
            dA (np.ndarray): gradient w.r.t. the block output S, shape (m, H, W, C_out).

        Returns:
            np.ndarray: gradient w.r.t. the block input X, shape (m, H, W, C).
        """
        dS = dA
        for i, layer in enumerate(reversed(self.layers)):
            dS_prev = layer.backward(dS)
            dS = dS_prev
        if self.project:
            return dS + self.projectile.backward(dA)
        else:
            return dS + dA
    
    def build(self, input_shape, rng, return_nparams=False):
        """Build the layers of the main path in order, then the projection if one is needed.

        Args:
            input_shape (tuple): shape of one example without the batch axis, e.g. (28, 28, 8).
            rng (np.random.Generator): generator used to initialize the weights.
            return_nparams (bool): if True, also return the number of parameters of the block.

        Returns:
            tuple or tuple of (tuple, int): output shape of the block, e.g. (28, 28, 8). If
                return_nparams is True, returns (output_shape, nparams).
        """
        input_shape_l = input_shape
        nparams = 0
        for layer in self.layers:
            if 'rng' in inspect.signature(layer.build).parameters:
                output_shape_l, nparams_layer = layer.build(input_shape_l, rng, return_nparams=True)
                nparams += nparams_layer
            else:
                output_shape_l, nparams_layer = layer.build(input_shape_l, return_nparams=True)
            input_shape_l = output_shape_l
        output_shape = output_shape_l
        if len(input_shape) != len(output_shape) or input_shape[:-1] != output_shape[:-1]:
            raise ValueError(f'ResidualBlock can change only the last axis (channels or neurons), '
                             f'got {input_shape} -> {output_shape}')
        self.project = False
        if input_shape != output_shape:
            self.project = True
            n_out = output_shape[-1]
            if len(input_shape) == 1:  # Dense input (n,): projection is a Dense layer
                self.projectile = Dense(n_out, bias=True, init=he)
            else:  # conv input (H, W, C): projection is a 1x1 convolution
                self.projectile = Conv2d(n_out, f=1, s=1, p=0, bias=True, init=he)
            _, nparams_layer = self.projectile.build(input_shape, rng=rng, return_nparams=True)
            nparams += nparams_layer
        if return_nparams:
            return output_shape, nparams
        else:
            return output_shape
    
    def parameters(self):
        """Collect (parameter, gradient) pairs of the layers in this block.

        Call after backward(), because the gradients are read from the layers at this point.

        Returns:
            dict: maps a name to a (param, grad) tuple. Names are 'layer{j}.W', 'layer{j}.b',
                'layer{j}.gamma' and 'layer{j}.beta', where j is the position of the layer in
                self.layers, and 'projection.W' and 'projection.b' for the skip-path projection
                (only if project is True). Sequential.parameters() adds the prefix 'Block{k}-',
                where k is the number of the block counted from 1.

        Note:
            The L2 penalty does not include these weights yet: Sequential.backward() and
            l2_penalty() only look at layers held directly by Sequential.
        """
        pairs = {}
        for j, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                pairs[f'layer{j}.W'] = (layer.W, layer.grads['dW'])
                if hasattr(layer, 'b'):
                    pairs[f'layer{j}.b'] = (layer.b, layer.grads['db'])
            if hasattr(layer, 'gamma') and hasattr(layer, 'beta'):
                pairs[f'layer{j}.gamma'] = (layer.gamma, layer.grads['dgamma'])
                pairs[f'layer{j}.beta'] = (layer.beta, layer.grads['dbeta'])

        if self.project:
            pairs['projection.W'] = (self.projectile.W, self.projectile.grads['dW'])
            pairs['projection.b'] = (self.projectile.b, self.projectile.grads['db'])
        
        return pairs
    
    def name(self):
        """Return the name of the layer, used in summaries."""
        return 'ResidualBlock'

                
