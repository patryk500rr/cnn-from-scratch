"""Model container: runs a list of layers forward and backward in sequence."""
import inspect
import numpy as np

class Sequential:
    """Ordered container of layers. Runs them forward and backward in sequence.

    The loss layer (SoftmaxCrossEntropy) is NOT part of the list. It is used
    separately:
        Z = model.forward(X)
        J = loss.forward(Z, Y)
        dZ = loss.backward()
        model.backward(dZ)
    """

    def __init__(self, layers, input_shape, rng):
        """Store the layers in order, from input to output.

        Args:
            layers (list): layer objects. Each must implement forward(x) and backward(dx).
        """
        self.layers = layers
        self.input_shape = input_shape
        self.rng = rng

    def forward(self, X):
        """Pass X through all layers in order.

        Args:
            X (np.ndarray): input batch, shape (m, ...).

        Returns:
            np.ndarray: output of the last layer, e.g. logits Z of shape (m, n_classes).
        """
        for layer in self.layers:
            X = layer.forward(X)  # output of one layer is the input of the next
        return X

    def backward(self, dout):
        """Pass the gradient back through all layers in reverse order.

        Args:
            dout (np.ndarray): gradient w.r.t. the output of the last layer, e.g. dZ, shape (m, n_classes).

        Returns:
            np.ndarray: gradient w.r.t. the input of the first layer, e.g. dX, shape (m, ...).
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)  # output gradient of one layer is input gradient of the previous
        return dout

    def parameters(self):
        """Collect (parameter, gradient) pairs from all layers that have weights.

        Call after backward(), because the gradients are filled in there.

        Returns:
            list of tuples: [(W, dW), (b, db), ...] in layer order.
                            Layers without parameters (ReLU, Tanh, Flatten, ...) contribute nothing.
        """
        pairs = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                pairs.append((layer.W, layer.grads['dW']))  # (n_in, n_out), (n_in, n_out)
                pairs.append((layer.b, layer.grads['db']))  # (n_out,), (n_out,)
        return pairs

    
    def build(self):
        input_shape = self.input_shape
        for layer in self.layers:
            if 'rng' in inspect.signature(layer.build).parameters:
                output_shape = layer.build(input_shape, self.rng)
            else:
                output_shape = layer.build(input_shape)

            input_shape = output_shape

    def summary(self):
        params_dict = {}
        shapes_dict = {}
        layers_count = {}
        input_shape = self.input_shape

        for layer in self.layers:
            if 'rng' in inspect.signature(layer.build).parameters:
                output_shape, n_params = layer.build(input_shape, self.rng, return_nparams=True)
            else:
                output_shape, n_params = layer.build(input_shape, return_nparams=True)

            layer_name = layer.name()
            
            if layer_name not in layers_count.keys():
                layers_count[layer_name] = 1
            else:
                layers_count[layer_name] += 1
            
            layer_name = layer_name + ' ' + str(layers_count[layer_name])
            params_dict[layer_name] = n_params
            shapes_dict[layer_name] = [input_shape, output_shape]

            input_shape = output_shape

        params_dict['Total Parameters'] = int(np.sum(list(params_dict.values())))

        return params_dict, shapes_dict