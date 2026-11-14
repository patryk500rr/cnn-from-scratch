"""Model container: runs a list of layers forward and backward in sequence."""
import inspect
import numpy as np
from data import get_minibatch
from evaluation import accuracy
import time
import copy

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
        pairs = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                pairs[f'layer{i} W, dW'] = (layer.W, layer.grads['dW'])  # (n_in, n_out), (n_in, n_out)
                pairs[f'layer{i} b, db'] = ((layer.b, layer.grads['db']))  # (n_out,), (n_out,)
        return pairs

    
    def build(self):
        """Create the weights of every layer, one layer after another.

        Each layer receives the output shape of the previous one, starting from
        self.input_shape (one example, without the batch axis). Layers with weights
        create W and b here, using self.rng. forward() cannot run before build().

        Returns:
            None.
        """
        input_shape = self.input_shape
        for layer in self.layers:
            if 'rng' in inspect.signature(layer.build).parameters:
                output_shape = layer.build(input_shape, self.rng)
            else:
                output_shape = layer.build(input_shape)

            input_shape = output_shape

    def summary(self):
        """Return the parameter count and the shapes of every layer.

        Works on deep copies of the layers and of self.rng, so the weights of this model
        and the random state used by fit() are not changed.

        Returns:
            tuple: (params_dict, shapes_dict).
                params_dict maps 'LayerName n' to the number of parameters of that layer,
                and 'Total Parameters' to the sum over all layers.
                shapes_dict maps the same keys to [input_shape, output_shape] of one example.
        """
        params_dict = {}
        shapes_dict = {}
        layers_count = {}
        input_shape = self.input_shape
        rng_copy = copy.deepcopy(self.rng)

        for layer in self.layers:
            layer_copy = copy.deepcopy(layer)
            if 'rng' in inspect.signature(layer_copy.build).parameters:
                output_shape, n_params = layer_copy.build(input_shape, rng_copy, return_nparams=True)
            else:
                output_shape, n_params = layer_copy.build(input_shape, return_nparams=True)

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

        print(f'Parameters: \n{params_dict}\nShapes: \n{shapes_dict}')
        return params_dict, shapes_dict
    
    def fit(self, loss, X_train, Y_train, X_dev, Y_dev, 
            optimizer, max_epochs, batch_size, lr, 
            return_=[], print_epochs=1, print_time=False):
        """Train the model with mini-batch gradient descent and evaluate after every epoch.

        For each epoch, the training set is shuffled with self.rng. For every mini-batch the
        method runs forward, the loss, backward and one optimizer step. After the epoch, the
        whole training set and the whole dev set are evaluated with evaluate(). Results are
        collected in dictionaries keyed by the epoch index (0, 1, ...).

        Args:
            loss (SoftmaxCrossEntropy): loss layer.
            X_train (np.ndarray): training inputs, shape (m_train, 28, 28, 1).
            Y_train (np.ndarray): one-hot training labels, shape (m_train, 10).
            X_dev (np.ndarray): dev inputs, shape (m_dev, 28, 28, 1).
            Y_dev (np.ndarray): one-hot dev labels, shape (m_dev, 10).
            optimizer (callable): update function, called as optimizer(self.parameters(), lr).
            max_epochs (int): number of epochs to run.
            batch_size (int): mini-batch size.
            lr (float): learning rate passed to the optimizer.
            return_ (list of str): what to return. Allowed items: 'J_history',
                'accuracy_history', 'time', 'all'. Required; passing None raises TypeError.
            print_epochs (int): print the statistics every print_epochs epochs.
            print_time (bool): if True, also print the epoch time and the average time per mini-batch.

        Returns:
            list: one entry for each item of return_, in the same order:
                'J_history'        -> [J_dict_train, J_dict_dev]
                'accuracy_history' -> [acc_dict_train, acc_dict_dev]
                'time'             -> time_dict
                'all'              -> [(J_dict_train, J_dict_dev), (acc_dict_train, acc_dict_dev), time_dict]
            Each dict maps epoch index -> value:
                J_dict_train[i]: mean loss on the whole training set after epoch i
                J_dict_dev[i]:   loss on the whole dev set after epoch i
                acc_dict_train[i], acc_dict_dev[i]: accuracy on the same sets after epoch i
                time_dict: key 'Epoch i, Epoch time [m], Average time per minibatch [s]',
                    value [epoch time in whole minutes, average minibatch time in seconds]
            The function returns only after the last epoch, not during training. The loss is
            not averaged over the mini-batches of the epoch, it is recomputed on the whole set.
        """
        t0 = time.perf_counter()

        if len(return_) != 0:
            for item in return_:
                assert item in ['J_history', 'accuracy_history', 'time', 'all']

        
        J_dict_train = {}
        J_dict_dev = {}
        acc_train_dict = {}
        acc_dev_dict = {}
        time_dict={}
        for i in range(max_epochs):
            t0_epoch = time.perf_counter()
            batch_loader_train = get_minibatch(X_train, Y_train, batch_size, self.rng)
            t_minibatch = 0
            n_batches = 0
            while True:
                try:
                    X_mb_train, Y_mb_train = next(batch_loader_train)
                except StopIteration: break
                t0_minibatch = time.perf_counter()

                Z = self.forward(X_mb_train)
                loss.forward(Z, Y_mb_train)

                dZ = loss.backward()
                self.backward(dZ)

                optimizer(self.parameters(), lr)
                
                t_minibatch += time.perf_counter() - t0_minibatch
                n_batches += 1

            J_list_train, acc_list_train = self.evaluate(loss, X_train, Y_train, batch_size)
            J_dev, acc_dev = self.evaluate(loss, X_dev, Y_dev)

            J_dict_train[i] = np.mean(J_list_train)
            acc_train_dict[i] = np.mean(acc_list_train)

            J_dict_dev[i] = J_dev
            acc_dev_dict[i] = acc_dev
            avg_time_minibatch = round(t_minibatch/n_batches, 4)
            t_epoch = round((time.perf_counter() - t0_epoch) / 60, 3)

            time_dict[f'Epoch {i}, Epoch time [m], Average time per minibatch [s]'] = [t_epoch, avg_time_minibatch]

            if (i % print_epochs) == 0 or (i - 1 == max_epochs):
                print(f'Epoch: {i}, loss: {round(np.mean(J_list_train), 6)}, accuracy train: {round(np.mean(acc_list_train), 3)}, accuracy dev: {round(acc_dev, 3)}')
            if print_time:
                print(f'Time per epoch: {t_epoch} min \nAverage time per minibatch: {avg_time_minibatch} s\n')

        to_return = []
        for item in return_:
            if item == 'J_history':
                to_return.append([J_dict_train, J_dict_dev])
            if item == 'accuracy_history':
                to_return.append([acc_train_dict, acc_dev_dict])
            if item == 'time':
                to_return.append(time_dict)
            if item == 'all':
                to_return.append([(J_dict_train, J_dict_dev), (acc_train_dict, acc_dev_dict), time_dict])

        print(f'Total time elapsed:  {round((time.perf_counter() - t0) / 60)} min')

        return to_return
    
    def evaluate(self, loss, X, Y, batch_size=None):
        """Compute loss and accuracy on a dataset without updating the model.

        Runs forward only: no backward and no parameter changes. The current code has
        two branches that return different types, see the Returns section.

        Args:
            model (Sequential): network with a forward method.
            loss (SoftmaxCrossEntropy): loss layer, used as loss.forward(Z, Y).
            X (np.ndarray): inputs, shape (m, 28, 28, 1).
            Y (np.ndarray): one-hot labels, shape (m, 10).
            batch_size (int or None): if given, the data is processed in mini-batches of this size.
                If None, the whole dataset goes through one forward pass.

        Returns:
            if batch_size is given: tuple (J_list, acc_list), one loss and one accuracy per batch.
            if batch_size is None: tuple (J, acc), a single loss and a single accuracy.
                The intended result is the mean over all examples, not per batch.

        Note:
            Because the forward pass overwrites the layers' cache, do not call evaluate between
            the forward and backward pass of a training batch.
        """
        if batch_size is not None:
            batch_loader = get_minibatch(X, Y, batch_size)
            J_list = []
            acc_list = []
            while True:
                try:
                    X_minibatch, Y_minibatch = next(batch_loader)
                except StopIteration:
                    break

                Z = self.forward(X_minibatch)
                J = loss.forward(Z, Y_minibatch)
                J_list.append(J)

                Y_hat = np.argmax(Z, axis=1)

                acc = accuracy(Y_hat, Y_minibatch.argmax(axis=1))
                acc_list.append(acc)

            return J_list, acc_list
        
        else:
            Z = self.forward(X)
            J = loss.forward(Z, Y)

            Y_hat = np.argmax(Z, axis=1)
            acc = accuracy(Y_hat, Y.argmax(axis=1))

            return J, acc
        
    def predict(self, X):
        """Return the predicted class index for every example.

        Runs forward() only: no backward and no parameter changes. The prediction is the
        index of the largest logit, so softmax is not needed. For large X, split X into
        batches to limit memory use.

        Args:
            X (np.ndarray): inputs, shape (m, 28, 28, 1).

        Returns:
            np.ndarray: predicted class indices, shape (m,), integers from 0 to 9.
        """
        Z = self.forward(X)
        return np.argmax(Z, axis=1)
    
    def save(self, name):
        """Save the weights of all layers that have W and b to a .npz file.

        The file name gets the '.npz' extension added. Each layer is stored under the keys
        'W<i>' and 'b<i>', where i is the position of the layer in self.layers. Only the
        weights are saved, not the architecture, the cache or the gradients.

        Args:
            name (str): file name without the extension, e.g. 'model' creates 'model.npz'.

        Returns:
            None.
        """
        name = name + '.npz'
        dict_ = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                dict_[f'W{i}'] = layer.W
                dict_[f'b{i}'] = layer.b
        np.savez(name, **dict_)

    def load_weights(self, file_name):
        """Load the weights from a .npz file written by save() into the layers of this model.

        IMPORTANT: build() must be called first. The layers only have W and b after build(),
        and the shapes of the loaded arrays are checked against them. Build the model with
        the same architecture and input_shape that was used when the file was saved.

        Args:
            file_name (str): path to the .npz file, including the extension.

        Raises:
            AssertionError: if the shape of a stored array does not match the layer.
            KeyError: if the file does not contain a key for a layer with weights.

        Returns:
            None.
        """
        with np.load(file_name) as data:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'W'):
                    assert (layer.W).shape ==  (data['W' + str(i)]).shape
                    layer.W = data['W' + str(i)]
                    assert (layer.b).shape ==  (data['b' + str(i)]).shape
                    layer.b = data['b' + str(i)]