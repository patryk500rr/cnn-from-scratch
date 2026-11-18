"""Model container: runs a list of layers forward and backward in sequence."""
import inspect
import numpy as np
from data import get_minibatch, augment as augment_batch
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

    def __init__(self, layers, input_shape, rng, l2_lambda=None):
        """Store the layers in order, from input to output.

        Args:
            layers (list): layer objects. Each must implement forward(x), backward(dx),
                build(input_shape, ...) and name().
            input_shape (tuple): shape of one example without the batch axis, e.g. (28, 28, 1).
            rng (np.random.Generator): generator used to initialize the weights in build()
                and to shuffle the training data in fit().
        """
        self.layers = layers
        self.input_shape = input_shape
        self.rng = rng
        self.l2_lambda = l2_lambda

    def forward(self, X, training=False):
        """Pass X through all layers in order.

        The flag training is passed only to layers whose forward accepts it (BatchNorm2d).
        Other layers are called as layer.forward(X).

        Args:
            X (np.ndarray): input batch, shape (m, ...).
            training (bool): True during training (batch statistics in batch norm, saved in cache
                for backward), False at test time (running statistics). Default False, so
                predict() and evaluate() work in test mode without extra arguments.

        Returns:
            np.ndarray: output of the last layer, e.g. logits Z of shape (m, n_classes).
        """
        for layer in self.layers:
            if 'training' in inspect.signature(layer.forward).parameters:
                X = layer.forward(X, training=training)
            else:
                X = layer.forward(X)  # output of one layer is the input of the next
        return X

    def backward(self, dout):
        """Pass the gradient back through all layers in reverse order.

        If l2_lambda is set, the gradient of the L2 penalty is added to dW of every layer with W:
        dW += (l2_lambda / m) * W, where m is the batch size. This matches l2_penalty(), whose
        value is (l2_lambda / (2m)) * sum(W**2), so the gradient and the penalty describe one function.
        Only layers held directly by Sequential get this term. Weights inside a ResidualBlock
        are not regularized yet.

        Args:
            dout (np.ndarray): gradient w.r.t. the output of the last layer, e.g. dZ, shape (m, n_classes).

        Returns:
            np.ndarray: gradient w.r.t. the input of the first layer, e.g. dX, shape (m, ...).
        """
        m = dout.shape[0]
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        if self.l2_lambda is not None:
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    layer.grads['dW'] += (self.l2_lambda / m) * layer.W
                if hasattr(layer, 'layers'):
                    for layer_residual in layer.layers:
                        if hasattr(layer_residual, 'W'):
                            layer_residual.grads['dW'] += (self.l2_lambda / m) * layer_residual.W
                    if layer.project:
                        layer.projectile.grads['dW'] += (self.l2_lambda / m) * layer.projectile.W
        return dout

    def parameters(self):
        """Collect (parameter, gradient) pairs from all layers that have weights.

        Call after backward(), because the gradients are filled in there. Layers that have a
        `layers` attribute (ResidualBlock) return their own parameters(), and their names get
        the prefix 'Block{k}-', where k counts the blocks from 1.

        Returns:
            dict: maps a unique name to a (param, grad) tuple. The names are 'layer{i}.W' and
                'layer{i}.b' for Dense and Conv2d (b only if the layer has a bias), and
                'layer{i}.gamma' and 'layer{i}.beta' for BatchNorm2d, where i is the position
                of the layer in self.layers, and 'Block{k}-<name>' for the parameters inside
                the k-th ResidualBlock (k counts the blocks from 1). Layers without parameters contribute nothing.
                Optimizers iterate over the values, grad_check over the items.
        """
        pairs = {}
        block_count = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'layers'):
                pairs_block = layer.parameters()
                block_count += 1
                for j in pairs_block:
                    pairs[f'Block{block_count}-{j}'] = pairs_block[j]
            if hasattr(layer, 'W'):
                pairs[f'layer{i}.W'] = (layer.W, layer.grads['dW'])  # (n_in, n_out), (n_in, n_out)
                if hasattr(layer, 'b'):  # layers built with bias=False have no b
                    pairs[f'layer{i}.b'] = (layer.b, layer.grads['db'])  # (n_out,), (n_out,)
            if hasattr(layer, 'gamma') and hasattr(layer, 'beta'):  # BatchNorm2d
                pairs[f'layer{i}.gamma'] = (layer.gamma, layer.grads['dgamma'])  # (C,)
                pairs[f'layer{i}.beta'] = (layer.beta, layer.grads['dbeta'])  # (C,)
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

        return params_dict, shapes_dict

    def fit(self, loss, X_train, Y_train, X_dev, Y_dev,
            optimizer, max_epochs, batch_size, lr, lr_scheduler=None,
            augment=False, return_=[], print_epochs=1, print_time=False, seed=123):
        """Train the model with mini-batch gradient descent and evaluate after every epoch.

        For each epoch, the training set is shuffled with self.rng. For every mini-batch the
        method runs forward, the loss, backward and one optimizer step. After the epoch, the
        training set is evaluated in mini-batches of batch_size (exact mean over all examples)
        and the dev set is evaluated in one pass. Results are collected in dictionaries keyed
        by the epoch index (0, 1, ...).

        Args:
            loss (SoftmaxCrossEntropy): loss layer.
            X_train (np.ndarray): training inputs, shape (m_train, 28, 28, 1).
            Y_train (np.ndarray): one-hot training labels, shape (m_train, 10).
            X_dev (np.ndarray): dev inputs, shape (m_dev, 28, 28, 1).
            Y_dev (np.ndarray): one-hot dev labels, shape (m_dev, 10).
            optimizer (object): optimizer with a step(parameters, lr) method, e.g. SGD() or Adam().
                Called as optimizer.step(self.parameters(), lr) after every mini-batch.
            max_epochs (int): number of epochs to run.
            batch_size (int): mini-batch size.
            lr (float): learning rate passed to the optimizer. Used in every epoch when
                lr_scheduler is None. If lr_scheduler is given, lr is alpha0, the learning rate
                of epoch 0, and the schedule gives the learning rate of the later epochs.
            lr_scheduler (object or None): learning rate schedule, e.g. ExponentialDecay from
                optimizers.py. At the start of each epoch fit() calls lr_scheduler.get_alpha(i, lr),
                where i is the epoch index counted from 0, and passes the result to
                optimizer.step() for all mini-batches of that epoch. Default None (constant lr).
            augment (bool): if True, each training mini-batch is randomly shifted by up to 2 pixels
                in each direction before forward(), using augment() from data.py. The background
                value is taken as X_train[0, 0, 0, 0], which is the normalized background of
                normalized data. Each example gets its own shift, drawn from the same rng as the
                shuffling, so runs stay reproducible for a given seed. The loss and accuracy on
                the training set after each epoch are computed on the original data. Dev and test
                sets are never augmented. Default False.
            return_ (list of str): what to return. Allowed items: 'J_history',
                'accuracy_history', 'time', 'all'. Default [] returns an empty list.
                Passing None raises TypeError.
            print_epochs (int): print the statistics every print_epochs epochs and after the last epoch.
            print_time (bool): if True, also print the epoch time and the average time per mini-batch.

        Returns:
            list: one entry for each item of return_, in the same order:
                'J_history'        -> [J_dict_train, J_dict_dev]
                'accuracy_history' -> [acc_dict_train, acc_dict_dev]
                'time'             -> time_dict
                'all'              -> [(J_dict_train, J_dict_dev), (acc_dict_train, acc_dict_dev), time_dict]
            Each dict maps epoch index -> value:
                J_dict_train[i]: mean cross-entropy over the whole training set after epoch i,
                    without the L2 penalty (the penalty is part of the optimized objective only)
                J_dict_dev[i]:   mean loss over the whole dev set after epoch i
                acc_dict_train[i], acc_dict_dev[i]: accuracy on the same sets after epoch i
                time_dict: key 'Epoch i, Epoch time [m], Average time per minibatch [s]',
                    value [epoch time in whole minutes, average minibatch time in seconds]
            The function returns only after the last epoch, not during training.
        """
        t0 = time.perf_counter()

        if not isinstance(augment, bool):
            raise TypeError(f'augment must be True or False, got {type(augment).__name__}')
        rng_data = np.random.default_rng(seed)
        bg_color = X_train[0, 0, 0, 0]  # background value of normalized data, used by augment()

        if len(return_) != 0:
            for item in return_:
                assert item in ['J_history', 'accuracy_history', 'time', 'all']

        J_dict_train = {}
        J_dict_dev = {}
        acc_train_dict = {}
        acc_dev_dict = {}
        time_dict = {}
        lr0 = lr  # alpha0 of the scheduler: it must stay fixed, so the decay does not compound
        for i in range(max_epochs):
            t0_epoch = time.perf_counter()
            batch_loader_train = get_minibatch(X_train, Y_train, batch_size, rng_data)
            t_minibatch = 0
            n_batches = 0

            if lr_scheduler is not None:
                lr = lr_scheduler.get_alpha(i, lr0)

            while True:
                try:
                    X_mb_train, Y_mb_train = next(batch_loader_train)
                except StopIteration:
                    break
                if augment:
                    X_mb_train = augment_batch(X_mb_train, rng_data, bg_color=bg_color, max_shift=2)
                t0_minibatch = time.perf_counter()

                Z = self.forward(X_mb_train, training=True)
                loss.forward(Z, Y_mb_train)  # sets the cache needed by loss.backward()

                dZ = loss.backward()
                self.backward(dZ)

                optimizer.step(self.parameters(), lr)

                t_minibatch += time.perf_counter() - t0_minibatch
                n_batches += 1

            J_train, acc_train = self.evaluate(loss, X_train, Y_train, batch_size)
            J_dev, acc_dev = self.evaluate(loss, X_dev, Y_dev)

            J_dict_train[i] = J_train
            acc_train_dict[i] = acc_train
            J_dict_dev[i] = J_dev
            acc_dev_dict[i] = acc_dev

            avg_time_minibatch = round(t_minibatch / n_batches, 4)
            t_epoch = round((time.perf_counter() - t0_epoch) / 60, 3)

            time_dict[f'Epoch {i}, Epoch time [m], Average time per minibatch [s]'] = [t_epoch, avg_time_minibatch]

            if print_epochs != 0:
                if (i % print_epochs) == 0 or i == max_epochs - 1:
                    print(f'Epoch: {i}, loss train: {round(J_train, 6)}, accuracy train: {round(acc_train, 3)}, accuracy dev: {round(acc_dev, 3)}')
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

        print(f'Total time elapsed:  {round((time.perf_counter() - t0) / 60, 1)} min')

        return to_return

    def evaluate(self, loss, X, Y, batch_size=None):
        """Compute the mean loss and the accuracy on a dataset without updating the model.

        Runs forward only: no backward and no parameter changes.

        Args:
            loss (SoftmaxCrossEntropy): loss layer, used as loss.forward(Z, Y).
            X (np.ndarray): inputs, shape (m, 28, 28, 1).
            Y (np.ndarray): one-hot labels, shape (m, 10).
            batch_size (int or None): if given, the data is processed in mini-batches of this size.
                If None, the whole dataset goes through one forward pass.

        Returns:
            tuple: (J, acc), two floats. J is the mean loss over all m examples and acc is the
                fraction of correct predictions over all m examples. Both are exact, also when
                batch_size does not divide m (the smaller last batch is weighted by its size).

        Note:
            The forward pass overwrites the layers' cache, so do not call evaluate between
            the forward and backward pass of a training batch.
        """
        m = len(X)

        if batch_size is None:
            Z = self.forward(X)
            J = loss.forward(Z, Y)
            acc = np.mean(np.argmax(Z, axis=1) == Y.argmax(axis=1))
            return J, acc

        J_sum = 0.0
        n_correct = 0
        for X_mb, Y_mb in get_minibatch(X, Y, batch_size):
            Z = self.forward(X_mb)
            J_mb = loss.forward(Z, Y_mb)  # mean loss over this batch
            J_sum += J_mb * len(X_mb)  # back to a sum over the batch examples
            n_correct += np.sum(np.argmax(Z, axis=1) == Y_mb.argmax(axis=1))

        return J_sum / m, n_correct / m

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
        """Save the weights of all layers to a .npz file.

        The file name gets the '.npz' extension added. Layers are stored under the keys
        'W<i>' and 'b<i>' (Dense, Conv2d; b only if the layer has a bias), and batch norm
        layers under 'gamma<i>', 'beta<i>', 'miu_run<i>' and 'std2_run<i>', where i is the
        position of the layer in self.layers. The running statistics are needed for test time.
        Layers inside the k-th ResidualBlock (k counts the blocks from 1) use the keys
        'Block<k>-W<j>', 'Block<k>-b<j>', 'Block<k>-gamma<j>', 'Block<k>-beta<j>',
        'Block<k>-miu_run<j>' and 'Block<k>-std2_run<j>', where j is the position of the layer in
        block.layers, and the skip-path projection uses 'Block<k>-projectile-W' and
        'Block<k>-projectile-b'. The architecture, the cache and the gradients are not saved.

        Args:
            name (str): file name without the extension, e.g. 'model' creates 'model.npz'.

        Returns:
            None.
        """
        name = name + '.npz'
        dict_ = {}
        block_count = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                dict_[f'W{i}'] = layer.W
                if hasattr(layer, 'b'):
                    dict_[f'b{i}'] = layer.b
            if hasattr(layer, 'gamma') and hasattr(layer, 'beta'):
                dict_[f'gamma{i}'] = layer.gamma
                dict_[f'beta{i}'] = layer.beta
                dict_[f'miu_run{i}'] = layer.miu_run
                dict_[f'std2_run{i}'] = layer.std2_run
            if hasattr(layer, 'layers'):
                block_count += 1
                for j, layer_residual in enumerate(layer.layers):
                    if hasattr(layer_residual, 'W'):
                        dict_[f'Block{block_count}-W{j}'] = layer_residual.W
                        if hasattr(layer_residual, 'b'):
                            dict_[f'Block{block_count}-b{j}'] = layer_residual.b
                    if hasattr(layer_residual, 'gamma') and hasattr(layer_residual, 'beta'):
                        dict_[f'Block{block_count}-gamma{j}'] = layer_residual.gamma
                        dict_[f'Block{block_count}-beta{j}'] = layer_residual.beta
                        dict_[f'Block{block_count}-miu_run{j}'] = layer_residual.miu_run
                        dict_[f'Block{block_count}-std2_run{j}'] = layer_residual.std2_run
                if layer.project:
                    dict_[f'Block{block_count}-projectile-W'] = layer.projectile.W
                    dict_[f'Block{block_count}-projectile-b'] = layer.projectile.b


        np.savez(name, **dict_)

    def load_weights(self, file_name):
        """Load the weights from a .npz file written by save() into the layers of this model.

        IMPORTANT: build() must be called first. The layers only have their arrays after build(),
        and the shapes of the loaded arrays are checked against them. Build the model with
        the same architecture, input_shape and bias settings that were used when the file was saved.
        Batch norm layers also get their gamma, beta and running statistics (miu_run, std2_run),
        which are needed for test-time predictions.

        Args:
            file_name (str): path to the .npz file, including the extension.

        Raises:
            AssertionError: if the shape of a stored array does not match the layer.
            KeyError: if the file does not contain a key for a layer with weights.

        Returns:
            None.
        """
        with np.load(file_name) as data:
            block_count = 0
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'W'):
                    assert (layer.W).shape == (data['W' + str(i)]).shape
                    layer.W = data['W' + str(i)]
                    if hasattr(layer, 'b'):
                        assert (layer.b).shape == (data['b' + str(i)]).shape
                        layer.b = data['b' + str(i)]
                if hasattr(layer, 'gamma') and hasattr(layer, 'beta'):
                    assert (layer.gamma).shape == (data['gamma' + str(i)]).shape
                    layer.gamma = data['gamma' + str(i)]
                    assert (layer.beta).shape == (data['beta' + str(i)]).shape
                    layer.beta = data['beta' + str(i)]
                    layer.miu_run = data['miu_run' + str(i)]
                    layer.std2_run = data['std2_run' + str(i)]
                if hasattr(layer, 'layers'):
                    block_count += 1
                    for j, layer_residual in enumerate(layer.layers):
                        if hasattr(layer_residual, 'W'):
                            layer_residual.W = data[f'Block{block_count}-W{j}']
                            if hasattr(layer_residual, 'b'):
                                layer_residual.b = data[f'Block{block_count}-b{j}']
                        if hasattr(layer_residual, 'gamma') and hasattr(layer_residual, 'beta'):
                            layer_residual.gamma = data[f'Block{block_count}-gamma{j}']
                            layer_residual.beta = data[f'Block{block_count}-beta{j}']
                            layer_residual.miu_run = data[f'Block{block_count}-miu_run{j}']
                            layer_residual.std2_run = data[f'Block{block_count}-std2_run{j}']
                    if layer.project:
                        layer.projectile.W = data[f'Block{block_count}-projectile-W']
                        layer.projectile.b = data[f'Block{block_count}-projectile-b']

    def l2_penalty(self, loss, m):
        """Add the L2 penalty to a mean loss: loss + (l2_lambda / (2m)) * sum of W**2 over all layers with W.

        Only the weight matrices W are penalized, not biases, gamma or beta. The penalty is
        scaled by the same m as the mean cross-entropy, so the gradient added in backward()
        (l2_lambda / m) * W is its exact derivative.

        Args:
            loss (float): mean cross-entropy of the batch.
            m (int): number of examples in that batch.

        Returns:
            float: loss with the penalty added.

        Raises:
            ValueError: if l2_lambda was not set when the model was created.
        """
        if self.l2_lambda is None:
            raise ValueError('L2 lambda not defined')
        
        squared_sum = sum([np.sum(layer.W**2) for layer in self.layers if hasattr(layer, 'W')])
        for layer in self.layers:
            if hasattr(layer, 'layers'):
                squared_sum += sum([np.sum(layer_res.W**2) for layer_res in layer.layers if hasattr(layer_res, 'W')])
                if layer.project:
                    squared_sum += np.sum(layer.projectile.W**2)

        return loss + (self.l2_lambda/(2*m)) * squared_sum
