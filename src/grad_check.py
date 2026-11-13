"""Numerical gradient check: compares analytic gradients from backward()
with centered finite differences (J(theta + eps) - J(theta - eps)) / (2 * eps).
"""
import numpy as np



def compute_loss(model, loss, X, Y):
    """Run a forward pass and return the scalar loss, without any backward pass.

    Args:
        model (Sequential): network to evaluate.
        loss (SoftmaxCrossEntropy): loss layer.
        X (np.ndarray): inputs, shape (m, ...).
        Y (np.ndarray): one-hot labels, shape (m, n_classes).

    Returns:
        float: mean cross-entropy loss J.
    """
    Z = model.forward(X)
    J = loss.forward(Z, Y)
    return J

def grad_check(model, loss, X, Y, eps, n_samples, rng):
    """Compare analytic and numerical gradients for every parameter of the model.

    For each parameter, up to n_samples randomly chosen entries are checked.
    Each entry is perturbed one at a time by +eps and -eps, and the parameter is
    restored afterwards. Use float64 and small inputs, and disable dropout.

    Args:
        model (Sequential): network, must implement forward, backward and parameters().
        loss (SoftmaxCrossEntropy): loss layer.
        X (np.ndarray): inputs, float64, shape (m, ...).
        Y (np.ndarray): one-hot labels, float64, shape (m, n_classes).
        eps (float): perturbation size, e.g. 1e-7.
        n_samples (int): maximum number of entries to check per parameter.
        rng (np.random.Generator): generator used to choose which entries to check.

    Returns:
        dict: maps a description such as 'W, layer: 0, shape: (5, 6)' to the
              relative error. Expected values are below 1e-7.
              Values near 1e-5 are suspicious, above 1e-3 usually mean a bug.
    """

    compute_loss(model, loss, X, Y)
    dZ = loss.backward()
    model.backward(dZ)
    params = model.parameters()


    err_dict = {}

    for i, (param, grad_analytic) in enumerate(params):
        grad_num = np.zeros_like(param)
        
        k = rng.choice(param.size, size=min(n_samples, param.size), replace=False)
        idx = np.unravel_index(k, param.shape)

        for coord in zip(*idx):
            orig = param[coord]
            param[coord] = orig + eps
            J_plus = compute_loss(model, loss, X, Y)

            param[coord] = orig - eps
            J_minus = compute_loss(model, loss, X, Y)

            grad_num[coord] = (J_plus - J_minus) / (2*eps)
            param[coord] = orig


        grad = grad_analytic.copy()[idx]

        denominator = (np.linalg.norm(grad) + np.linalg.norm(grad_num[idx]))
        if denominator == 0:
            err = 0
        else:
            err = np.linalg.norm(grad - grad_num[idx]) / denominator

        if param.ndim == 2:
            param_type = 'W'
        elif param.ndim == 1:
            param_type = 'b'
        elif param.ndim == 4:
            param_type = 'W conv'
        err_dict[f'{param_type}, layer: {i}, shape: {param.shape}'] = err

    return err_dict
        

        

        



