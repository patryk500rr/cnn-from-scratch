"""Parameter update rules."""


def SGD(parameters, lr):
    """Apply one gradient descent update in place: param -= lr * grad.

    Call this after model.backward(), so that the gradients are up to date.

    Args:
        parameters (list): (param, grad) pairs, as returned by model.parameters().
            The arrays are updated in place, so the layers see the new values.
        lr (float): learning rate.
    """
    for param, d_param in parameters:
        param -= lr * d_param
            
        
        
        

