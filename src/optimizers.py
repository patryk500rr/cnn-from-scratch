def step_SGD(parameters, lr):
    for param, d_param in parameters:
        param -= lr * d_param
            
        
        
        

