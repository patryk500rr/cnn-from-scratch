import numpy as np

class Dense:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, A):
        Z = np.dot(A, self.W) + self.b

    
