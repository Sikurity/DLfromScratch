import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    # y = XW + b
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)         # dx = dout * W.T
        self.dW = np.dot(self.x.T, dout)    # dW = x.T * dout
        self.db = np.sum(dout, axis=0)      # db = sum(dout)

        return dx
