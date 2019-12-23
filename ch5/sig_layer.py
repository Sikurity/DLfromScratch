import numpy as np


class Sigmoid:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        out = 1. / (1. + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1. - self.out) * self.out

        return dx


if __name__ == '__main__':
    x = np.array([[1.0, -0.5]])
    print(x)

    # Forwad
    print(Sigmoid().forward(x))
