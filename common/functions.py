import numpy as np


def identity_function(x):
    return x


def step_function(x):
    # y = x > 0 # if element of x <= 0 then false // else then true
    # return y.astype(np.int) # invert to int
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def identity_function(x):
    return x


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):                          # t -> Answer = 1, Not Answer = 0
    if y.ndim == 1:     # For Batch
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    batch_size = y.shape[0]

    if t.size == y.size:
        t = t.argmax(axis=1)
        y = y[np.arange(batch_size), t]

    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)