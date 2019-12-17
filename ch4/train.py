import sys, os
import numpy as np
from ch3 import nn_mnist


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):                          # t -> Answer = 1, Not Answer = 0
    if y.ndim == 1:     # For Batch
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    batch_size = y.shape[0]
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size   # if y is not one-hot encoding
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


if __name__ == '__main__':
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y0 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    y1 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])

    print(mean_squared_error(y0, t))
    print(mean_squared_error(y1, t))
    print(cross_entropy_error(y0, t))
    print(cross_entropy_error(y1, t))

    sys.path.append(os.pardir)  # Set to bring parent directory's file
    dataset_dir = os.path.dirname(os.path.abspath('__file__'))
    save_file = dataset_dir + "/mnist.pkl"

    (x_train, t_train), (x_test, t_test) = nn_mnist.load_mnist(dataset_dir, save_file, normalize=True, one_hot_label=True)

    print(x_train.shape)
    print(t_test.shape)

    batch_size = 10
    batch_mask = np.random.choice(x_train.shape[0], 10)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
