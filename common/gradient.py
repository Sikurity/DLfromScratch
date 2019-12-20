import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, *args, use_asterisk=False):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(*args)

    if use_asterisk:
        xs = list(args)
        for i in range(len(xs)):
            t = xs[i]

            df_dx_i = []
            xs[i] = t + h  # f(x+h)
            df_dx_i.append(f(*xs))

            xs[i] = t - h  # f(x+h)
            df_dx_i.append(f(*xs))
            grad[i] = (df_dx_i[0] - df_dx_i[1]) / (2 * h)

            xs[i] = t
    else:
        xs = args[0]
        it = np.nditer(xs, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index

            tmp_val = xs[idx]
            xs[idx] = tmp_val + h
            fxh1 = f(xs)  # f(x+h)

            xs[idx] = tmp_val - h
            fxh2 = f(xs)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            xs[idx] = tmp_val
            it.iternext()

    return grad


def gradient_descent(f, *init_x, lr=0.01, step=100, debug=False):
    x = init_x

    for i in range(step):
        grad = numerical_gradient(f, *x)
        x -= lr * grad
        if debug:
            print('step: ', i)
            print('grad: ', grad)
            print('newX: ', x)

    return x


if __name__ == '__main__':
    def f(x):
        return 0.01 * x ** 2 + 0.1 * x

    x = np.arange(0.0, 20.0, 0.1)

    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.plot(x, f(x))
    # plt.show()
    #
    # plt.xlabel("x")
    # plt.ylabel("f'(x)")
    # plt.plot(x, numerical_diff(f, x)[0])
    # plt.show()

    def f2x(*x):
        return x[0]**2 + x[1]**2

    init_x = [-3.0, 4.0]
    gradient_descent(f2x, *init_x, lr=0.1, step=100, debug=True)
    # gradient_descent(f2x, *init_x, lr=10.0, step=100, debug=True)
    # gradient_descent(f2x, *init_x, lr=1e-10, step=100, debug=True)
