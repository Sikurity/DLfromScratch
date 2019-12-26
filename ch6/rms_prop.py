import numpy as np


class RMSprop:
    """
    RMSprop
        Adagrad와 달리 기울기를 단순 누적하지 않고
        지수 가중 이동 평균(Exponentially weighted moving average)를
        사용하여 최신 기울기들이 더 크게 반영되도록 하였다.
    """

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)