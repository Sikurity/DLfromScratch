import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class MultiNet:
    def __init__(self, input_size, hidden_size, output_size, init_w_std=0.01):
        self.params = {'W1': init_w_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': init_w_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size)}

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.last = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.last.forward(y, t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.last.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {'W1': self.layers['Affine1'].dW,
                 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db,}


        return grads


if __name__ == '__main__':
    net = MultiNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    x = np.random.rand(100, 784)    # batch=100
    t = np.random.rand(100, 10)
    y = net.predict(x)

    grads = net.numerical_gradient(x, t)

    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)