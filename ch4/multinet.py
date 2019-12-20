import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class MultiNet:
    def __init__(self, input_size, hidden_size, output_size, init_w_std=0.01):
        self.params = {'W1': init_w_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': init_w_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size)}

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        y1 = np.dot(x, W1) + b1
        s1 = sigmoid(y1)
        y2 = np.dot(s1, W2) + b2
        s2 = softmax(y2)

        return s2

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

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