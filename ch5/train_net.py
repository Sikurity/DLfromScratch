import numpy as np
import os
from dataset.mnist import load_mnist
from ch5.multi_layer import MultiNet
import matplotlib.pyplot as plt

dataset_dir = os.path.dirname(os.path.abspath('__file__'))
save_file = dataset_dir + "/mnist.pkl"

(x_train, t_train), (x_test, t_test) = load_mnist(dataset_dir, save_file, normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Hyper Parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

net = MultiNet(input_size=784, hidden_size=100, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = net.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= lr * grads[key]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_train, t_train)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(i, ' train acc, test acc | ', str(train_acc) + ', ' + str(test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# print(train_loss_list)
