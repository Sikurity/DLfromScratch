import matplotlib.pyplot as plt
from common.functions import *

x = np.random.rand(1000, 100)
node_num = 100
hidden_layer_size = 5
activation_output = {}

for i in range(hidden_layer_size):
    if i > 0:
        x = activation_output[i - 1]

    w = np.random.randn(node_num , node_num) * 1
    a = np.dot(x, w)
    z = sigmoid(a)
    activation_output[i] = z

for layer_no, layer_weight in activation_output.items():
    plt.subplot(1, len(activation_output), layer_no + 1)
    plt.title(str(layer_no + 1) + "-layer")
    plt.hist(layer_weight.flatten(), 30, range=(0, 1))
plt.show()
