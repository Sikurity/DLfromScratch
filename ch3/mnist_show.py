import sys, os
import numpy as np
from PIL import Image
from ch3.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == '__main__':
    sys.path.append(os.pardir)  # Set to bring parent directory's file
    dataset_dir = os.path.dirname(os.path.abspath('__file__'))
    save_file = dataset_dir + "/mnist.pkl"
    (x_train, t_train), (x_test, t_test) = load_mnist(dataset_dir, save_file, flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)

    img = img.reshape(28,28)
    print(img.shape)
    img_show(img)
