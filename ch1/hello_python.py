import numpy as np

x = 10
print(x)
print(type(x))

x = 3.1415
print(x)
print(type(x))

x = 'hello'
print(x)
print(type(x))

print(type(10 * 3.1415))

a = [1, 2, 3, 4, 5]
print(a)
print(len(a))

print(a[0])
a[2] = 0
print(a[4])
print(a[0:2])
print(a[:2])
print(a[-1])
print(a[1:-1])
print(a[1:])
print(a[-2])


d = {'attr': -1}
print(d['attr'])

d['attr'] = 0
print(d['attr'])


def isTrueOrFalse(x):
    if x:
        print('True')
    else:
        print('False')

x = True
isTrueOrFalse(x)

x = not x   # False
isTrueOrFalse(x)

class MyClass:
    # Constructor
    def __init(self, param):
        self.param = param

    def method1(self):
        print('method1 executed and param: ', self.param)

    def method2(self):
        print('method2 executed and param: ', self.param)

x = np.array([1, 2, 3], dtype='float32')
print(type(x))

y = np.array([-1, -1, -1], dtype='float32')

print(x + y)
print(x + 2 * y)
print(x + 3 * y)

print(x * y)

print(x / y)

print(x ** y)

print(x // y)

print(x / 2.0)

# broadcast of numpy
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])

print(A * B)
# [[1, 2]    [[10, 20]
#  [3, 4]] X  [10, 20]]

X = np.array([[0, 1], [2, 3], [4, 5]])
print(X[0])
print(X[0][0])

for row in X:
    print(row)

print(X.flatten())

print(X > 2)
print(X[X > 2])

# matplotlib
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)    # 0, 0.1, 0.2, ... 5.9, 6.0
y = np.sin(x)
z = np.cos(x)

plt.plot(x, y, label="sin")
plt.plot(x, z, label="cos", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()    # 범례 표시
plt.show()

from matplotlib.image import imread

img = imread('Lenna.png')

plt.imshow(img)
plt.show()
