import numpy as np


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


if __name__ == '__main__':
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # Forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)

    # Backward
    d_price = 1
    d_apple_price, d_tax = mul_tax_layer.backward(d_price)
    d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)

    print(d_apple, d_apple_num, d_tax)
