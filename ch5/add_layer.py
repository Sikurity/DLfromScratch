import numpy as np
from ch5.mul_layer import MulLayer


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


if __name__ == '__main__':
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # Forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    # Backward
    d_price = 1
    d_all_price, d_tax = mul_tax_layer.backward(d_price)
    d_apple_price, d_orange_price = mul_apple_layer.backward(d_all_price)

    d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)
    d_orange, d_orange_num = mul_orange_layer.backward(d_orange_price)

    print(price)
    print(d_apple_num, d_apple, d_orange, d_orange_num, d_tax)
