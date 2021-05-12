#!/usr/bin/py3

"""
@author : apple
@version python3.9
// TODO : 2021/5/1
神经网络的学习,对神经结点进行相关对分析.求出梯度值(gradient)
"""

import torch


# value of tensor为分数或者为复数
def main():
    # 在tensor函数中 默认requires_grad(要求梯度值为 False)
    x = torch.tensor(3.5, requires_grad=True)
    print(x)
    print("<===================================>")
    y = (x - 1) * (x - 2) * (x - 3)
    print(type(y), y)
    y.backward()
    print(x.grad)
    # 基于Pytorch通过微积分的链式求导法则求出相关的梯度值
    print("<====================================>")
    x = torch.tensor(4.0, requires_grad=True)
    y = x * x
    z = 2 * y + 3
    # noinspection PyUnresolvedReferences
    z.backward()
    print(x.grad)
    print("<+++++++++++++++++++++++++++++++++++++>")
    test()


def test():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    print(a, b)
    x = 2 * a + 3 * b
    y = 2 * a * a + 3 * b * b * b
    z = 2 * x + 3 * y
    # 求出 dz/da 和 dz/da 的梯度值当 a = 2.0 和 b = 3.0时
    z.backward()
    print(a.grad, b.grad)


if __name__ == '__main__':
    main()
