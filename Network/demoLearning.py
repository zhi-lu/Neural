#!/usr/bin/python3
"""
@author apple
@version : python3.9
// TODO : 2021/4/21
// 对神经网络对学习...
"""
import torch
import torch.nn as nn


# 创建一个分类器
class Classifier( nn.Module ):
    # 继承的构造方法
    def __init__(self):
        super().__init__()

        # 定义一个神经网络
        # noinspection PyTypeChecker
        self.model = nn.Sequential(
            # 从784映射到200的全映射关系
            nn.Linear( 784, 200 ),
            nn.Sigmoid(),
            nn.Linear( 200, 10 ),
            nn.Sigmoid(),
        )
        # 创建线性误差函数
        self.loss_function = nn.MSELoss()
        # 创建优化器,使用简单的梯度下降
        self.optimizer = torch.optim.SGD( self.parameters(), lr=0.01 )

        pass

    def forward(self, inputs):
        # 直接运行模型
        return self.model( inputs )

    # noinspection PyMethodOverriding
    def train(self, inputs, targets):
        outputs = self.forward( inputs )
        loss = self.loss_function( outputs, targets )
        # 将开始的梯度设置为0
        self.optimizer.zero_grad()
        loss.backward()
        # 更新的学习参数
        self.optimizer.step()