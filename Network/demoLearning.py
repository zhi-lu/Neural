#!/usr/bin/python3
"""
@author apple
@version : python3.9
// TODO : 2021/4/21
// 对神经网络的学习...
"""

import pandas
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# 创建一个分类器
class Classifier( nn.Module ):
    # 继承父类的构造方法
    def __init__(self):
        super().__init__()
        # 定义一个计数器
        self.counter = 0
        # 定义一个统计列表
        self.progress = []
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
        self.counter += 1
        # 每10个统计一次
        if self.counter % 10 == 0:
            self.progress.append( loss.item() )
            pass
        # 每 10000个进行统计
        if self.counter % 10000 == 0:
            print( "counter :", self.counter )
            pass

    # 进行相关的绘图
    def plot_progress(self):
        data_frame = pandas.DataFrame( self.progress, columns=['loss'] )
        data_frame.plot( ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker=".",
                         grid=True, yticks=(0, 0.25, 0.5) )
        pass


# 创建绘图器
class MnistDataset( Dataset ):
    # 进行初始化
    def __init__(self, csv_file):
        self.data_csv = pandas.read_csv( csv_file, header=None )

    # 计算相关的长度
    def __len__(self):
        return len( self.data_csv )

    # 返回第N项
    def __getitem__(self, index):
        # 目标图像
        label = self.data_csv.iloc[index, 0]
        # 独热编译
        # noinspection PyRedundantParentheses
        target = torch.zeros( (10) )
        target[label] = 1.0
        # 图像数据, 以0~255为取值范围, 以 0~1为标准化
        image_values = torch.FloatTensor( self.data_csv.iloc[index, 1:].values ) / 255.0
        # 返回标题,图像,和目标张量
        return label, image_values, target

    pass

    # 进行相关的绘图
    def img_plot(self, index):
        # 生成28*28的图
        img = self.data_csv.iloc[index, 1:].values.reshape( 28, 28 )
        # 创建标题
        plt.title( "title:" + str( self.data_csv.iloc[index, 0] ) )
        # 图片定义的格式
        plt.imshow( img, interpolation='none', cmap='Blues' )
        # 图片展示
        plt.show()


if __name__ == '__main__':
    mnist_dataset = MnistDataset( "/Users/apple/Downloads/neural/mnist_train.csv" )
    c = Classifier()
    epochs = 3

    for i in range( epochs ):
        print( "进行相关的训练", i + 1, "of", epochs )
        for label_, image_values_tensor, target_tensor in mnist_dataset:
            c.train( image_values_tensor, target_tensor )
            pass
        pass
    # 绘制损失图
    c.plot_progress()
