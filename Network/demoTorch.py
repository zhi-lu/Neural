#!/usr/bin/python3

"""
@author : apple
@version : python 3.9
// TODO : 2021/5/21
// 机器学习 Pytorch 进行相关对深度学习,和图像的分析
"""
import pandas
import matplotlib.pyplot as plt


def main():
    data_frame = pandas.read_csv("/Users/apple/Downloads/neural/mnist_train.csv", header=None)
    print(data_frame.head())
    # 第一行的数据
    raw = 13
    data = data_frame.iloc[raw]
    # 将第一个数字作为标题
    label = data[0]
    # 将一行剩下的数据重新生成28*28的图片
    img = data[1:].values.reshape(28, 28)
    plt.title("title:" + str(label))
    plt.imshow(img, interpolation='none', cmap='Blues')
    plt.show()


if __name__ == '__main__':
    main()
