# @Time : 2022/9/1 12:34 
# @Author : 张恩硕
# @File : d2l_torch_convolutional_neural_network.py 
# @Software: PyCharm

import torch
from d2l import torch as d2l
from torch import nn

'''
卷积层
'''


def corr2d(X, K):
    '''
    计算二维互相关运算
    :param X: 输入的张量
    :param K: 卷积核
    :return: 卷积结果
    '''
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    # 实现二维卷积层
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def detection_vertical_edge():
    '''
    卷积层的一个简单应用： 检测图像中不同颜色的边缘
    :return:
    '''
    x = torch.ones(6, 8)
    x[:, 2:6] = 0
    print(x)
    k = torch.tensor([[-1.0, 1.0]])
    print(corr2d(x, k))


def learn_detection_vertical_edge():
    '''
    通过梯度下降学习卷积核的参数
    :return:
    '''
    x = torch.ones(6, 8)
    x[:, 2:6] = 0
    x = x.reshape((1, 1, 6, 8))
    y = torch.zeros(6, 7)
    y[:, 1] = -1
    y[:, 5] = 1
    y = y.reshape((1, 1, 6, 7))

    conv = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    lr = 3e-2
    epochs = 15
    for epoch in range(epochs):
        y_hat = conv(x)
        l = (y_hat-y)**2
        conv.zero_grad()
        l.sum().backward()
        conv.weight.data[:] -= lr*conv.weight.grad
        if (epoch%2) == 1:
            print(f'第{epoch+1}次训练，loss为{l.sum()}')
    print(conv.weight.data.reshape((1, 2)))


