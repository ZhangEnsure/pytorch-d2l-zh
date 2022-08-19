# @Time : 2022/8/19 18:03 
# @Author : 张恩硕
# @File : d2l_torch_chap.py 
# @Software: PyCharm

import torch
from torch import nn
from d2l import torch as d2l


# from torch.utils import data
# from IPython import display
# import torchvision
# from torchvision import transforms

# dropout
def dropout_layer(X, dropout):
    """
    我们实现 dropout_layer 函数,该函数以dropout的概率丢弃张量输入X中的元素
    :param X: 待处理的张量
    :param dropout: 丢弃的概率
    :return: 处理后的张量
    """
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # torch.rand 返回的是 [0,1] 之间的均匀分布
    # torch.rand(size=X.shape) > dropout 其实返回的是一个布尔值的矩阵
    mask = (torch.rand(size=X.shape) > dropout).float()
    return X * mask / (1.0 - dropout)


# 学习定义具有两个隐藏层的多层感知机，每个隐藏层包含 256 个单元
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    """
    手动定义实现 net 网络
    """

    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_training = is_training
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.linear1(X.reshape(-1, self.num_inputs))
        H1 = self.relu(H1)
        if self.is_training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.linear2(H1)
        H2 = self.relu(H2)
        if self.is_training:
            H2 = dropout_layer(H2, dropout2)
        H3 = self.linear3(H2)
        return H3


