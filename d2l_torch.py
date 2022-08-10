# @Time : 2022/8/9 23:23 
# @Author : 张恩硕
# @File : d2l_torch.py
# @Software: PyCharm

import torch
import random
from d2l import torch as d2l

"""
chap1-chap4
"""


# linear-regression-scratch

def synthetic_data(w, b, num_examples):
    """
    构造一个人造数据集，该数据集是一个带有噪声的线性数据集
    y = w*x + b
    x 中的每一行都包含一个二维数据样本， y 中的每一行都包含一维标签值（一个标量）
    """
    x = torch.normal(0, 1, size=(num_examples, w.shape[0]))
    y = torch.matmul(x, w) + b
    y = y + torch.normal(0, 0.01, y.shape)
    return x, y


def data_iter(batch_size, features, labels):
    """
    定义一个data_iter函数，接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
    在向前传播计算中，我们需要一次取一个批量大小的样本进行计算
    """
    # 返回 features 的第一维长度
    numberExample = len(features)
    # 需要一个打乱下标的列表 indices
    indices = list(range(numberExample))
    random.shuffle(indices)
    # 从 0 到 num_example 之间一次取出 batch_size 个大小的数据
    for i in range(0, numberExample, batch_size):
        # 获得打乱下标后的顺序 batch_size 个下标列表
        batch_indices = indices[i:min(i + batch_size, numberExample)]
        yield features[batch_indices], labels[batch_indices]
