# @Time : 2022/8/9 23:23 
# @Author : 张恩硕
# @File : d2l_torch.py
# @Software: PyCharm

import torch
import random
from d2l import torch as d2l
from torch.utils import data
import torchvision
from torchvision import transforms


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


def sgd(params, lr, batch_size):
    """
    李沐老师实现的 sgd 优化方法
    :param params: 作为一个列表传入
    :param lr: 学习率
    :param batch_size: 计算损失时没有求平均，是 batch_size 个样本的损失和
    :return: 无
    """
    # 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
    with torch.no_grad():
        for param in params:
            # error warning: param = param - param.grad*lr/batch_size
            # 这里涉及到 python 对象参数传递的问题，可见 test_object_pass 函数
            # 这里的 params id 不发生变化，所以这里修改，实参也发生变化
            param -= param.grad*lr/batch_size
            # pytorch会不断的累加变量的梯度，所以每更新一次参数，都要使对应的梯度清零
            param.grad.zero_()


def linreg(X, w, b):
    """
    定义模型（向前传播）
    :param X: 特征
    :param w: 权重
    :param b: 偏差
    :return: 向前传播计算结果
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    定义损失函数
    :param y_hat: 预测值
    :param y: 实际标签值
    :return: 平方损失
    """
    return (y_hat - y.reshape(y_hat.shape))**2/2


# linear-regression-concise

def load_array(data_arrays, batch_size, is_train=True):
    """
    在上面我们手动实现了根据 batch_size 返回样本和标签，这里我们使用框架来实现
    :param data_arrays: 样本和标签组成的元组
    :param batch_size: 批量大小
    :param is_train:
    :return:
    """
    # TensorDataset:把输入的两类数据进行一一对应
    TensorDataset = data.TensorDataset(*data_arrays)
    # DataLoader：重新排序
    return data.DataLoader(TensorDataset, batch_size, shuffle=is_train)


# image-classification-dataset





