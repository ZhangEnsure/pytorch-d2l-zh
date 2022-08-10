# @Time : 2022/8/9 23:23 
# @Author : 张恩硕
# @File : d2l_torch.py
# @Software: PyCharm

import torch
from d2l import torch as d2l

"""
chap1-chap4
"""


# linear-regression-scratch

def synthetic_data(w, b, num_examples):
    """
    构造一个人造数据集，该数据集是一个带有噪声的线性数据集
    y = w*x + b
    """
    x = torch.normal(0, 1, size=(num_examples, w.shape[0]))
    y = torch.matmul(x, w) + b
    y = y + torch.normal(0, 0.01, y.shape)
    return x, y



