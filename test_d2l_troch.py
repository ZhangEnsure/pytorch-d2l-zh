# @Time : 2022/8/10 12:01 
# @Author : 张恩硕
# @File : test_d2l_troch.py 
# @Software: PyCharm

from d2l_torch import *
import torch
from torch import nn


# linear-regression-scratch

def getLinRegData():
    """
    :return: 人造线性数据集
    """
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    return features, labels


def test_synthetic_data():
    features, labels = getLinRegData()

    d2l.set_figsize()
    d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), s=1)
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)
    d2l.plt.show()  # pycharm 专属


# test_synthetic_data()

def test_data_iter():
    features, labels = getLinRegData()
    batch_size = 100
    # 我们打印第一个迭代对象的 features 的形状进行验证
    print(next(iter(data_iter(batch_size, features, labels)))[0].shape)


# test_data_iter()


def LinRegTrain():
    """
    linear-regression-scratch
    :return:
    """
    w_learn = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b_learn = torch.zeros(1, requires_grad=True)
    batch_size = 100
    epochs = 3
    lr = 0.1
    net = linreg
    loss = squared_loss
    features, labels = getLinRegData()
    for epoch in range(epochs):
        for x, y in data_iter(batch_size, features, labels):
            l = loss(net(x, w_learn, b_learn), y)
            # print(l.shape)
            l.sum().backward()
            sgd([w_learn, b_learn], lr, batch_size)
        with torch.no_grad():
            trainLoss = loss(net(x, w_learn, b_learn), y)
            print(f'epoch {epoch + 1}, loss {float(trainLoss.mean()):.6f}')


# LinRegTrain()


# 不可变对象参数修改后 id 地址发生变化，实参的原 id 地址所指向的值不变
# 可变参数例如 list，在函数形参中如果发生对象的修改变化则 id 不变，非对象修改
# 则属于创建一个新的对象，id 发生变化
# data = [1, 2, 3]
# print(id(data))


def test_object_pass(x):
    print(id(x))
    x.pop()
    print(x)
    print(id(x))
    x = x + [3]
    print(x)
    print(id(x))


# test_object_pass(data)
# print(f'{data}')
# print(id(data))
"""
2799466353920
2799466353920
[1, 2]
2799466353920
[1, 2, 3]
2799466354112 # 这里的修改对象，属于创建一个新的对象
[1, 2]
2799466353920 # 原 data 对象的 id 不变，但是内容被对象内修改导致值发生了变化
"""


# linear-regression-concise

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


def moduleLinReg():
    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    net.apply(init_weights)
    loss = nn.MSELoss()
    # 记得加入 net.parameters()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    features, labels = getLinRegData()
    batch_size = 10
    epochs = 3
    data_iter= load_array((features, labels), batch_size)
    for epoch in range(epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            # 在优化函数中将梯度清零
            trainer.zero_grad()
            l.backward()
            trainer.step()
        trainLoss = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {float(trainLoss.mean()):.6f}')

moduleLinReg()