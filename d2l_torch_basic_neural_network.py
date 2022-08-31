#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn.functional as F
from torch import nn


# 自定义块，最主要的就是实现两个方法。
# 1. 初始化方法不要忘记调用父类的__init__()
# 2. forward()方法是父类 Module.__call__() 调用的
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))


net = MLP()
net(torch.rand(2, 20))


# 我们同样可以自定义实现一个顺序块。
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 每一个 module 例如 nn.Linear() 都是 nn.Module 子类的实例，我们把它保存在 Module 类的成员 _modules 中
            # _modules 类型是 OrderedDict
            self._modules[str(idx)] = module

    def forward(self, x):
        # OrderedDict 保障了按照成员添加的顺序遍历他们
        for block in self._modules.values():
            x = block(x)
        return x


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(torch.rand(2, 20))


# 搭配混合使用各种组合块的方法。
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20))
chimera(torch.rand(2, 20))

# 参数管理
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())

print(net[2].state_dict()["weight"])

# 打印输出结果显示，这是一个torch.nn.parameter.Parameter的实例
print(type(net[2].bias))
# 参数类的实例，包括值、梯度和额外信息
print(net[2].bias)
# 进一步访问参数值
print(net[2].bias.data)
# 一次访问所有的参数
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['0.weight'].data)


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

# 打印模型
print(rgnet)

'''
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
'''

# 打印第一个顺序块的第二个块的第一个module
# print(rgnet[0][1][0].bias.data)


# 块的初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


# net.apply(init_normal)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


# net.apply(init_constant)


# 对某些块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


# net[0].apply(xavier)
# net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)

# 参数绑定
'''
这里的 net[2]、net[4] 都是一样的参数
'''
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])

'''
层
'''


class CenteredLayer(nn.Module):
    '''构造一个没有任何参数的自定义层'''

    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        进行均值为 0 的处理
        :param X: input
        :return:
        '''
        return X - X.mean()


# layer = CenteredLayer()
# layer(torch.FloatTensor([1, 2, 3, 4, 5]))
# 将层作为组件合并到更复杂的模型中
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())


class MyLinear(nn.Module):
    '''
    带参数的层，相当于一个自带 relu 的全连接层
    '''

    def __init__(self, in_units, units):
        super().__init__()
        # 这里要使用 nn.Parameter()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

# linear = MyLinear(5, 3)
# print(linear.weight)

# 保存模型
net = MLP()
torch.save(net.state_dict(), 'mlp.params')

# 恢复模型
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))

