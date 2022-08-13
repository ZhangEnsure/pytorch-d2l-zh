# @Time : 2022/8/9 23:23 
# @Author : 张恩硕
# @File : d2l_torch.py
# @Software: PyCharm

import torch
import random
from d2l import torch as d2l
from torch.utils import data
from IPython import display
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
            param -= param.grad * lr / batch_size
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
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


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

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    在 jupyter notebook 中绘图
    :param imgs: 图片列表
    :param num_rows: 子图的行数
    :param num_cols: 子图的列数
    :param titles: 子图的标题
    :param scale:
    :return:
    """
    # 先列后行!
    figsize = (num_cols * scale, num_rows * scale)
    _, axs = d2l.plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    # 在用 plt.subplots 画多个子图中，ax = ax.flatten()将ax由n*m的Axes组展平成1*nm的Axes组
    # 将二维矩阵展平为一维的向量，可以与匹配的img进行配对
    axs = axs.flatten()
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        if titles:
            ax.set_title(titles[i])
        # 设置取消 x y轴坐标
        # set_visible() 别拼错了！
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # Returns True if obj is a PyTorch tensor.
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
    return axs


# softmax-regression-scratch

w = torch.normal(0, 0.1, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)


def softmax(x):
    """
    实现 softmax 函数
    :param x: 未规范化输出
    :return:
    """
    x = torch.exp(x)
    partital = x.sum(dim=1, keepdims=True)
    return x / partital


def net(X):
    """
    设计向前传播函数
    1. 这里定义的 net 函数，是为了让后面的训练函数统一形式，而应用于框架实现、手动实现
    2. 这里需要的是规范化的输出，可以直接进入 cross_entropy 计算损失
    3. 因为这里是手动实现 softmax 回归，所以 w b 都是全局变量定义的
    """
    return softmax(torch.matmul(X.reshape(-1, w.shape[0]), w) + b)


def updater(batch_size):
    """
    设计 updater 函数；实用函数
    1. 作用同上面的 net 函数，即统一训练形式
    """
    lr = 0.1
    d2l.sgd([w, b], lr, batch_size)


def cross_entropy(y_hat, y):
    """
    实现交叉熵损失函数
    reduction='none'
    """
    return -torch.log(y_hat[range(len(y)), y])


def accuracy(y_hat, y):
    """
    工具函数
    返回预测正确的元素个数
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # axis = 1，代表需要让列消失（合并），所以是对每一行求最大值所在的列下标
        y_hat = y_hat.argmax(axis=1)
    cpm = y_hat.type(y.dtype) == y
    return float(cpm.type(y.dtype).sum())


class Accumulator:
    """
    工具类，用以存储正确预测的数量和预测总数
    """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    # 魔法方法，重载[]运算符
    def __getitem__(self, index):
        return self.data[index]


def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度,实用函数
    :param net: 模型网络（框架）/自定义实现
    :param data_iter:
    :return: acc
    """
    if isinstance(net, torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            # y.numel()是样本总数
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期(epoch)
    :param net: 这里自适应了手动实现和模型实现
    :param train_iter: 训练数据的迭代器
    :param loss: 损失函数
    :param updater: 优化器
    :return: train_loss;train_acc
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            # 这里要求模型的损失函数必须是 batch_size 样本的损失和
            # 使用框架时要注意这点
            l.mean().backward()
            updater.step()
        else:
            # 手动实现要求损失是一个向量
            l.sum().backward()
            # 因为这里要除以 batch_size
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    :param net: 待训练模型
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param loss: 损失函数
    :param num_epochs: 迭代训练集次数
    :param updater: 优化器
    :return: -
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        #         print(type(train_metrics), train_metrics)
        # 在测试数据集上评估我们的精度
        test_acc = evaluate_accuracy(net, test_iter)
        # 这里的参数不是太懂
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
