# @Time : 2022/8/10 12:01 
# @Author : 张恩硕
# @File : test_d2l_troch.py 
# @Software: PyCharm

from d2l_torch import *
import torch

def test_synthetic_data():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    d2l.set_figsize()
    d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), s=1)
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)
    d2l.plt.show() # pycharm 专属

# test_synthetic_data()