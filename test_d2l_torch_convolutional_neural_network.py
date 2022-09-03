# @Time : 2022/9/1 12:55 
# @Author : 张恩硕
# @File : test_d2l_torch_convolutional_neural_network.py.py 
# @Software: PyCharm

from d2l_torch_convolutional_neural_network import *


def test_corr2d():
    X = torch.arange(0, 9).reshape((3, 3))
    K = torch.tensor([0, 1, 2, 3]).reshape(2, 2)
    print(corr2d(X, K))


# detection_vertical_edge()
'''output
tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
tensor([[ 0., -1.,  0.,  0.,  0.,  1.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  1.,  0.]])
'''

# learn_detection_vertical_edge()
'''
第2次训练，loss为5.468018054962158
第4次训练，loss为0.936331570148468
第6次训练，loss为0.16485238075256348
第8次训练，loss为0.030836956575512886
第10次训练，loss为0.006475823000073433
第12次训练，loss为0.0016198670491576195
第14次训练，loss为0.0004902561195194721
tensor([[-1.0003,  0.9971]])
'''

def test_corr2d_multi_in():
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))

# test_corr2d_multi_in()

def test_corr2d_multi_in_out():
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    K = torch.stack([K, K + 1, K + 2], 0)
    print(corr2d_multi_in_out(X, K))

# test_corr2d_multi_in_out()