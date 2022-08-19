# @Time : 2022/8/19 20:05 
# @Author : 张恩硕
# @File : test_d2l_torch_chap.py 
# @Software: PyCharm

from d2l_torch_chap import *

# test class Net begin
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
net = Net(num_inputs, num_hiddens1, num_hiddens2, num_outputs)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# end