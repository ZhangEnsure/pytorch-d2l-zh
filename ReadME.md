# 动手学习深度学习

在 Bilibili 学习[李沐老师的课程过程](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)中，收获了不少的代码实战经验。吴恩达老师的课程更多偏向于知识的讲解而缺少实践，虽然有代码课，但是更多的是对知识的理解。李沐老师的课程不仅讲解基础知识，而且讲解实战代码。沐老师的一些代码精简干练、复用性高，我希望不仅可以熟练应用和复用沐老师的代码，而且能够深刻理解代码的设计和实现，以便在学习深度学习的过程中也可以提升自己的 python、pytorch 编程能力。

在本仓库中，主要存放沐老师的一些 d2l 库 torch 模块的代码以及对该代码的注释分析、自我实现等等，以便自己以后的复习和调用。对课程内容的学习记录在 [CSDN](https://blog.csdn.net/Mr_Yuwen_Yin) 上，在此不做过多分析和介绍。

# 学习内容提纲

## 线性回归

在线性回归部分，我们人工构造了一个带有噪声的线性数据集，并尝试使用手工和 pytorch 框架两种方式实现线性回归，并设计了生成数据集函数 synthetic_data()、按批量获取数据函数 data_iter() load_array()、向前传播 net() 函数、损失函数 squared_loss() 函数、优化函数 SGD() 等等。

在手工实现线性回归代码中，我曾经踩坑 SGD 的手动实现，并借此机会认真学习了一下 python 的参数传递规则。总结下来就是可变类型变量（例如列表）作为形参传入，在函数中如何修改该变量的方式，决定了是否创建一个新的变量（分配内容地址），此时新变量的任何修改均不影响原实参，如果未分配内存创建新的变量，则原实参的值也随之变化，具体请参考 test_object_pass() 函数。此外，还有损失函数需要传递 batch_size 批量大小等等。

## softmax线性回归

softmax线性回归这里，我们处理的数据是来自 Fashion-MNIST 数据集，我们需要学习该数据集的读取方式、可视化函数 show_images() 的实现。

手动实现部分，为了可以和框架实现部分实现训练代码重用，我们需要从一开始便对手动实现部分的代码进行规范化。例如：

1. 向前传播的 net 函数
2. 损失函数 loss
3. 优化器函数 trainer

其次，我们需要定义 softmax 函数，实现我们对 y_hat 输出值的规范化。Accumulator 类是应用于计算训练集和测试集在模型 net 上的 acc 精确性。train_epoch_ch3 函数是实现一次 epoch 的计算问题，特别地，我们要分别对手动和框架实现进行不同处理。

train_epoch_ch3 中，最主要的是 loss 计算后的形式我们要清楚。手动实现的 loss 是一个向量，我们需要 sum() 求和后进行自动梯度求导计算，在 updater 中还需要求 loss 的平均值后进行参数更新。在框架实现中，我们使用参数 reduction='none'，这样的话，得到的损失就是一个向量，随后我们将损失求和取平均后进行自动求导。不过需要注意的是，pytorch 的变量梯度是累加的计算，我们在计算新的梯度前需要把上次计算的梯度清零。

除此之外，在分类问题，我们常常使用的是交叉熵 CrossEntropyLoss 损失函数，对这个函数的介绍，我在 CSDN 中写了详细的[文章](https://blog.csdn.net/Mr_Yuwen_Yin/article/details/126174583)供参考，包括手动实现 CrossEntropyLoss 等等。

## 多层感知机 multilayers perceptrons

在softmax regression 中，我们的模型只有输入层和输出层。以 Fashion-MNIST 数据集为例，我们将一个 28*28 的图像展平为 784 的图片向量，并且一次读取一个 batch_size 批量的样本，但是这次我们加入了一个隐藏层叠加在输入层和输出层之间，并且使用非线性激活函数作为隐藏层的激活函数。总结来说，多层感知机也就是使用隐藏层和激活函数来得到非线性模型，在代码部分其实没有什么新知识。

## 深度学习计算

### 层和块

块可以描述单个层、多个层组成的组件或者整个模型本身。从编程的角度，块就是一个类。我们需要在类中定义相关参数和方法，例如一个将输入转换为输出的向前传播函数，一个存储参数的初始化方法。

> 请注意，我们自定义的块实现中 `super().__init__()` 是调用父类 Module 的构造函数执行必要的初始化操作。

自定义层可以先我们特定的行为，并且可以在网络设计中使用该层。带参数的层是可以进行参数更新的，不过定义的参数需要使用 `nn.Parameter()` 声明。

### 参数管理

访问参数，我们可以通过 net[i].weight 方法访问 Sequential 定义的网络中的某 i 层的权重数据，同样可以通过 for loop 遍历 `net.named_parameters()` 获取 `name, param.shape` 等等数据。

参数初始化，我们可以为模型中的某一层设置特定的初始化函数，这样的话需要使用 `net[i].apply()` 方法。

共享参数，也就是将模型中的某几个层绑定相同的参数，同步更新。

### 读写文件

在这里我们主要介绍了保存模型参数和读取模型参数的两个方法。需要注意的是，我们这里并没有保存模型的结构，而是后续重写模型的参数。

## 卷积神经网络 Convolutional Neural Network

之前的 mlp 十分适合处理表格类的数据，列是特征，行是样本。我们可以通过寻找到一个恰当的模式去拟合样本特征。但是对于高维的感知数据，CNN 可能是一个更好的选择。例如，对于一个灰度图 `2D [k,l]`样本，如果使用全连接层 mlp 处理的话，隐藏层的权重矩阵应该是 `4D [i,j,k,l]`。因为每一个神经元都要与输入的每一个特征连接，这便确定了 4D 的后两维维数。

但是直观上理解，mlp 是试图学习某个特征点处的权值，如果我们做分类问题的话，我们的目标点是否在特定坐标是无联系的，它可能出现在图片中的任意地点，也就是我们要保证`平移不变性`，这就要求我们的卷积核是在不同坐标处应该是一致的。与此同时，我们的分类器应该在一定范围内聚合局部特征，具有`局部性`，不应过大联系较远区域，这就限制了卷积核的大小。

在卷积层中，输入张量和核张量通过**互相关运算**，并在添加标量偏置后输出。这里想到在计算所研究生面试时导师问我的一个问题，1*1 卷积核卷积相当于什么？我当时啥也不会，就实话实说了。现在看来这个问题比较简单，就是相当于一个全连接层。所幸，最后导师还是要我了。