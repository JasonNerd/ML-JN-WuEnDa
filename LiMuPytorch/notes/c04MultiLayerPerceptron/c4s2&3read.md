# 多层感知机的实现
&emsp;&emsp;在softmax回归中，为了解决多分类问题，我们使用了一个全连接层对输入进行了仿射变换得到一个输出，为了使输出符合概率的数学性质，对于这个输出进行了softmax操作, 也即取指数后再归一化。而在多层感知机中，我们有多个全连接层，对于单个的节点，将上一层的输出进行仿射变换后再用一个非线性函数进行非线性变换得到最终输出，因此就可以得到非线性的模型。

## 实验过程
1. 导入Fashion-MNIST图像分类数据集，得到训练集和测试集
2. 假定为单隐含层网络，隐含层节点数为256，注意图片大小28x28=784，输出为10个分类，使用几个张量W1 W2 b1 b2表示参数，可以用`nn.Parameter`将参数组合起来
3. 实现一个激活函数，例如ReLU。
4. 实现模型函数，也即从输入到输出的假设函数
5. 仍然使用交叉熵损失
6. 进行训练，画出损失下降图，给出测试集总体精确度

## 使用高级API简化代码
1. 通过Sequential将MLP搭建起来，同样使用apply(init_weights)初始化参数
2. 设置batch_size, lr, num_epochs等超参数，使用交叉熵损失函数和SGD优化算法，读取训练集和测试集，作出损失图和测试精度图
![](https://zh-v2.d2l.ai/_images/output_mlp-concise_f87756_30_0.svg)

## 参考代码
### 参考代码1
```py
###
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
###
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]
###
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
###
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
###
loss = nn.CrossEntropyLoss(reduction='none')
###
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

### 参考代码2
```py
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```