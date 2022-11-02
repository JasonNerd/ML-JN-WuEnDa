# 自定义层
深度学习成功背后的一个因素是神经网络的灵活性： 我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。 例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。 未来，你会遇到或要自己发明一个现在在深度学习框架中还不存在的层。 在这些情况下，你必须构建自定义层。在本节中，我们将向你展示如何构建。

## 不带参数的层
1. 构造一个没有任何参数的自定义层CenteredLayer类, 它从其输入中减去均值。 
2. 将CenteredLayer作为一个组件放到一个更复杂的模型中```net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())```
3. 作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0

## 带参数的层
1. 实现自定义版本的全连接层 MyLinear 。 回想一下，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。 在此实现中，我们使用修正线性单元作为激活函数。 该层需要输入参数：in_units和units，分别表示输入数和输出数。
2. 使用 MyLinear 构造一个网络
```py
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```
## 练习
1. 设计一个接受输入并计算张量降维的层，它返回$$y_k = \sum_{i, j} W_{ijk} x_i x_j$$

2. 设计一个返回输入数据的傅立叶系数前半部分的层。

## 代码参考
**CenteredLayer**
```py
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
Y = net(torch.rand(4, 8))
Y.mean()
```
**MyLinear**
```py
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
linear.weight
linear(torch.rand(2, 5))
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```
