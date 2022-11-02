# 暂退法（Dropout）

## 偏差-方差平衡(bias-variance tradeoff)
以下来自西瓜书的经典论述：假设我们有很多个数据集$D$，数据集之间相互独立, 都可以作为模型$F$的输入, 每个数据集都有对应的标记$y_D$, 标记可能包含着噪声$\epsilon$, 也即$y_D$与真实值$y$之间的差异。据此, 偏差$var$被定义为模型$F$预测输出的期望$E(F_D)$(可以理解为模型在所有数据集上预测值的均值)与真实标记之间的差异, 方差被定义为模型$F$在各个数据集上的预测输出与期望$E(F_D)$之间的差异的期望。

进一步讲, 如果独立数据集的数量是无限的，那么期望$E(F_D)$就近似描绘了算法模型的预测期望或者表达能力，我们说一个算法趋向于具有高偏差，意味着这个算法表示能力较弱，拟合能力较差，但是这一算法往往在各个数据集上的表现都相差不大，具有较低的方差。另一方面，如果模型在一些数据集上工作的很好(例如训练集), 但在另一些数据集(例如测试集)表现不佳, 这就会导致高方差，能够表现好这说明算法表达能力很强，或者说灵活性较强, 而表现不稳定就说明泛化能力较差。

总结起来，依据偏差-方差分解（不用知道是什么）可知，泛化误差是噪声加方差再加上偏差。随着训练程度的增长，模型的表达能力逐渐提高，那么偏差就会逐渐减小，同时模型的不稳定性逐渐提高，那么方差就会逐渐增大，此时泛化误差曲线可能呈碗状曲线，也即从开始很高逐渐降低直到最低点接着又开始上升。因此在模型表达能力足够强时，我们需要在合适的时机停止训练，使其在具备较好的表达能力的同时泛化能力也不差。

## 扰动的稳健性
我们期待“好”的预测模型能在未知的数据上有很好的表现： 经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标。

简单性以较小维度的形式展现，参数的范数也代表了一种有用的简单性度量，简单性的另一个角度是平滑性，即函数不应该对其输入的微小变化敏感。例如，当我们对图像进行分类时，我们预计向像素添加一些随机噪声应该是基本无影响的。 1995年，克里斯托弗·毕晓普证明了 具有输入噪声的训练等价于Tikhonov正则化 [Bishop, 1995]。 这项工作用数学证实了“要求函数光滑”和“要求函数对输入的随机噪声具有适应性”之间的联系。

斯里瓦斯塔瓦等人 [Srivastava et al., 2014] 就如何将毕晓普的想法应用于网络的内部层提出了一个想法： 在训练过程中，他们建议在计算后续层之前向网络的每一层注入噪声。 因为当训练一个有多层的深层网络时，注入噪声只会在输入-输出映射上增强平滑性。

这个想法被称为**暂退法（dropout）**。 **暂退法在前向传播过程中，计算每一内部层的同时注入噪声**，这已经成为训练神经网络的常用技术。 这种方法之所以被称为暂退法，因为我们从表面上看是在训练过程中丢弃（drop out）一些神经元。 **在整个训练过程的每一次迭代中，标准暂退法包括在计算下一层之前将当前层中的一些节点置零**。

在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差。 换言之，每个中间活性值$h$以*暂退概率$p$*由随机变量$p'$替换，如下所示：
$$\begin{split}\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 概率为 } p \\
    \frac{h}{1-p} & \text{ 其他情况}
\end{cases}
\end{aligned}\end{split}$$


## 实践中的暂退法
当我们将暂退法应用到隐藏层，以p的概率将隐藏单元置为零时， 结果**可以看作是一个只包含原始神经元子集的网络**。 如图删除了h2和h5， 因此输出的计算不再依赖于h2或h5，并且它们各自的梯度在执行反向传播时也会消失。 这样，输出层的计算不能过度依赖于任何一个元素。
![](https://zh-v2.d2l.ai/_images/dropout2.svg)

## 从零开始实现
1. 从均匀分布中抽取样本，并实现 dropout_layer 函数， 该函数以dropout的概率丢弃张量输入X中的元素， 如上所述重新缩放剩余部分：将剩余部分除以1.0-dropout。

2. 使用Fashion-MNIST数据集。 我们定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元

3. 定义模型。我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后）， 并且可以为每一层分别设置暂退概率： 常见的技巧是在靠近输入层的地方设置较低的暂退概率。 下面的模型将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5， 并且暂退法只在训练期间有效

4. 对模型进行训练和测试。
![](https://zh-v2.d2l.ai/_images/output_dropout_1110bf_54_0.svg)

## 简洁实现
![](https://zh-v2.d2l.ai/_images/output_dropout_1110bf_78_0.svg)
```py
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 练习
1. 如果更改第一层和第二层的暂退法概率，会发生什么情况？具体地说，如果交换这两个层，会发生什么情况？设计一个实验来回答这些问题，定量描述你的结果，并总结定性的结论。
2. 增加训练轮数，并将使用暂退法和不使用暂退法时获得的结果进行比较。
3. 当应用或不应用暂退法时，每个隐藏层中激活值的方差是多少？绘制一个曲线图，以显示这两个模型的每个隐藏层中激活值的方差是如何随时间变化的。
3. 为什么在测试时通常不使用暂退法？
4. 以本节中的模型为例，比较使用暂退法和权重衰减的效果。如果同时使用暂退法和权重衰减，会发生什么情况？结果是累加的吗？收益是否减少（或者说更糟）？它们互相抵消了吗？
5. 如果我们将暂退法应用到权重矩阵的各个权重，而不是激活值，会发生什么？
6. 发明另一种用于在每一层注入随机噪声的技术，该技术不同于标准的暂退法技术。尝试开发一种在Fashion-MNIST数据集（对于固定架构）上性能优于暂退法的方法。

## 参考代码
dropout_layer
```py
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```
arguments initialization
```py
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```
model definition
```py
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```