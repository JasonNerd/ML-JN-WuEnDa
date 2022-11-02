# 第5章-深度学习计算
看起来这章是要介绍ptrorch中形式化的模型定义及细节理解
## 第一节-层和块
### 神经网络的分层架构
在初次介绍神经网络时，针对单一输出的线性回归模型，有：单个神经网络 
（1）接受一些输入； 
（2）生成相应的标量输出； 
（3）具有一组相关 ***参数(parameters)***，更新这些参数可以优化某目标函数
对于处理分类任务的sofmax模型，仍然保持一个从输入到输出的全相联结构，此时输出是一个向量，随后的多层感知机引入了隐藏层和激活函数以应对复杂的非线性学习，且仍保持全相联的结构。也即：
&emsp;&emsp;***整个模型接受原始输入（特征），生成输出（预测）， 并包含一些参数（所有组成层的参数集合）。 同样，每个单独的层接收输入（由前一层提供）， 生成输出（到下一层的输入），并且具有一组可调参数， 这些参数根据从下一层反向传播的信号进行更新。***

### 神经网络的分块架构
&emsp;&emsp;事实证明，研究讨论“比单个层大”但“比整个模型小”的组件更有价值。例如，在计算机视觉中广泛流行的ResNet-152架构就有数百层， 这些层是由**层组（groups of layers）**的重复模式组成。 在其他的领域，如自然语言处理和语音， **层组以各种重复模式排列**的类似架构现在也是普遍存在。为了实现这些复杂的网络，我们引入了**神经网络块**的概念。 **块（block）可以描述单个层、由多个层组成的组件或整个模型本身**。 使用块进行抽象的一个好处是可以将一些块组合成更大的组件， 这一过程通常是递归的
![](https://zh-v2.d2l.ai/_images/blocks.svg)

### 实现一个MLP块
从编程的角度来看，块由*类（class）*表示。它的任何子类都必须定义一个**将其输入转换为输出的前向传播函数**， 并且必须存储任何必需的参数。最后，为了计算梯度，块必须具有反向传播函数。 在定义我们自己的块时，由于自动微分的实现，我们只需要考虑前向传播函数和必需的参数。
在实现我们自定义块之前，我们简要总结一下每个块必须提供的基本功能
    1. 将输入数据作为其前向传播函数的参数。

    2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。

    3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。

    4. 存储和访问前向传播计算所需的参数。

    5. 根据需要初始化模型参数。
### 顺序块(sequential)
现在我们可以更仔细地看看Sequential类是如何工作的， 回想一下Sequential的设计是为了把其他模块串起来。 为了构建我们自己的简化的MySequential， 我们只需要定义两个关键函数：
1. 一种将块逐个追加到列表中的函数。
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

### 在前向传播函数中执行代码
Sequential类使模型构造变得简单， 允许我们组合新的架构，而不必定义自己的类。 然而，并不是所有的架构都是简单的顺序架构。 当需要更强的灵活性时，我们需要定义自己的块。 例如，我们可能希望在前向传播函数中执行Python的控制流。 此外，我们可能希望执行任意的数学运算， 而不是简单地依赖预定义的神经网络层。
我们可以混合搭配各种组合块的方法。 在下面的例子中，我们以一些想到的方法嵌套块

### 你可能会开始担心操作效率的问题。
 毕竟，我们在一个高性能的深度学习库中进行了大量的字典查找、 代码执行和许多其他的Python代码。 Python的问题全局解释器锁 是众所周知的。 在深度学习环境中，我们担心速度极快的GPU可能要等到CPU运行Python代码后才能运行另一个作业。

### 5.1.5. 小结
* 一个块可以由许多层组成；一个块可以由许多块组成。
* 块可以包含代码。
* 块负责大量的内部处理，包括参数初始化和反向传播。
* 层和块的顺序连接由Sequential块处理。

### 5.1.6. 练习
* 如果将MySequential中存储块的方式更改为Python列表，会出现什么样的问题？
* 实现一个块，它以两个块为参数，例如net1和net2，并返回前向传播中两个网络的串联输出。这也被称为平行块。
* 假设你想要连接同一网络的多个实例。实现一个函数，该函数生成同一个块的多个实例，并在此基础上构建更大的网络。


### 参考代码
下面的代码生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
```py
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```
在下面的代码片段中，我们从零开始编写一个块。 它包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。
```py
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```
Sequential的设计
```py
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```
此操作可能不会常用于在任何实际任务中， 我们只是向你展示如何将任意代码集成到神经网络计算的流程中
```py
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

```
我们可以混合搭配各种组合块的方法。 在下面的例子中，我们以一些想到的方法嵌套块
```py
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```