## 5.3.2. 小结
延后初始化使框架能够自动推断参数形状，使修改模型架构变得容易，避免了一些常见的错误。

我们可以通过模型传递数据，使框架最终初始化参数。

## 5.3.3. 练习
如果你指定了第一层的输入尺寸，但没有指定后续层的尺寸，会发生什么？是否立即进行初始化？

如果指定了不匹配的维度会发生什么？

如果输入具有不同的维度，你需要做什么？提示：查看参数绑定的相关内容。

## 评论
支持的，使用 torch.nn.LazyLinear，但是PyTorch的这个功能正处于开发阶段，API或功能的变化随时可能发生。
以下给出样例代码
```py
import torch
from torch import nn
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(),nn.Linear(256,10))
print(net)
[net[i].state_dict() for i in range(len(net))]
low = torch.finfo(torch.float32).min/10
high = torch.finfo(torch.float32).max/10
X = torch.zeros([2,20],dtype=torch.float32).uniform_(low, high)
net(X)
print(net)
```

英文版中其实已经补全了pytorch部分的代码，如下：
```py
import torch
from torch import nn

"""延后初始化"""
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# print(net[0].weight)  # 尚未初始化
print(net)

X = torch.rand(2, 20)
net(X)
print(net)
```
Q1: 如果你指定了第一层的输入尺寸，但没有指定后续层的尺寸，会发生什么？是否立即进行初始化？
A1: 可以正常运行。第一层会立即初始化,但其他层同样是直到数据第一次通过模型传递才会初始化(不知道题目理解的对不对)
```py
net = nn.Sequential(
    nn.Linear(20, 256), nn.ReLU(),
    nn.LazyLinear(128), nn.ReLU(),
    nn.LazyLinear(10)
)
print(net[0].weight)
print(net[2].weight)
net(X)
print(net[2].weight)
```
Q2: 如果指定了不匹配的维度会发生什么？
A2: 会由于矩阵乘法的维度不匹配而报错
```py
X = torch.rand(2, 10)
net(X)
```
Q3: 如果输入具有不同的维度，你需要做什么？提示：查看参数绑定的相关内容。
A3: 如果输入维度比指定维度小，可以考虑使用padding填充；如果输入维度比指定维度大，可以考虑用pca等降维方法，将维度降至指定维度。
