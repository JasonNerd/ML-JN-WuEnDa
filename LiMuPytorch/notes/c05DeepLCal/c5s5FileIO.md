# 读写文件
check_point: 检查点, 中断返回点
到目前为止，我们讨论了如何处理数据， 以及如何构建、训练和测试深度学习模型。 然而，有时我们希望保存训练的模型， 以备将来在各种环境中使用（比如在部署中进行预测）。 此外，当运行一个耗时较长的训练过程时， 最佳的做法是定期保存中间结果， 以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。 

## 加载和保存张量
1. 对于**单个张量**，我们可以直接调用load和save函数分别读写它们。
2. 我们可以存储一个**张量列表**，然后把它们读回内存。
3. 我们甚至可以写入或读取从字符串映射到**张量的字典**。 当我们要读取或写入模型中的所有权重时，这很方便。

## 加载和保存模型参数
保存单个权重向量（或其他张量）确实有用， 但是如果我们想保存整个模型，并在以后加载它们， 单独保存每个向量则会变得很麻烦。 毕竟，我们可能有数百个参数散布在各处。 因此，深度学习框架提供了内置函数来保存和加载整个网络。 需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型。 
为了恢复模型，我们实例化了原始多层感知机模型的一个备份。 这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。
由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同。 让我们来验证一下。

## 思考
1. 即使不需要将经过训练的模型部署到不同的设备上，存储模型参数还有什么实际的好处？

2. 假设我们只想复用网络的一部分，以将其合并到不同的网络架构中。比如说，如果你想在一个新的网络中使用之前网络的前两层，你该怎么做？

3. 如何同时保存网络架构和参数？你会对架构加上什么限制？

## 代码
加载和保存张量
```py
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
x2

y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

加载和保存模型（参数）
```py
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
Y_clone = clone(X)
Y_clone == Y

```