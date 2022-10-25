# Section01 - Chapter03
> Linear Regression Model
> Softmax Regression
etc.
本文件主要记录阅读这一章遇到的问题

### [yield是什么？](https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-scratch.html#id3)
[python中yield的用法详解——最简单，最清晰的解释](https://blog.csdn.net/mieleizhi0522/article/details/82142856)

首先，如果你还没有对yield有个初步分认识，那么你先把yield看做“return”，这个是直观的，它首先是个return，普通的return是什么意思，就是在程序中返回某个值，返回之后程序就不再往下运行了。看做return之后再把它看做一个是生成器（generator）的一部分（带yield的函数才是真正的迭代器）
带yield的函数是一个生成器，而不是一个函数了，这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数，这一次的next开始的地方是接着上一次的next停止的地方执行的，所以调用next的时候，生成器并不会从foo函数的开始执行，只是接着上一步停止的地方开始，然后遇到yield后，return出要生成的数，此步就结束

### [为什么在squared_loss函数中需要使用reshape函数？](https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-scratch.html#id6)

似乎不用也没问题 $乐:>$


### [with torch.no_grad()](https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-scratch.html#id7)
[【pytorch系列】 with torch.no_grad():用法详解](https://blog.csdn.net/sazass/article/details/116668755)
```py
x = torch.randn(10, 5, requires_grad = True)
y = torch.randn(10, 5, requires_grad = True)
z = torch.randn(10, 5, requires_grad = True)
with torch.no_grad():
    w = x + y + z
    print(w.requires_grad)
    print(w.grad_fn)
print(w.requires_grad)


False
None
False

```

### [from torch.utils import data](https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-concise.html#id2)

[TensorDataset](https://blog.csdn.net/anshiquanshu/article/details/109398797)
```py
 
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
 
a = torch.tensor([[11, 22, 33], [44, 55, 66], [77, 88, 99], [11, 22, 33], [44, 55, 66], [77, 88, 99], [11, 22, 33], [44, 55, 66], [77, 88, 99], [11, 22, 33], [44, 55, 66], [77, 88, 99]])
b = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
train_ids = TensorDataset(a, b) 
# 切片输出
print(train_ids[0:2])
print('#' * 30)
# 循环取数据
for x_train, y_label in train_ids:
    print(x_train, y_label)
# DataLoader进行数据封装
print('#' * 30)
train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
for i, data in enumerate(train_loader, 1):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))   # y data (torch tensor)
```

### [nn.Sequential](https://blog.csdn.net/xddwz/article/details/90704950)

torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。通俗的话说，就是根据自己的需求，把不同的函数组合成一个（小的）模块使用或者把组合的模块添加到自己的网络中。

```
# 第一种方法
conv_module = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
 
# 具体的使用方法
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_module = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
    def forward(self, input):
        out = self.conv_module(input)
        return out
```

### torch.optim模块
[torch.optim.SGD参数详解（除nesterov）](https://blog.csdn.net/weixin_46221946/article/details/122644487)
params：要训练的参数，一般我们传入的都是model.parameters()。
lr：learning_rate学习率，会梯度下降的应该都知道学习率吧，也就是步长。
weight_decay正则化系数（权重衰退）是在L2正则化理论中出现的概念

如何查看optim模块]()

```py
#model define...
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
"""
Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
"""
optimizer=optim.Adam(model.parameters(),0.01)
print("optimizer.param_groups的长度:{}".format(len(optimizer.param_groups)))
for param_group in optimizer.param_groups:
    print(param_group.keys())
    print([type(value) for value in param_group.values()])
    print('查看学习率: ',param_group['lr'])
"""
optimizer.param_groups的长度:1
dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
[<class 'list'>, <class 'float'>, <class 'tuple'>, <class 'float'>, <class 'int'>, <class 'bool'>]
查看学习率:  0.01
"""
```

### nn.MSELoss()
返回一个函数指针，例如`loss = nn.MSELoss()`，随后调用`loss(y, y_hat)`即可得到均方误差

### Python-import导入上级目录文件
[Python-import导入上级目录文件](https://zhuanlan.zhihu.com/p/64893308)
导入下级目录模块也很容易，需在下级目录中新建一个空白的__init__.py文件再导入：

from dirname import xxx
如在file1.py中想导入dir3下的file3.py，首先要在dir3中新建一个空白的__init__.py文件。
```git
-- dir0
　　| file1.py
　　| file2.py
　　| dir3
　　　| __init__.py
　　　| file3.py
　　| dir4
　　　| file4.py
```
要导入上级目录下模块，可以使用sys.path： 　
```py
import sys 
sys.path.append("..") 
import xxx　
```

### ewew
wewew



### ewew
wewew


