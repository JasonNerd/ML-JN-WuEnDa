# Back Propagation
&emsp;&emsp;给出5000张20$\times$20的手写数字灰度图，构造包含1个隐含层的神经网络，使用前向传播算法构造假设函数h，写出代价函数，注意随机初始化参数$\Theta$。随后使用Back Propagation写出梯度函数并对代价函数进行优化。
<br>
&emsp;&emsp;输入层节点数量为特征数20*20=400, 隐含层节点数量定为25, 输出节点数即为类别数10，则神经网络的结构为$(400+1)\times(25+1)\times10$。$x$输入为$5000\times400$, 构造$a^1=[1, x]$, 也即在第一列的位置插入一列1, 

## 试验记录
1. 随机初始化
   ```python
   np.random.rand(a, b) # 0-1间的大小为(a,b)的随机数组
   np.random.randint(a, b, size=(c, d)) # a-b间的大小为(c,d)的随机数组
   ```