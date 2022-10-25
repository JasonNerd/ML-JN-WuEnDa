# Multiple Layer Perceptron (多层感知机)
## 从线性模型到非线性模型
&emsp;&emsp;在softmax模型中，我们通过一个全连接层外接一个偏置将输入映射到输出，这个全连接层是线性的，或者说仿射变换， 它是一种带有偏置项的线性变换。然而实际问题并不总是这样，例如，我们想要根据体温预测死亡率。 对于体温高于37摄氏度的人来说，温度越高风险越大。 然而，对于体温低于37摄氏度的人来说，温度越高风险就越低。 在这种情况下，我们也可以通过一些巧妙的预处理来解决问题。 例如，我们可以使用与37摄氏度的距离作为特征。

## 在网络中加入隐藏层
&emsp;&emsp;一个可能的做法是将许多全连接层堆叠在一起。 每一层都输出到上面的层，直到生成最后的输出。这种架构通常称为多层感知机（multilayer perceptron），通常缩写为MLP。

![](https://zh-v2.d2l.ai/_images/mlp.svg)

&emsp;&emsp;这个多层感知机有4个输入，3个输出，其隐藏层包含5个隐藏单元。 输入层不涉及任何计算，因此使用此网络产生输出只需要实现隐藏层和输出层的计算。 因此，这个多层感知机中的层数为2。 注意，这两个层都是全连接的。接下来是各个矩阵的辨析，首先是小批量样本，它包含n个样本，每一个样本由d维的特征向量描述，即$\mathbf{X} \in \mathbb{R}^{n \times d}$, 其次是隐含层\隐藏层变量$\mathbf{H} \in \mathbb{R}^{n \times h}$，也称为隐藏表示(hidden representations)，它是由输入$\mathbf{X} \in \mathbb{R}^{n \times d}$乘以隐藏层权重$\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$加上隐藏层偏置$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$得到的。最后是单隐藏层多层感知机的输出$\mathbf{O} \in \mathbb{R}^{n \times q}$, 它是由隐藏层变量$\mathbf{H} \in \mathbb{R}^{n \times h}$输出层权重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$加上输出层偏置$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$得到的。$$\begin{split}\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}\end{split}$$

&emsp;&emsp;这样做确实增加了模型复杂度，但目前它仍然是线性的变换。
$$\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.$$

如果从感知机的角度讲，每一个节点都是感知节点，偏置视为阈值，对于仿射变换应该再外套一层非线性函数，称之为激活函数（activation function）。 激活函数$\sigma$的输出被称为活性值（activations）

$$\begin{split}\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}\end{split}$$
我们可以继续堆叠这样的隐藏层， 例如$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$和$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$

## 激活函数 (activation function)
计算加权和并加上偏置来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算
### ReLU函数(Rectified linear unit，ReLU)
$$\operatorname{ReLU}(x) = \max(x, 0).$$
ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素
![](https://zh-v2.d2l.ai/_images/output_mlp_76f463_18_0.svg)
使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题（稍后将详细介绍）。
注意，ReLU函数有许多变体，包括**参数化ReLU（Parameterized ReLU，pReLU） 函数 [He et al., 2015]**。 该变体为ReLU添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过：
$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### sigmoid函数
sigmoid函数将输入变换为区间(0, 1)上的输出
$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$
![](https://zh-v2.d2l.ai/_images/output_mlp_76f463_42_0.svg)
$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

### 双曲正切函数
与sigmoid函数类似， tanh(双曲正切)函数能将其输入压缩转换到区间(-1, 1)上。 注意，当输入在0附近时，tanh函数接近线性变换。 函数的形状类似于sigmoid函数， 不同的是tanh函数关于坐标系原点中心对称.
![](https://zh-v2.d2l.ai/_images/output_mlp_76f463_66_0.svg)

## ToDo List
绘制各种激活函数的图像
```py
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

```