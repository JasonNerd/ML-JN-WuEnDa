# Neural Network(神经网络)
题外话(verbose)：
* 关于vscode与github的互动, 我起初的想法是:
  1. 我在本地写代码，写着写着忽然觉得有必要为他创建一个github仓库来管理代码。此时我已经安装了git命令工具，vscode的源代码管理也有对应的图形化界面，先认证，然后提交和推送到远程仓库，这里我想让他自己把文件夹作为仓库直接推上去（新建），不可行
  2. 因此，你需要一开始就在github上新建一个仓库，再克隆下来，接着使用vscode编辑你的代码、笔记等等文件，之后再进行更多的操作。具体做法参考：[Windows+VScode配置与使用git，超详细教程，赶紧收藏吧](https://blog.csdn.net/czjl6886/article/details/122129576)
* github时不时就打不开 DNS污染
  [告别无法访问的github（附解决方案）](https://cloud.tencent.com/developer/article/1904883)

### 缘起
在前面关于线性回归模型和对率回归模型的讨论中，我们为了使模型能够适应更多的情景，一方面对于特征进行转化，例如使用多项式，另一方面对于标记值进行转换等等。在多项式模型中，当幂次过高时会出现过拟合现象，一种可行的办法是使用正则化来缓解过拟合现象。而神经网络则是通过将对率回归模型进行嵌套以提高表达能力，它从生物学神经的角度出发，使用一个对率模型作为单个的神经元，神经元之间使用权值矩阵进行连接从而形成神经网络

### 神经网络
* **神经元-（MP神经元模型）**
  ![](https://files.mdnice.com/user/35698/e7b9d9ed-4c65-4482-8cd3-9736af369472.png)
  如图所示为MP神经元模型，它包括一个输入$x_i(i=1\rightarrow n)$，一系列的权值$\omega_i(i=1\rightarrow n)$(预置参数、待学习)，接着把它们对应相乘($\omega^Tx$)得到输入，神经元的处理是将其与一个阈值比较接着套上一个转化函数(例如sigmoid函数或者阶跃函数等)，函数值即为输出值$y$.
  因此：
  $$
  y=sigmoid(w_1x_1+w_2x_2+\dots+w_nx_n-\theta)
  $$
  如果使用熟悉的记号
  $$
  \bold{\theta} = [\theta_0, \theta_1, \dots, \theta_n]_{(n+1)\times1} \\
  x_0=1 \\
  x=[x_0, x_1, \dots, x_n]_{(n+1)\times1}\\
  so\enspace that, \\
  y=sigmoid(\theta^Tx)
  $$
  那么这将与对率函数的假设函数相同
* **感知机(Perceptron)**
  感知 (Perceptron 由两层神经元组成:
  ![](https://files.mdnice.com/user/35698/7c99728e-217c-4383-ac3b-bd797b5d326c.png)
  $y=w_1x_1+w_2x_2-\theta$, 能够进行线性的划分，例如与或非运算，
  * AND, $only\enspace when\enspace x_1=x_2=1\rightarrow y=1$
    ![](https://files.mdnice.com/user/35698/f0efad5f-5552-4ba5-ba7d-9c66e4b0eac4.png)
    $w_1=1, w_2=1, \theta=-1.5$
  * OR, $only\enspace when\enspace x_1=x_2=0\rightarrow y=0$
  ![](https://files.mdnice.com/user/35698/76295f0f-11d5-45b7-a2f5-3bace948af0d.png)
  $w_1=1, w_2=1, \theta=-0.5$
  * NOT
  ![](https://files.mdnice.com/user/35698/3e601006-e1b4-474d-985c-dcb684743c39.png)
  $w_1=-1, w_2=0, \theta=0.5$
  那么XOR异或问题呢？先看图示
  ![](https://files.mdnice.com/user/35698/bd418492-e504-43c4-8b57-9fc88996d106.png)
  可见不是一个线性分类问题，那么仅使用一层是不行的，考虑$x_1\otimes x_2 = (\lnot x_1\land x_2)\lor(x_1\land\lnot x_2)$，可以将其分成两层，先是输入层$x_1,x_2$, 接下来一层是两个神经元，分别计算两个括号值，再有一层含一个神经元求并值。
  ![](https://files.mdnice.com/user/35698/49465e3c-bebb-40df-ada5-9c5f67a88c3b.png)
---
* **神经网络**
  * 实际上，上面的实现异或这一非线性划分的两层感知机可以视为一个具备完整结构的神经网络，它包含一个输入层(input layer)，计算中间值的隐含层(hidden layer)以及一个输出层(output layer)，同时输出值只有一个，因而是个二分类问题。
  * 一个典型的神经网络结构如下图所示：
  ![](https://files.mdnice.com/user/35698/79339480-9e90-4443-95ab-86564029a585.png)
  由图可知，蓝色部分为输入层，含有3个特征，有两个隐含层，一个输出层，输出是一个四维向量，因此是一个多分类问题，且包含四个类别。其中$h_{\Theta}(x)$是形如$[0,1,0,0]$的向量（该实例表示分类结果是第二类）。
  接下来将对神经网络进行更为详细的探讨，首先是一些记号表示：
    数据集: $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})\}$
    1. $L$: 神经网络的层数，例如上图中$L=4$。
    2. $S_l$: 第$l$层的神经单元数, 例如上图中$S_L=4$, 也即输出层包含4个结点(注意不考虑偏置节点)。
    3. $\Theta^{(l)}$: 从第$l$层连接到$l+1$层的权值矩阵。
    4. $a^{(l)}_j$: 表示第$l$层第j个节点的输出值，而$a^{(l)}$则是一个$S_l+1$维的向量, 作为$l+1$层的输入, 其中$a^{l}_0=1$是偏置节点(起到阈值的效果).又例如$a^{(1)}_1=x_1$表示第一层第一个结点值(下标从0开始)。
    5. $z^{(l)}_j$: 表示$a^{(l)}_j$的输入，此处$l>=2, j>0$. 因此$z^{(l)}$是一个$S_l$维的向量。
  * 接下来是实例探讨，如图：
    ![](https://files.mdnice.com/user/35698/0cc01571-f1de-44ff-bb97-e7653ed8e441.png)
    例如，计算$a_2^{(2)}$有:
    $$z_2^{(2)}=\Theta^{(1)}_{20}x_0+\Theta^{(1)}_{21}x_1+\Theta^{(1)}_{22}x_2+\Theta^{(1)}_{23}x_3\\a_2^{(2)}=sigmoid(z_z^{(2)})$$
    则权值矩阵$\Theta^{(1)}$的形状是3x4, 据此$\Theta^{(l)}$的形状是$S_{l+1}\times(S_l+1)$
    将其向量化, $z^{(2)}=[z^{(2)}_1, z^{(2)}_2, z^{(2)}_3]$以及$x=[x_0, x_1, x_2, x_3]$并假设$z^{(2)}$和$x$均为列向量：
    $$
    z^{(2)}=\Theta^{(1)}x \\
    a^{(2)}=[a^{(2)}_0, sigmoid(z^{(2)})], a^{(2)}_0=1
    $$
    据此计算出$a^{(3)} = sigmoid(\Theta^{(2)}a^{(2)})=h_\Theta(x)$

  * 代价函数
    以上得到了假设函数$h_\Theta(x)$, 接下来是代价函数的书写，仍假设有$m$条样本数据，输出类别数为$K$。则依据直观思想，代价函数为这$K$个类别的代价之和，其形式类似于对率回归的代价函数，如下:
    $$
      J_\Theta=-\frac{1}{m}\sum^m_{i=1}\sum^K_{k=1}\left[y^{(i)}_k\ln(h_\Theta(x^{(i)})_k)+(1-y^{(i)}_k)\ln(1-h_\Theta(x^{(i)})_k)\right]\\+\frac{\lambda}{2m}\sum^{L-1}_{l=1}\sum^{S_{l+1}}_{i=1}\sum^{S_l}_{j=1}\Theta^{(l)}_{ij}
    $$
    其中$h_\Theta(x^{(i)})_k$表示$K$维向量假设函数输出的第$k$个值
  * <font color = #A0A>梯度</font>
    梯度的求解似乎很麻烦呢！不过也会一步一步的推导呢！这里的方法主要是BP算法(Backward Propagation)，接下来将一步步的说明
    误差$\delta_j^{(l)}$表示第$l$层第$j$个节点的误差值, $\delta^{(l)}$则是第$l$层的误差向量
    ![](https://files.mdnice.com/user/35698/a1a69877-6592-46a5-ad54-b304218ca579.png)
    从第4层算起:
    $$\begin{align*}
        & \delta^{(4)} = h_\Theta(x)-y \\
        & \delta^{(3)} = (\Theta^{(3)})^T\delta^{(4)}\cdot g'(z^{(3)}) \\
        & =(\Theta^{(3)})^T\delta^{(4)}\cdot g(z^{(3)})(1-g(z^{(3)})) \\
        & = (\Theta^{(3)})^T\delta^{(4)}\cdot a^{(3)}\cdot (1-a^{(3)}) \\
        & \delta^{(2)} = (\Theta^{(2)})^T\delta^{(3)}\cdot a^{(3)}\cdot (1-a^{(2)}) \\
    \end{align*}
    $$
  * 有了误差计算后，就可以完整的实现一个BP算法了，给定训练数据集$\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})\}$
  令$\Delta_{ij}^{(l)}=0$
  ![](https://files.mdnice.com/user/35698/9aae86be-411b-47f7-91bf-1036625f2912.png)
  ![](https://files.mdnice.com/user/35698/2cbbe77c-5dd3-4ae6-86ca-39bca3f80ed0.png)
  ![](https://files.mdnice.com/user/35698/51601ecd-33bd-4f39-ae6f-a6d98f74d427.png)
---
* 完整的神经网络训练过程
    1. 选择合适的网络结构。Pick a network architecture (connectivity pattern between neurons)一般来说，只包含一个隐含层的神经网络是最常见的，如果选择包含多个隐含层的网络，各隐含层的节点数应当相当相等。
    2. 对$\Theta$随机初始化，不可以初始化为0。Randomly initialize weights. 
    3. Implement forward propagation。实现前向传播计算$h_\Theta(x)$
    4. Implement code to compute cost function $J(\Theta)$实现代价函数
    5. Implement backprop to compute partial derivatives。通过BP算法计算代价函数对各个$\theta$的偏导数.
    6. 梯度检测。Use gradient checking to compare computed using backpropagation VS using  numerical estimation. Then disable gradient checking code. ($estimation=\frac{J_{\theta+\epsilon}-J_{\theta-\epsilon}}{2\epsilon}$)
    7. 对$J(\Theta)$进行优化。Use gradient descent or advanced optimization method with backpropagation to try to  minimize $J(\Theta)$

---
### 实验三、XXXX
---
### 问题记录
* 如何设置markdown的字体颜色
    ```html
    <font color=#ABC>something</font>
    ```
    * 示例
    <font color=CornflowerBlue>CornflowerBlue</font>
    <font color=Chartreuse>Chartreuse</font>
    <font color=DarkRed>DarkRed</font>
    <font color=GoldenRod>GoldenRod</font>
    <font color=#A0A>A0A</font>
    [<font color = #A0A>【Markdown笔记】设置字体颜色</font>](https://blog.csdn.net/u012028275/article/details/115445362)
---
* LaTex数学公式中的空格与换行
  [<font color = #A0A>Latex 中的空格汇总</font>](https://blog.csdn.net/hysterisis/article/details/114123131)
  如下分别是换行和 空格（最常用的一种）
  ```markdown
    \\
    \enspace
  ```