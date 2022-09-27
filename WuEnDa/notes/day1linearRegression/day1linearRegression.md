# 第一章-机器学习概论
### 引言
  机器学习($Machine$ $Learning$)这一学科诞生于人工智能的中，它是计算机的一种新的能力，不同于通常的任务，例如编写程序实现一个文件管理系统，或许你需要分层的分模块的去实现它，但它总还是可以使用确定的程序逻辑去表达的，机器学习面对的任务通常是需要收集分析大量来自生活（自然、社会）中的数据，换句话说，机器学习就是用数据编程。例如给一张图片，人依据经验能够很快判断出图中是否有一只猫，那么我们能否按照通常的业务逻辑处理思想解决它呢？例如我们把图像的每一个像素点的信息（X、Y坐标、颜色的RGB表示）作为程序输入，这个算法要怎么设计以得到Y或者N这样肯定的判断呢？例如分块看像素平均值？最大值？等等等等，这些方法似乎都不太可行。于是不妨以结果为导向，给出许多许多的照片，这些照片中有的有猫，有的则没有，给出这些已知的结果，让机器去学习它们，从而在面对新的照片时能够一定程度上给出它的答案

### 机器学习在生活中的应用
* 数据挖掘($Database$ $mining$)
        例如: 网络点击数据（推荐算法、用户个性化推荐），医疗记录（汇编成医疗大数据，辅助诊断治疗），生物学（基因编码、相似基因片段寻找等），工程应用
* 一些无法编程的应用
        例如，自动驾驶（小汽车、无人机）、手写体识别（自动信箱投递）、自然语言处理(NLP)、计算机视觉(CV)
* ... ...

### 什么是机器学习
* 它是一门致力于研究如何通过计算的手段，利用经验（数据）来提升自身的性能的学科, 研究的主要内容是从数据中产生 `模型(model)` 的算法，也即 `学习算法(Learning Algorithm)`
* 机器学习并没有一个严谨的定义，一个可能地说法是
  * A computer program is said to *learn from* experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
  * 例如假设你的邮件系统有过滤垃圾邮件的功能，那么识别一封邮件是否为垃圾邮件就是任务T，经验E就是哪些被你归为垃圾邮件，度量P可以是正确分类的邮件数量

### 术语表述
* 给一个例子，判断 breast cancer 是 malignant 或者 benign, 假定影响肿瘤是否为良性的依据是肿瘤的大小(Tumor size), 并且有一系列已知的数据例如`[{size:3, condition: benign}, {size:7, condition: malignant}, ... ...]`那么据此进行学习，其目标是区分肿瘤是良性(0)或者恶性(1)，这样的任务称为`分类学习(Classification)`，也即预测值为离散值，特别的，这里只预测是或者不是，即为二分类学习。
* 与此同时，若预测值为连续值，称之为 `回归学习(Regression)` 例如，基于房屋面积(HouseArea)来预测房屋售价，同样的，这里也会有一些已知的销售数据。
* 观察以上两个例子, 对于其输入数据例如`[{size:3, condition: benign}, {size:7, condition: malignant}, ... ...]`, 可以看到其结果都是已知的，于是`Classification`以及`Regression`合称为 `监督学习(Supervised Leaning)`
* 对于以上几段话, 有一些概念需要明确, 以肿瘤良性或恶性的预测判断为例, 假定还有一个影响因素是年龄(Age), 将已有的数据用表格展示如下：
  
    |size|age|benign|
    |:-:|:-:|:-:|
    |3|43|N|
    |5|28|Y|
    |4|32|N|
    |8|30|Y|
    |6|45|N|

    * 整个表格的内容称为 数据集(dataset), 其中用于训练的子集称为 训练集(trainning set), 数据集由一个个条目组成, 每一个条目称为 样本(sample) 或 实例(instance/example), 对于每一个 实例, 有多个 属性(attribute)/特征(feature) 进行描述, 例如这里是两(d)个, 分别是 size age, 称这个样本是二(d)维的, 这样, 这一组属性就张成了一个 属性空间, 这里属性空间是二维的, 同时对于每一组属性都有一个对应值Y or N, 称为属性值 或者 标记, 因此 实例 对应于属性空间的一个点或者向量, 又被称为 特征向量. 另外, 再看 数据集大小 即为数据集包含的实例数目, 这里是5
  
* 与$supervised$ $learning$相对应的是$unsupervised$ $learning$, 即无监督学习
* 任务会包含一系列的输入数据, 它们并没有事先标记, 任务目标是找到数据的内在联系或者结构并进行划分, 据此可能得到一些新的特征, 这样的任务称为 聚类分析(clustering analysis).典型的聚类问题例如
    1. 主题搜索, 将网页上的新闻关于同一个主题的归到一个组
    2. 基因组中特定基因的表达程度
    3. 社交网络分析, 比如判断哪些人是在一个圈子里
    4. 客户市场细分(market sementation), 将客户划分进更细微的市场, 以便针对性的实施对策
    5. ... ...
有一个有趣的例子, 称为鸡尾酒算法, 有两个人同时讲话, 与此同时有两个麦克风进行录音, 那么是否能依据两个输入把两个人声分离
又或者, 有一段bgm和一个人的诗朗诵, 能否进行分离
这是无监督学习的另一个例子

下一节我们将以一个简单的例子 单变量线性回归($univariate$ $linear$ $regression$) 来讨论一个机器学习问题的实际处理过程, 与此同时讨论这一过程中的细节问题

## 单变量线性回归(Linear Regression with One Variable)
* 机器学习的核心是学习算法$learning$ $algorithm$。在监督学习任务中, 学习算法的工作是确定一个从数据集(dataset)到目标值的映射, 使得当得到一个新的样本时, 依据这样一个映射(模型)能够得到一个估计值, 并且希望这个估计值尽可能与真实值符合。
* 这里谈到了许多的问题, 学习算法的目标是依据数据生成模型, 或者一个映射, 这样的映射可能有多种形式, 可能很简单例如线性函数(称为假设函数$hypothesis$), 或者很复杂例如神经网络. 另外, 注意到评价一个算法或者模型的好坏总是离不开真实数据集的, 脱离数据集去谈论学习算法谁更“聪明”是没有意义的, 因此, 学会依据实际数据集去选择合适的算法或者改进乃至于创造一个全新的算法是一个机器学习工作者最宝贵的特质, 所谓经典套路要学懂, 但也必须随机应变.
* 另外, 在确定一个模型, 或者说确定各个参数时, 它们总是在训练集上进行的, 对于模型的评判不能靠训练集上的准确度来衡量, 而是应该在新的数据集上进行测试, 也即应当在测试集上也工作得很好.这样一种能力称之为 泛化能力(generalization)另外一个问题是假设没有更多的数据集的话, 可以对原始数据集进行划分, 有多种划分方法.
* 最后针对估计值与真实值符合程度的问题, 也就是模型性能的度量, 对于不同问题有不同的考虑, 例如准确率、查准率、查全率等等

### 房价预测问题
问题总结起来是利用特征的线性组合(h)去尽可能的逼近标记值(y), 使用均方误差刻画逼近的程度, 称为代价函数(J), 目标是使J最小, 方法之一是利用梯度下降算法(Gradient Descent), 它的核心是使参数(自变量)沿着导数(绝对值)减小的方向不断移动, 最终收敛于某个最小值(极小值)
* 标记(mark) or y 是 房价 price
* 样本包含一个 特征(feature) 也即 面积x(area)
* 假设函数(hypothesis)是一个一次函数
  * $h_\theta(x) = \theta_0+\theta_1*x$
* 使用 均方差(mean square) 作为 代价函数(cost function), 对应于上述 "估计值与真实值符合程度" 的问
* 目标是不断的改变$\theta_0$和$\theta_1$的值, 使得cost function 最小
![](http://www.ai-start.com/ml2014/images/10ba90df2ada721cf1850ab668204dc9.png)

$x^{(i)}$表示数据集中第i个实例的属性部分, $y^{(i)}$为值

一种可行的方法是 梯度下降算法(gradient descent), 对于损失函数, 它的自变量是$\theta_0$和$\theta_1$, 设定一个$\theta_0$和$\theta_1$的初值, 并使它们向着导数减小的方向同步改变
![](http://www.ai-start.com/ml2014/images/7da5a5f635b1eb552618556f1b4aac1a.png)
* 其中是$\alpha$是 学习率 $learning$ $rate$, 它控制着更新幅度大小, $\alpha$不能设置的过大, 这样可能导致算法发散, 也不能太小以至于算法收敛过慢.
* *Gradient descent can converge to a local minimum, even with the $\alpha$ fixed.* 也即学习速率没必要手动减小, 因为随着收敛过程的进行, 微分项会自动的减小

至此, $\theta$的更新过程变为:
![](https://files.mdnice.com/user/35698/ad922c1c-e742-4d0f-81ed-785134b3aa62.png)
事实上, 这样一个过程又称为 批量梯度下降(batch gd)，指的是在梯度下降的每一步中，我们都用到了所有的训练样本, 也即求和

### 具体实现(jupyter notebook)
* 代价函数$J_{(\theta_0, \theta_1)}$的计算
  我们应该使用 向量(vector) 或者 矩阵(matrix) 来统一的对数据进行操作, 而不是使用`for-while-loop`自己去遍历它们。设 $\bold{\theta}$ = $[\theta_0, \theta_1]_{(1\times2)}$, X = $[1_{m\times1}, \vec{x}_{(m\times1)}]_{(m\times2)}$, y = $[y_1, y_2, \dots, y_m]_{(m\times1)}$
  * 因此, $h(X) = X*\theta^T, J_{(\theta_0, \theta_1)} = \frac{innnerSum([h(X)-y]^2)}{2*m}$
  * 其中, $h(X)$以及$y$均为m行的列向量, 则他们的差也是m维的列向量, 这里的平方是指对向量的每一个元素进行平方, innerSum是指将向量中的每一个元素进行求和。代码实现如下
  ```python
  def cost(X, y, theta):
    """
    X, y, theta均为np.matrix类型
    """
    m = len(y)
    err = X*theta.T-y
    return np.sum(np.power(err, 2))/(2*m)
  ```
* 梯度下降算法(gradient descent)
基本的公式形式为：$\theta_j \leftarrow \theta_j - \alpha*\Delta_j$
其中$\Delta_j$是$J_{(\theta)}$对$\theta_j$的偏导数, 代码的实现关键也就在这里, 如何以一个向量的形式去实现它呢？针对$\theta_0$, 有:
  ```python
  theta_0 = theta_0 - alpha * (np.sum(err)/m)
  # 其中的err, m等变量与前述一致
  ```
  针对$\theta_1$, 多了一个$x_i$项(这里的$i$是从行看的, $i$的范围从1到m, 例如在房价预测中, $x_i$指第$i$个房子的面积), 考虑抽出$X = [1_{m\times1}, \vec{x}_{(m\times1)}]_{(m\times2)}$包含特征的第一列$\vec{x}$并记为$x_1$(换名为$x_1$是方便后续多变量回归分析的讨论)。
  因此有:
  $\Delta_1 = \frac{1}{m}\cdot{(x_1^T\times err)}$
  $x_1^T$是一个m维的行向量, 而err为m维列向量, 这个乘法是数学意义上的向量乘法, 结果是一个实数, 恰好表达了$\theta_1$的更新过程
  ```python
  delta = np.multiply(err, X[:,1])
  theta_1 = theta_1 - alpha*(1/m)*delta
  ```

### 完整的代码实现
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. 查看原始数据的情况
data = pd.read_csv('./ex1data1.txt', header=None, names=['population', 'profit'])
data.head() # 前几行
data.info() # 97x2(float)
data.plot(kind='scatter', c='red', marker='x', x='population', y='profit')
plt.show()

# 2. 构造矩阵X, y, theta
data.insert(0, 'ones', 1) 
n = data.shape[1]
X = np.matrix(data.iloc[:,0:n-1]) # 所有的行, 取前n-1列
y = np.matrix(data.iloc[:,n-1:n]) # 所有的行, 取最后一列
theta = np.matrix(np.zeros(n-1)) # 初值为0, 0
# X.head(), y.head() # 查看划分是否正确
# X.shape, y.shape, theta.shape

# 3. 构造costFunc(X, y, theta)
def costFunc(X, y, theta):
    m = len(y)
    err = X*theta.T-y
    return np.sum(np.power(err, 2))/(2*m)
# 4. 初始化 迭代次数, 学习速率alpha, 进行梯度下降
def gradientDecent(X, y, theta, alpha, iter_n):
    m = len(y)
    cost = np.zeros(iter_n)
    for i in range(iter_n):
        err = X*theta.T-y
        delta = (X.T*err).T/m
        theta = theta - alpha*delta
        cost[i] = costFunc(X, y, theta)
    return theta, cost
iter_n = 300
alpha = 0.018
theta, cost = gradientDecent(X, y, theta, alpha, iter_n)

# 5. 作出拟合曲线, 作出 代价-迭代次数曲线
fig, ax = plt.subplots() # 绘制多种曲线
ax.scatter(data.population, data.profit, c='red', marker='x') # 注意操作 .
x = np.linspace(data.population.min(), data.population.max(), 100) # 注意操作 linspace
h = theta[0,0]+theta[0,1]*x
ax.plot(x, h)
plt.show()
plt.plot(range(iter_n), cost) 
# 可以截取cost的不同部分查看, 例如前10次, 实际上在3到4次的时候就变得很小了（相对）
# 此后减小的速度大大放缓
plt.show()
```

### 多变量回归分析
* 多变量线性回归, 此时特征不止一个了, 例如对于房价预测, 除了面积因素, 还可以考虑 房屋使用年数、卧室数量等等, 设共有$n$个特征, 仍然假设数据集包含m个样本。仍然记$x^{(i)}$为数据集中第$i$个训练实例, $x^{(i)}$是一个n(在下面公式中, 添加了$x_0=1$, 成为$n+1$维)维行向量, 对应第$j$个特征的值为$x^{(i)}_j$, $x_j$是一个m维列向量, 由m个样本中第j个特征的值组成.
* $h$和$J$以及$\Delta_j$的讨论
  $h(x^{(i)}) = \theta_0+\theta_1*x^{(i)}_1+ ... +\theta_n*x^{(i)}_n$
  $X = [x^{(1)}, x^{(2)}, \dots, x^{(m)}]^T = [1, x_1, x_2, \dots, x_n]_{m\times(n+1)}$
  $\theta = [\theta_0, \theta_1, \dots, \theta_n]_{1\times(n+1)}$
  $h(X) = X\theta^T_{m\times1}$
  $J(\theta) = \frac{1}{2m}\cdot\sum^{m}_{i=1}(h(x^{(i)})-y^{(i)})^2$
  $\Delta_j = \frac{1}{m}\cdot(x_j^T\times(X\theta^T-y))$
* 经常见到这样的情况, 也即各个特征的取值范围相差很大, 因此需要$feature$ $scaling$也即特征缩放, 一般的取$\frac{x-\mu}{\delta}$
* $\alpha$学习率的选取, 通常可以这样尝试
  $\alpha = 0.01, 0.03, 0.1, 0.3, 1, 3, 10, ....$
* 特征和多项式回归
  * 特征的转换 
  有时, 对于给出的一些特征, 需要对其作出一定的变换来得到更优的模型, 例如预测房屋价格的问题中, 假设给出的特征是 $frontage$ = 房屋临街宽度, $depth$ = 房屋侧边长度.可以直接构造假设函数为:
  $h(x) = \theta_0 + \theta_1* frontage + \theta_2* depth$
  但另一方面看, 房屋价格更可能与 特征$area = {frontage\times depth}$ 也即房屋面积相关. $area$或许是一个更好的特征.因此有时需要依据实际数据选择合适的特征
  * 据此, 提出多项式回归, 例如假设函数的形式是:
  $h(x;\theta) = \theta_0 + \theta_1*x + \theta_2*x^2$是一个二次曲线, 那么作变换 $x_1 = x, x_2 = x^2$就有$f(x;\theta) = \theta_0 + \theta_1*x_1 + \theta_2*x_2$
  * 此外, 多项式可以拟合任意曲线, 除了常数和整数次幂还包括1/2次幂
### 正规方程
正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：$\frac{\partial }{\partial {{\theta }_{j}}}J\left( {{\theta }_{j}} \right)=0$ 。
 假设我们的训练集特征矩阵为 X（包含了${{x}_{0}}=1$）并且我们的训练集结果为向量 y，则利用正规方程解出向量 $\theta ={{\left( {{X}^{T}}X \right)}^{-1}}{{X}^{T}}y$ 。
上标T代表矩阵转置，上标-1 代表矩阵的逆。

### 实验过程中遇到的问题及解决方案
1. [jupyter notebook 更换主题](https://www.cnblogs.com/shanger/p/12006161.htm)
  ```cmd
  pip install jupyterthemes grade3 jt -t grade3 -f fira -fs 17 -cellw 90% -ofs 15 -dfs 15 -T -T
  ```
   
2. [关于dataframe删除某一行或某一列的方法](https://blog.csdn.net/qiwsir/article/details/114867900)
  ```python
  df.drop(columns=['b'])       # drop a column
  ```

3. 一些常见操作pd, np, plt
   1. ddd
   2. ddd
   3. ddd

4. Markdown基本教程
   1. [Markdown基本教程](https://editor.mdnice.com/?outId=d988634c248640a7bd2f20d8c32b5e4a)
   2. [Markdown数学公式](https://blog.csdn.net/jyfu2_12/article/details/79207643)
  