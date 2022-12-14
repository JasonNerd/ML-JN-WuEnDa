# Support Vector Machines(支持向量机)
## 支持向量机的直观理解--他要做什么？
&emsp;&emsp;支持向量机，这个名字并不能望文生义，也就是很难从字面上理解这个概念到底是在描述的是什么。

&emsp;&emsp;首先，他是用于监督学习的分类任务中的算法。其次，以二分类任务为例，假设两个类是可以线性划分的，也即，假设样例只有两个特征，将每个样本点绘制在二维平面上，支持向量机讨论的是找到一个“最合适的”直线，这一直线能够很好的划分两类数据，并且“最合适”的意思是这一直线（超平面）离两类样本的间隔最大，而一类样本相对于一个超平面的距离自然是该类中距离该超平面最近的点对应的间隔，也即要求最短间隔最大，直观上看，直线应恰好位于样本点的中央。

<div align=center><img src="http://www.ai-start.com/ml2014/images/e68e6ca3275f433330a7981971eb4f16.png" width=80%/></div>

&emsp;&emsp;如上图，支持向量机要做的就是找到这个黑色的决策边界，使得当有新的示例加入时，这一个决策能够尽可能正确的工作。但是，**当面对异常值时，支持向量机应当具备这样的策略，不至于太差，主要是正则化的使用**。如下图所示，支持向量机算法应该能够仍然选择黑色的决策边界，而不是洋红色那一个。

<div align=center><img src="http://www.ai-start.com/ml2014/images/b8fbe2f6ac48897cf40497a2d034c691.png" width=80%/></div>

## 从线性回归模型开始讨论支持向量机的构造--进一步理解支持向量机
&emsp;&emsp;线性回归模型(Linear Regression Model)是使用特征的线性组合来对连续型的标记值进行拟合，这个线性组合被称为**假设函数**$h_{\theta}(x)=\theta^Tx=\omega^Tx+b$，LRM的目标是尽可能的使$h_{\theta}(x)$与$y$接近，为此定义了**代价函数**，它表征了$h$与$y$的差距，其数学表现形式是均方差，也即$J=\frac{1}{m}\sum(y-h)^2$，这里$m$表示样本数量。为了使$J$最小，一种方法是**梯度下降算法**，这是一种迭代算法，核心在于确定向量$\theta$变化的方向（也即导数变化的负方向，梯度）以及每一次迭代的步长（**学习率**）。另一方面，为了拟合非线性变化的标记值y，引入了**多项式特征**，意思是将原有特征的多项式形式作为新的特征，例如$x_1x_2$或者$x_1^2$等。这样做就可以拟合非线性的标记值，但另一方面当幂次过高时，可能出现的情况是假设函数可以很好的拟合已知数据集（**训练集**），但对于新的数据点（**验证集**）误差就很大，出现了**过拟合**现象，假设函数学习到了训练集的局部数据特征，这也被称之为**高方差问题(High Variance)**，方差描述的是假设函数对抗数据扰动的能力，这里出现的新数据使得代价函数值显著增大，采取的对策是在代价函数中**加入正则化项$\lambda\sum\theta_j^2$**，基本思想是由于要使$J$尽可能小，加入这一惩罚项后，$\theta$也会对应的减小，而假如高幂次项的系数更小了（极端假设为0，相当于没有高幂次项），这也使得曲线更平滑，不至于过拟合。观察到正则化项前的系数$\lambda$，$\lambda$太大则会引发另一个问题，也即**欠拟合**，或者**高偏差（bias）问题**，这表明假设函数本身的学习能力较差，此时可以减小正则化系数，增加更多的有效特征等。

&emsp;&emsp;那么接下来考虑分类问题，例如二分类问题，可以先这样想，此时仅仅是$y$的值不再是连续变化在实数域取值了，变成了两个值，例如1（正类）或者0（负类）（当然也可以选择任何其他的值，例如1和-1）。此时假设我们仍然LRM来对样本进行拟合，或许我们也可以得到一条直线，似乎也能把样本分开？（或许？）。但是$\theta^Tx$可能很大或者很小，因此引入sigmoid函数，它可以进行$\R\rightarrow(0,1)$的映射，函数形式为$sigmoid(z)=\frac{1}{1+e^{-z}}$，此时就得到**对率回归模型**(Logistic Regression Model)的假设函数$h=sigmoid(\theta^Tx)$。此时$h$可以看作样本为正例的概率，$h$不小于0.5时预测样本x为正例否则负例。另一方面假设样本为正例此时$y=1$如果$h$接近0则赋予较大的代价值，接近1则赋予较小的代价值，因此代价函数变为$\sum y(-\ln h)+(1-y)(\ln(1-h))$。巧妙地是，sigmoid函数的数学性质很好，这一函数是可以优化求解的，形式上与线性回归相同。

&emsp;&emsp;在对率回归模型中，对$x$分类的依据是$h$是否大于0.5，依据sigmoid函数的性质，这等价于判断$\theta^Tx=\omega^Tx+b$是否大于0。对于训练好的参数$\omega$和$b$，$\omega^Tx+b=0$就形成了决策边界。回到svm的指导思想，他是要最近距离最大。对于任意一个样本空间的点$x$，该点与决策平面的距离是$$d=\frac{|\omega^Tx+b|}{|\omega|}$$。为方便讨论，设负类标记值为-1，正类为1，则当决策边界正确分类时，若$\omega^Tx+b<0$则$y=-1$，也即他们同号。因此$$d=\frac{y(\omega^Tx+b)}{|\omega|}$$，遍历所有样本点得到$dmin$，因此问题转化为
$$ max \rightarrow dmin $$ s.t. $$d=\frac{|\omega^Tx+b|}{|\omega|} >= dmin$$
同时，由于$\omega$和$b$总是可以同步缩放的，此时不影响超平面的方程。不妨设当样本的某些点满足s.t.的等号条件时有$1=\omega^Tx+b$或$-1=\omega^Tx+b$，这些点即为**支持向量**。问题转化为:

$$ max \frac{1}{||\omega||}$$ s.t. $$y_i(\omega^Tx_i+b)>=1 (i=1, ..., m)$$

## 从代价函数的变化看支持向量机
&emsp;&emsp;对率回归模型的代价函数基本形式是：
$$J=-\frac{1}{m}\sum_{i=1}^m(y_i*ln(h)+(1-y_i)*ln(1-h))+\frac{\lambda}{2m}\sum_{j=1}^m\theta^2_j$$
不考虑正则化项, 对于单个样本
$$J = y*(-ln\frac{1}{e^{-\theta^Tx}+1})+(1-y)*(1-ln\frac{1}{e^{-\theta^Tx}+1})$$
也即
$$J = y*cost_1(\theta^Tx)+(1-y)*cost_0(\theta^Tx)$$
$y=1$时$J=cost_1$而$y=0$时$J=cost_0$，分别作出$cost_0$和$cost_1$的函数图像如下:

<div align=center><img src="http://www.ai-start.com/ml2014/images/66facb7fa8eddc3a860e420588c981d5.png" witdh=80%/></div>

&emsp;&emsp;洋红色的曲线即为SVM对应的两个$cost$函数，可见SVM将其进行了简化，但同时又与对率函数的很接近。并且我们对代价函数进行了一定的抽象，同时在$SVM$中，不再使用$A+\lambda B$的形式，而是使用$C*A+B$的形式，同时不考虑m这个值，就有了

<div align=center><img src="http://www.ai-start.com/ml2014/images/cc66af7cbd88183efc07c8ddf09cbc73.png" witdh=80%/></div>


&emsp;&emsp;观察到，当样本分类为正类时，必须满足条件$\theta^Tx>=1$否则必须满足$\theta^Tx<=-1$，因而支持向量机有时被称为**大间距分类器**，此时将得到一个“有趣的”决策边界。

### 如何理解SVM的决策边界具有这样的大间隔
&emsp;&emsp;考虑SVM的总体代价函数：
$$C\sum_{i=1}^m[ycost_1+(1-y)cost_0]+\frac{1}{2}\sum_{j=1}^n\theta_j^2$$

&emsp;&emsp;为了便于理解，尽量考虑理想化情况，假设C无限大，为了优化J，就必须满足有$y=1\rightarrow cost_1=0$ 而 $y=0\rightarrow cost_0=0$使得前面的项无限接近于0，因此优化目标就变为了$minimize \rightarrow \sum_{j=1}^n\theta_j^2$，从向量角度的观点，也即是要使得$||\theta||$最小化，此即为$\theta$的模长。优化问题表征如下：

<div align=center><img src="http://www.ai-start.com/ml2014/images/03bd4b3ff69e327f7949c3d2a73eed8a.png" witdh=80%/></div>

**因此支持向量机做的全部事情，就是极小化参数向量范数的平方，或者说长度的平方**

&emsp;&emsp;**向量内积**的讨论：$(u, v) = u^Tv = v^Tu$。因此$\theta^Tx$也即$\theta$向量与样本$x$向量的内积，不考虑截距$\theta_0$，假定只有两个特征，这就变为了两个平面向量的点乘。$$\theta^Tx=\theta.x=||\theta||*||x||*cost<\theta, x>$$
接着引入**投影**的概念，他表示的是一个向量在另一个向量上的投影，是一个标量，具备正负号，取决于两个向量夹角，就有了：$$\theta^Tx=\theta.x=Proj^x_\theta*||\theta||$$

&emsp;&emsp;于是一切变得明晰起来，如下图：

<div align=center><img src="http://www.ai-start.com/ml2014/images/5eab58ad9cb54b3b6fda8f6c96efff24.png" witdh=80%/></div>

&emsp;&emsp;图中，由于截距项为$0$，因此决策边界总是过原点，同时，决策边界上的点总满足$\theta^Tx=0$，因此决策边界与$\theta$向量垂直。以正例中位于第四象限的样本$x$为例，在满足$Proj^x_\theta*||\theta||>=1$的条件下（也即可以正确划分样本的条件下），为了减小$\theta$的模长，则$x$在$\theta$方向上的投影长度越大越好，图中左侧给出了一个坏的边界绿色线条，作出$\theta$方向，再给出投影，也即红色短线的长度，比较小，因而是个坏边界。而如果像右侧取y轴作为决策界限，此时$\theta$方向为x轴正方向，样本在$\theta$方向上的投影都比较大。再次见证**SVM是一个大间隔分类器**。

&emsp;&emsp;**总结起来，SVM向量机是一个大间隔分类器，简化了逻辑回归中的代价函数，所做的事情就是在保证大间距也就是正确分类的条件下最小化参数向量的模长，换句话说，就是总体来看，样本在参数向量方向上的投影最大。**

## 在python中使用SVM进行数据训练
// TODO: