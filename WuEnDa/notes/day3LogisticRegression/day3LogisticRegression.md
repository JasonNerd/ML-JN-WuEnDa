# 对率回归模型(Logistic Rregression Model)

### 内容回顾与模型引出
$\space$$\space$$\space$$\space$$\space$对率回归模型(Logistic Regression Model)或者称逻辑回归模型是一类针对分类问题而产生的模型，它是从线性回归模型中衍生而来的。
$\space$$\space$$\space$$\space$$\space$在线性回归模型中，我们使用特征的线性组合来对连续的标记值进行拟合，并确保总体拟合值与标记值尽可能接近，为了刻画这种接近程度，定义了一个损失函数或者代价函数，也即均方差，表现为两点间的距离。同时为了方便计算机对参数进行迭代，从微分角度给出了参数变化方向也即梯度函数，并使用学习率控制参数变化的快慢以确保能够收敛。
$\space$$\space$$\space$$\space$$\space$注意到，线性回归模型也不是仅仅只能够拟合线性的标记值分布。如果对其中的特征进行变换例如它的幂并将其作为新的特征，就可以以多项式曲线的方式对标记值进行拟合，理论上可以逼近任意曲线，然而，当幂次过高时可能出现过拟合现象，这可以通过正则化的手段进行处理，其思想核心在于尽可能减小参数值以使得曲线不会过于扭曲。
$\space$$\space$$\space$$\space$$\space$另一方面，还可以考虑对$y$值进行变换，例如，假设通过对数据的观察我们猜测$y$与$x$呈指数函数关系，例如$y=ke^{mx}, (k>0, m\neq 0)$，那么不妨在等式两边取对数，此时就变成最简单的线性回归模型了。基于这一思想，设想一个函数将任意$R$内的实数都映射到(0, 1)内, 也即从$h(x)=\theta^T X$变为$h(x)=g^{(-1)}(\theta^TX)$, 同时还要定义一个代价函数cost，使得在给定一组参数值$\theta$时，若$y=0$(以二分类为例)但$h(x)\rightarrow1$时cost会很大(此时表示预测结果反了，就应该给予较大的代价), 同时cost也必须是凸函数并存在关于$\theta$的导数，这样才可以求梯度并可以迭代收敛。
### 逻辑回归模型的建立
$\space$$\space$$\space$$\space$$\space$貌似找到这样一个函数很困难，但有人找到了并告诉了我，因此这里我也将告诉你。首先是这个$R\rightarrow(0,1)$的函数，可以取$sigmoid$函数，意为S形的曲线函数，函数定义是$sigmoid(z)=\frac{1}{1+e^{-z}}$。因此有假设函数$h(x)=sigmoid(\theta^TX)$。另外，代价函数可以取为$J(\theta)=\frac{1}{m}\sum^m_{i=1}[-y^{(i)}log(h(\boldsymbol{x^{(i)}}))-(1-y^{(i)})log(1-h(\boldsymbol{x^{(i)}}))]$。注意这里$\boldsymbol{x^{(i)}}$是一个向量, 表示第i个实例（特征向量），假设有n个特征，则$\boldsymbol{x^{(i)}}$就是n维向量。最后再求它对$\theta_j$的偏导，值得注意的是，对于$sigmoid$ $function$ $g$，有$g'=g(1-g)$。这里推导过程不展开讲，但令人惊讶的是，它的数学形式竟和线性回归模型中的偏导数相同！也即$\frac{\partial}{\theta_j} J(\theta) = \frac{1}{m}\sum^m_{i=1}[h(\boldsymbol{x^{(i)}}))-y^{(i)}]x^{(i)}_j$
$\space$$\space$$\space$$\space$$\space$那么陈述至此或许你已经明白，对率回归只不过是在线性回归的基础上做了一点改动而已，尽管这样的改动最初或许是十分难以想到的，但没关系我们现在已经掌握了它。补充几点说明，已经训练好了参数，当给出一个新的示例，例如$x^{(m+1)}$，那么如何知道它的类别呢？（以二分类示例）只需要计算$h(x^{(m+1)})$然后与0.5比较，大于0.5即为正类，否则为负类。同时结合sigmoid函数图像知当$\theta^TX>0$是就有$h(x)=g(\theta^TX)>0.5$，因而$\theta^TX=0$就成为了决策边界，如果是仅有两个特征且为二分类的问题，那么可以进行可视化，会看到$\theta^TX=0$形成了类边界。那么对于多分类问题，可以采用oneVSall或者oneVSrest的方法，不再展开陈述。同样的，逻辑回归模型也可以采取正则化的方法缓解过拟合。
### 关键代码实现
* 将课件中的公式以向量化的形式完整呈现出来，这样的代码是原汁原味的，但是参数（初值）调整会十分麻烦，不能随意的赋初值，否则绝大多数情况不会收敛到合理值。
    ```python
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    # 1 读取数据
    data = pd.read_csv("ex2data1.txt", header=None, names=["ScA", "ScB", "Accepted"])
    # 2 画散点 正类负类分开
    pos_d = data[data['Accepted'] == 1]
    neg_d = data[data['Accepted'] == 0]
    ax, fig = plt.subplots()
    fig.scatter(pos_d.ScA, pos_d.ScB)
    fig.scatter(neg_d.ScA, neg_d.ScB, c='r', marker='x)
    plt.show()
    # 3 划分X, y, 初始化theta
    data.insert(0, 'Ones', 1)
    n = data.shape[1]
    X = np.matrix(data.iloc[:, 0:n-1])
    y = np.matrix(data.iloc[:, n-1:n])
    theta = np.matrix(np.zeros(n-1))
    # X.shape, y.shape, theta.shape
    # ((100, 3), (100, 1), (1, 3))
    # 4 写sigmoid函数
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    # 5 写costFunc
    def costFunc(theta, X, y):
        m = len(y)
        h_x = sigmoid(X*theta.T)
        return ((-y.T)*np.log(h_x)-(1-y.T)*np.log(1-h_x))/m
    # costFunc(theta, X, y) # 0.693147
    # 6 写出梯度下降算法
    def gradientDescent(theta, X, y, alpha, iters):
        cost = np.zeros(iters)
        for i in range(iters):
            cost[i] = costFunc(theta, X, y)
            h_x = sigmoid(X*theta.T)
            err = h_x-y
            delta = (X.T*err)/m
            theta = theta - alpha*delta.T
        return theta, cost
    # 7 初始化参数，进行训练，作出cost图、决策边界曲线
    alpha = 0.0003
    iter_n = 2000
    theta = np.matrix([-30, 2, 1])
    theta, cost = gradientDescent(X, y, theta, iter_n, alpha)
    plt.plot(range(iter_n), cost)
    plt.show()
    # cost[-1], theta
    # (0.20651168067356068, matrix([[-30.02499304,   0.24517859,   0.24078545]]))
    # 决策边界图代码略
    ```

* 使用库函数，但需要实现代价函数cost和梯度（一步）gradient，注意参数类型合法以及参数顺序
  ```python
  """
  import scipy.optimize as opt
  result = opt.fmin_tnc(func=costFunc, x0=theta, fprime=gradient, args=(X, y))
  """
  def costFunc(theta, X, y):
    pass
  def gradient(theta, X, y):
    return delta.T
  # 定义好后, 传入fmin_tnc方法中, args是costFunc除第一个参数theta以外的其它参数, gradient返回导数向量
  # fmin_tnc会在内部自动设置学习率并不断调整, theta会不断迭代, 且为array类型, 期间多次报错矩阵乘法维数不断, 仔细检查并且单独运行发现无异样, 后来猜测theta不断更新并反复传入cost和gradient中计算, 因此应当先转换为matrix类型
  ```

### 非线性的二分类与正则化参数的使用
* 代价函数
$$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}$$

* 梯度
如果我们要使用梯度下降法令这个代价函数最小化，因为我们未对${{\theta }_{0}}$ 进行正则化，所以梯度下降算法将分两种情形：
![](https://files.mdnice.com/user/35698/0f37d50b-1da9-41fe-905b-590c980c3cb5.png)
对上面的算法中 j=1,2,...,n 时的更新式子进行调整可得： 
${{\theta }_{j}}:={{\theta }_{j}}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}})x_{j}^{(i)}}$
