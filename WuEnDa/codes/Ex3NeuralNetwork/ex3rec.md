# Neural Network 神经网络
* <font color=#A0A>多分类学习任务-手写数字识别</font>
  1. 实验预处理
      * 数据是5000张20$\times$20的灰度图像，每个图像转为400维的向量，也即特征为400个，于是形成5000$\times$24的矩阵，另外还有标记值，依次为0~9。
       * 数据存放在mat文件中。可以借助scipy库进行读取，建议将其中的一些向量还原为灰度图像看一看
  2. 实验方法
     * 使用OvR方法或者OvO方法进行训练
     * 使用NN进行训练
---
* <font color=#A0A>试验记录$log$</font>
1. mat文件读取
   ```python
   from scipy.io import loadmat
   data = loadmat('ex3data1.mat')
   type(data) # dict
   ```
2. 将矩阵绘制为灰度图
   ```python
   matplotlib.pyplot.imshow(array)
   # 另一种方法
   from PIL import Image
   picture = Image.fromarray(array)
   picture = picture.convert("L")
   picture.save(save_path)
   ```
3. np.array()操作
   [NumPy数组元素增删改查](http://c.biancheng.net/numpy/array-curd.html)
   ```python
   # 1. 插入行或列
   numpy.insert(arr, i, val, axis)
   # arr：要输入的数组
   # i：表示索引值，在该索引值之前插入 values 值；
   # val：要插入的值；
   # axis：指定的轴，如果未提供，则输入数组会被展开为一维数组。
   ```
4. [非线性规划（scipy.optimize.minimize）](https://www.jianshu.com/p/94817f7cc89b)
   [Python scipy.optimize.minimize用法及代码示例](https://vimsky.com/examples/usage/python-scipy.optimize.minimize.html)
   ```python
    scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    # 其中：
    # fun：目标函数，返回单值，
    # x0：初始迭代值，
    # args：要输入到目标函数中的参数
    # method：CG，BFGS，Newton-CG，L-BFGS-B，TNC，SLSQP等

    # 返回值
    """
    print(res.fun)
    print(res.success)
    print(res.x)
    """
   ```