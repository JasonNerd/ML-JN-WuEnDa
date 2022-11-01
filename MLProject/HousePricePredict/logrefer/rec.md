### skewness and kurtosis
偏度（skewness），是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。定义上偏度是样本的三阶标准化矩。

偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）

峰度（peakedness；kurtosis）又称峰态系数。表征概率密度分布曲线在平均值处峰值高低的特征数。直观看来，峰度反映了峰部的尖度。随机变量的峰度计算方法为：随机变量的四阶中心矩与方差平方的比值。

### seaborn-pandas-plt
1. 直方图
    ```python
    sns.histplot(trainDF["SalePrice"])
    plt.show()
    ```
2. 散点图
    ```python
    var = 'GrLivArea'
    data = pd.concat([trainDF['SalePrice'], trainDF[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice')
    ```
3. 盒状图
    ```python
    var = 'OverallQual'
    data = pd.concat([trainDF[var], trainDF["SalePrice"]], axis=1)
    sns.boxplot(x=var, y="SalePrice", data=data)
    plt.show()
    ```

### RMSE和RMSLE有什么区别？
RMSLE计算公式
![](https://files.mdnice.com/user/35698/acd54d97-4c15-4aee-b97d-29ba97c42dbc.png)
请注意，在公式中，X 是预测值，Y 是实际值。

当我们看到 RMSE 的公式时，它看起来就像是对数函数的差异。实际上，对数的微小差异是赋予RMSLE自身独特属性的主要因素。由于对数的性质，RMLSE 可以大致视为预测值和实际值之间的相对误差。RMSLE 对实际变量的低估比高估的惩罚更大。
### 如何初始化模型参数
[Pytorch 中torch.nn.Linear的权重初始化](https://blog.csdn.net/D_handsome/article/details/122715621)
apply()函数
```py
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
#nn.Flatten将多维张量压缩为二维（一行）进而进行线性操作、
 
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
 
net.apply(init_weights)
```
### torch.optim.Adam
[torch.optim.Adam优化器参数学习](https://www.cnblogs.com/BlueBlueSea/p/14269033.html)

```py
Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```
* params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
* lr (float, 可选) – 学习率（默认：1e-3）
* betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
* eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
* weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）