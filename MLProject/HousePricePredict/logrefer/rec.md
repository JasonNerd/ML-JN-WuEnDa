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