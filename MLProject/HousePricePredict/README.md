# [房价预测-深度学习项目一](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)
## 问题描述
* 你的任务是对测试集中的每一个房子作出售价预测。我们将使用RMSE(Root-Mean-Squared-Error)来评判你的预测结果
* 你提交的预测结果文件类似于
  ```git
  Id,SalePrice
  1461,169000.1
  1462,187724.1233
  1463,175221
  ...
  ```
## 实验步骤
### 数据描述
[kaggle 房价预测 特征描述](https://blog.csdn.net/heheda_22/article/details/108937695)
数据包括训练数据集和测试数据集，对于训练数据集train.csv，它包含了1460条数据, 除ID以及预测目标SalePrice外一共有79个特征。
> 部分数据描述
    MSSubClass: 参与销售的公寓的类型, 整数, 包含(新旧程度套间楼层数量等)15种
    MSZoning: 住宅所在区域类型, 字符串, 包含(A[农业], C[商业], FV, I, RH, RP, RM)7种
    LotFrontage: 房屋到街道的直线距离, 整数
    LotArea: 房屋占地面积, 整数
    Street: 通往房屋的道路类型, 字符串(Grvl: 碎石路, Pave: 铺好的路)2种
    Alley: 通往房屋的小巷的类型, 字符串(Grvl, Pave, NA: 没有通往房屋的小路)3种
    LotShape: 房屋的一般形状, 字符串(规则程度)4种
    LandContour: 房屋所在地的平整程度, 字符串(平地、山坡、洼地等)4种
    Utilities: 可用设施, 字符串(电气水是否可用)4种
    LotConfig: 地段布局, 字符串(临街、死角等)4种
    LandSlope: 坡度, 字符串()3种
    Neighborhood: 周边区域, 字符串(是否与学校、山地、公园、铁路等邻近)25种
    Condition1: 接近主干道或者铁路()9
    Condition2: 接近主干道或者铁路(如果存在第二个)9
    BldgType: 5     HouseStyle: 8       OverallQual: 10     OverallCond: 10     YearBuilt: ()
    YearRemodAdd:() RoofStyle: 6        RoofMatl: 8         Exterior1st: 17     Exterior2nd: 17
    MasVnrType: 5   MasVnrArea: 5       ExterCond: 5        Foundation: 6       BsmtQual: 6

### 数据预处理与训练
1. 读取训练数据, 查看哪些字段有缺失值, 作出直方图
2. 考虑是否舍弃部分特征, 例如缺失值较多的特征, 可以构造一个过滤函数, 他返回舍弃部分特征数据的副本
3. 对于剩余的数值类型的特征, 缺失值用均值填充, 然后做均值归一化
4. 对于剩余的字符串类型的特征, 它们往往是说明房屋在某方面的特点, 是离散的, 需要进行one-hot编码
5. 选择模型, 例如线性回归模型
6. 你可以使用多种技术或者方法:
    1. 正则化
    2. PCA降维(特征之间的相关性)
    3. 尝试增加多项式特征(特征与目标间的相关性)
    4. 尝试吴恩达老师的建议, 先快速构造一个系统, 然后再去完善它

## 项目结构
$code$-存放实验代码的文件夹  
$logrefer$-存放实验中遇到的问题以及参考资料的文件夹
$data$-存放实验用到的数据
$output$-存放输出数据的文件夹

## 要点总结