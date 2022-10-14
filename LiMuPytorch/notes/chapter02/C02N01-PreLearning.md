# 数据的存储与操作
为了完成对数据的操作，必须要有一种合适的形式将数据存储在计算机中，通常形式为n维的数组，在python语言中，在机器学习领域，他的名字叫做**张量(tensor)**。torch.tensor具备许多重要的的功能，例如**自动微分(auto-grad)**

## 1. 张量(tensor)的性质与构造
张量表示由一个数值组成的数组，这个数组可能有多个维度。 具有一个轴的张量对应数学上的**向量（vector）**； 具有两个轴的张量对应数学上的**矩阵（matrix）**


```python
import torch
x = torch.arange(12)  # arange()
x # 是一个一维的tensor
```




    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])



可以通过$shape$**属性**访问tensor实例的形状


```python
x.shape
```




    torch.Size([12])



注意x是一个行向量，若无特别指定，x将放在内存中，并且使用CPU进行运算操作
如果只想获取tensor的元素个数，可以使用$tensor.numel()$**函数**


```python
x.numel()
```




    12



$tensor.reshape(a, b)$, 将tensor的形状改为axb的形状，其中行或列可以缺省，用-1代替，另外，reshape函数的参数也可以是元组，例如reshape((a, b, c))


```python
X34 = x.reshape(3, -1)
X34
```




    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])



有时我们需要得到**特定形状的元素全0或全1的张量**，或者**特定形状的随机初始化的张量**，使用函数zeros(a, b)或者ones(a, b)或者randn(a, b)(注：随机采样自标准正态分布，范围在0-1之间，可依据实际调整数据范围)即可，同样这里也可以是元组。另外，我们也可以使用python的(嵌套)列表(list)来为tensor赋初值


```python
X0 = torch.ones((2, 3, 3))
X1 = torch.zeros((2, 3, 3))
Xr = torch.randn((2, 3, 3))
X0, X1, Xr
```




    (tensor([[[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]],
     
             [[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]]]),
     tensor([[[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
     
             [[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]]]),
     tensor([[[-1.4777, -0.8771,  0.8572],
              [-0.7417,  1.2801, -0.3386],
              [-2.1799,  0.5772,  0.0328]],
     
             [[ 0.7768,  1.2282,  0.7051],
              [-0.6947,  2.6272,  0.6684],
              [ 0.5836, -0.0804,  0.7068]]]))



## 2. 张量的四则运算
这里主要是相同形状的tensor按元素对应的进行标量运算，例如$+, -, *, /, **, \% $，以及torch.exp(), torch.log()等，向量点积与矩阵运算等线性代数操作放在后面讲.


```python
x = torch.tensor([1, 3.0, 4, 2])
y = torch.tensor([2, 1, 2, 3])
x+y, x-y, x*y, x/y, x**y, x%y
```




    (tensor([3., 4., 6., 5.]),
     tensor([-1.,  2.,  2., -1.]),
     tensor([2., 3., 8., 6.]),
     tensor([0.5000, 3.0000, 2.0000, 0.6667]),
     tensor([ 1.,  3., 16.,  8.]),
     tensor([1., 0., 0., 2.]))




```python
torch.exp(x), torch.log(x)
```




    (tensor([ 2.7183, 20.0855, 54.5981,  7.3891]),
     tensor([0.0000, 1.0986, 1.3863, 0.6931]))



此外还可以多个张量连结$concatenate$在一起，使用$cat$函数，对于矩阵，我们可以沿行（轴-0，形状的第一个元素） 和按列（轴-1，形状的第二个元素）连结。以及$sum$对张量中的元素求和（得到单元tensor）。(注意这个dim是有取值范围的，例如假设x,y都是行向量，那么他们只能在行方向上拓展，也即拓展后还是一个行向量。)


```python
X = torch.arange(6).reshape((2, 3))
Y = torch.tensor([2, 3, 4, 3, 2, 1]).reshape((2, 3))
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```




    (tensor([[0, 1, 2],
             [3, 4, 5],
             [2, 3, 4],
             [3, 2, 1]]),
     tensor([[0, 1, 2, 2, 3, 4],
             [3, 4, 5, 3, 2, 1]]))




```python
X.sum()
```




    tensor(15)



## 3. 广播
在某些情况下，即使形状不同，我们仍然可以通过调用**广播机制(broadcasting mechanism)**来执行按元素操作.在大多数情况下，我们将沿着数组中长度为1的轴进行广播


```python
X = torch.tensor([2, 5, 7]).reshape((1, 3))
Y = torch.tensor([3, 4]).reshape((2, 1))
X, Y, X+Y
```




    (tensor([[2, 5, 7]]),
     tensor([[3],
             [4]]),
     tensor([[ 5,  8, 10],
             [ 6,  9, 11]]))



## 4. 索引和切片


```python
X = torch.arange(6).reshape((2,3))
# 取1行，例如第二行
X_row2 = X[1, :]
# 取一列，例如第三列
X_col3 = X[:, 2]
# 取一个元素，例如第2行第2列
x_22 = X[1, 1]
# 取一块，例如，第1，2行和第1，2列
X_22 = X[0:2, 0:2]
X, X_row2, X_col3, X_22,x_22
```




    (tensor([[0, 1, 2],
             [3, 4, 5]]),
     tensor([3, 4, 5]),
     tensor([2, 5]),
     tensor([[0, 1],
             [3, 4]]),
     tensor(4))



## 5. 内存的消耗
1. Y = X + Y, 将加法结果重新赋值给Y后，Y的地址将改变
2. Y[:] = X + Y
3. Y += X


```python
# 使用id表征数据地址
X = torch.tensor([1, 3, 5, 7, 9, 6]).reshape((2,3))
Y = torch.tensor([2, 4, 4, 6, 3, 8]).reshape((2,3))
before = id(Y)
Y = X + Y # t <- X + Y, Y <- t, so we allocate new memory in this statement
before == id(Y)
```




    False




```python
# 对于第二个操作，地址会改变吗？
Z = torch.zeros_like(Y)
print("id(Z)=", id(Z))
Z[:] = X + Y
print("id(Z)=", id(Z))
```

    id(Z)= 2117180683184
    id(Z)= 2117180683184
    


```python
# 对于第三个操作，地址会改变吗？
print(id(X))
X += Y
print(id(X))
```

    2117181327536
    2117181327536
    
