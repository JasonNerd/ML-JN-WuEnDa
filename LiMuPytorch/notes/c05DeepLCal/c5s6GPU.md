# GPU
我们回顾了过去20年计算能力的快速增长。 简而言之，自2000年以来，GPU性能每十年增长1000倍。

本节，我们将讨论如何利用这种计算性能进行研究。 首先是如何使用单个GPU，然后是如何使用多个GPU和多个服务器（具有多个GPU）。

我们先看看如何使用单个NVIDIA GPU进行计算。 首先，确保你至少安装了一个NVIDIA GPU。 然后，下载NVIDIA驱动和CUDA 并按照提示设置适当的路径。 当这些准备工作完成，就可以使用nvidia-smi命令来查看显卡信息。

## 计算设备
在PyTorch中，CPU和GPU可以用torch.device('cpu') 和torch.device('cuda')表示。 应该注意的是，cpu设备意味着所有物理CPU和内存， 这意味着PyTorch的计算将尝试使用所有CPU核心。 然而，gpu设备只代表一个卡和相应的显存。 如果有多个GPU，我们使用torch.device(f'cuda:{i}') 来表示第块GPU（从0开始）。 另外，cuda:0和cuda是等价的。

## 张量与GPU
我们可以查询张量所在的设备。 默认情况下，张量是在CPU上创建的。
需要注意的是，无论何时我们要对多个项进行操作， 它们都必须在同一个设备上。 例如，如果我们对两个张量求和， 我们需要确保两个张量都位于同一个设备上， 否则框架将不知道在哪里存储结果，甚至不知道在哪里执行计算。
![]()
```py
Z = X.cuda(1)
print(X)
print(Z)
```
## 思考题
Q1: 尝试一个计算量更大的任务，比如大矩阵的乘法，看看CPU和GPU之间的速度差异。再试一个计算量很小的任务呢？
A1: 当计算量较大时，GPU明显要比CPU快；当计算量很小时，两者差距不明显。
```py
# 计算量较大的任务
X = torch.rand((10000, 10000))
Y = X.cuda(0)
time_start = time.time()
Z = torch.mm(X, X)
time_end = time.time()
print(f'cpu time cost: {round((time_end - time_start) * 1000, 2)}ms')
time_start = time.time()
Z = torch.mm(Y, Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000, 2)}ms')

# 计算量很小的任务
X = torch.rand((100, 100))
Y = X.cuda(0)
time_start = time.time()
Z = torch.mm(X, X)
time_end = time.time()
print(f'cpu time cost: {round((time_end - time_start) * 1000)}ms')
time_start = time.time()
Z = torch.mm(Y, Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000)}ms')
```
Q2: 我们应该如何在GPU上读写模型参数？
A2: 使用net.to(device=torch.device(‘cuda’))将模型迁移到gpu上，然后再按照之前的方法读写参数。

Q3: 测量计算1000个 100*100 的矩阵乘法所需的时间。记录输出矩阵的Frobenius范数，一次记录一个结果 vs 在GPU上保存并仅传输最终结果。
（中文版翻译有点问题，英文原版这句话是log the Frobenius norm of the output matrix one result at a time vs. keeping a log on the GPU and transferring only the final result，所以实质是要我们作对比）
```py
# 一次记录一个结果
time_start = time.time()
for i in range(1000):
    Y = torch.mm(Y, Y)
    Z = torch.norm(Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000)}ms')
Y = X.cuda(0)
# 在GPU上保存并仅传输最终结果
time_start = time.time()
for i in range(1000):
    Y = torch.mm(Y, Y)
Z = torch.norm(Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000)}ms')
```
## 代码
```py
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```
