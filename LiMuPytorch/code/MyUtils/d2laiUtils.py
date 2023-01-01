import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib_inline import backend_inline
import torchvision
from torchvision import transforms
from torch.utils import data
from IPython import display
from torch import nn
import collections
import re
import random
from torch.nn import functional as F
import math

class Timer:
    """时间基准测试, 可以计算程序片段的运行时间"""
    def __init__(self):
        self.timeList = []  # 记录多次运行时间
        self.start()
        
    def start(self):
        """记录初始化时的时间戳"""
        self.tick = time.time()
    
    def stop(self):
        self.timeList.append(time.time()-self.tick)
        return self.timeList[-1]
    
    def sum(self):
        """时间总和"""
        return sum(self.timeList)
    
    def avg(self):
        return self.sum()/len(self.timeList)
    
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
       
# 一个用于绘图的函数 plot, 可以方便的在同一张图里画多条曲线
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#@save
def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')
#@save
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

def loadFashionMnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
    
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型(定义见第3章)"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期(定义见第3章)"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    info = ''
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        info += f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}\n'
    print(info)
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def read_text(url=None):
    if not url:
        url = "../data/TheTimeMachine.txt"
    with open(url, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub(r"[^a-zA-Z]+", " ", line).strip().lower() for line in lines]

def tokenize(lines, token="word"):
    if token == "word":
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 将tokens展平为一个单词列表，对其中的单词进行计数并按词频倒序排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), 
                                   key=lambda x: x[1], reverse=True)
        # id->token, token->id
        # 先处理保留词元，例如未定义词语
        self.id2token = ['<unk>']+reserved_tokens
        self.token2id = {token: idx for idx, token in enumerate(self.id2token)}
        # 接下来对_token_freqs进行处理
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            self.id2token.append(token)
            self.token2id[token]=len(self.id2token)-1
    
    def __len__(self):
        return len(self.id2token)
    # 通过token获取id, 注意这是一个递归操作, 也即是可以传入列表的
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # dict().get(key, default_value)
            return self.token2id.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.id2token[indices]
        return [self.to_tokens[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0
    # @property使得外部类可以通过函数名访问私有属性
    @property
    def token_freqs(self):
        return self._token_freqs    

def count_corpus(tokens):
    """词频计数"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    lines = read_text()
    tokens = tokenize(lines, token="char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    corups = [vocab[token] for line in tokens for token in line]
    return corups, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps):]
    num_subseqs = (len(corpus)-1) // num_steps
    start_indices = list(range(0, num_steps*num_subseqs, num_steps))
    random.shuffle(start_indices)
    def subseq(i):
        return corpus[i: i+num_steps]
    batch_num = num_subseqs // batch_size
    for i in range(0, batch_num*batch_size, batch_size):
        batch_indices = start_indices[i: i+batch_size]
        X = [subseq(i) for i in batch_indices]
        Y = [subseq(i+1) for i in batch_indices]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    bias = random.randint(0, num_steps-1)  # [a, b]
    num_tokens = ((len(corpus)-1-bias) // batch_size) * batch_size
    Xs = torch.tensor(corpus[bias: bias+num_tokens]).reshape(batch_size, -1)
    Ys = torch.tensor(corpus[bias+1: bias+num_tokens+1]).reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(num_batches):
        X = Xs[:, i*num_steps: (i+1)*num_steps]
        Y = Ys[:, i*num_steps: (i+1)*num_steps]
        yield X, Y

# 把上面两个采样函数包装成为一个类
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random=False, max_tokens=10000):
        if use_random:
            self.iter_func = seq_data_iter_random
        else:
            self.iter_func = seq_data_iter_sequential
        self.corups, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.iter_func(self.corups, self.batch_size, self.num_steps)

# 接下来创建一个函数，返回这个迭代器和词表
def load_data_time_machine(batch_size, num_steps, use_random=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random, max_tokens)
    return data_iter, data_iter.vocab

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 隐状态初始化
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 下面的rnn函数定义了如何在一个时间步内计算隐状态和输出。 
# 循环神经网络模型通过inputs最外层的维度实现循环， 以便逐时间步更新小批量数据的隐状态H。
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.id2token[i] for i in outputs])

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random):
    """训练网络迭代一个周期"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # 累积损失 和 词元数量
    for X, Y in train_iter:
        if state is None or use_random:
            state = net.begin_state(X.shape[0], device)  # 如果是第一次或者使用随机序列 每次都重新初始化
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else: # state对于nn.LSTM以及自定义模型是一个tuple
                for s in state:
                    s.detach_()
        X, y = X.to(device), Y.T.reshape(-1).to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()
        if isinstance(net, nn.Module):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l*y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random=False):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    # 训练和预测
    animater = Animator(xlabel="epoch", ylabel="perplexity", legend=["train"], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random)
        if (epoch+1) % 10 == 0:
            print(predict("time traveller "))
            animater.add(epoch+1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.n_in = self.n_out = vocab_size
        self.n_hidden = self.rnn.hidden_size
        # 是否双向
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.n_hidden, self.n_out)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.n_hidden*2, self.n_out)
    
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.n_in).to(torch.float32)  # X: num_steps, batch_size, num_in
        output, state = self.rnn(X, state) # output: num_steps, batch_size, num_hidden
        # output 实际为每个时间步的隐状态在第0维上连接得到的输出
        # 因此从 output 到 y, 需要 1: reshape(-1, num_hidden) 2: Linear(n_hidden -> n_out)
        Y = self.linear(output.reshape(-1, output.shape[-1]))
        return Y, state
    
    def begin_state(self, batch_size, device):
        if not isinstance(self.rnn, torch.nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.n_hidden),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.n_hidden),
                                device=device),
                    torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.n_hidden), 
                                device=device))
