
# 支持向量机实验记录
## 实验步骤
<font face="宋体" size=4>

1. xxx
2. xxx
3. xxx
4. xxx
5. xxx

</font>

## 关键过程的理解与代码实现
<font face="宋体" size=4>

1. xxx
2. xxx
3. xxx
4. xxx
5. xxx

</font>

## 完整的代码实现
```python
import numpy as np

```

## 遇到的问题与解决方法
<font face="宋体" size=4>

### **关于github无法访问的问题**
&emsp;&emsp;首先是浏览器无法访问 github.com ，网页加载不出来，这一问题网上的解决办法是：先查询github.com的ipv4地址，接着将该地址复制粘贴到host文件形成映射，然后刷新DNS缓存，具体做法是打开CMD，键入命令`ipconfig /flushdns`后回车，最后通过`ping github.com`验证发现可以ping通，这样就ok啦。实际上了，到这一步可能都成功了，但是网页仍然访问不稳定，就像在掷骰子--**这将使你异常烦躁**。一个终极解决方案是使用**Dev-SideCar**，这是一个神器，网页秒开，珍惜使用吧！关于github非正常访问的问题，可能是DNS污染，为什么从去年开始突然会有这样的污染，不可说。
&emsp;&emsp;其次是git push报错，例如**某某文件过大**、**连接超时**、**SSL校验错误**等等。尤其是连接超时，解决办法千奇百怪，不一定都适用，并且可能在解决的过程中产生新的问题。一个办法是
```git
git config --global https.proxy
git config --global --unset https.proxy
```
也就是git代理多开关几次，注意这时devsidecar可以关咯，然后通过
```git
git config --global -l
```
查看哪些设置是被更改的（如果有些地方误设，还请取消掉）

### **关于github无法访问的问题**
&emsp;&emsp;首先是浏览器

### **关于github无法访问的问题**
&emsp;&emsp;首先是浏览器

</font>