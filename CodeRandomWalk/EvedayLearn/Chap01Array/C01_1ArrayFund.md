# 数组基础
**数组是存放在连续内存空间上的相同类型数据的集合**
## 数组基本要点
1. 数组下标都是从0开始的。
2. 数组内存空间的地址是连续的

正是因为数组的在内存空间的地址是连续的，所以我们**在删除或者增添元素的时候，要移动其他元素**的地址。

那么二维数组在内存的空间地址是连续的么？不同编程语言的内存管理是不一样的，以C++为例，在C++中二维数组是连续分布的
像Java是没有指针的，同时也不对程序员暴露其元素的地址，寻址操作完全交给虚拟机。
![](https://img-blog.csdnimg.cn/20201214111631844.png)

## refer code
```c++
void test_arr() {
    int array[2][3] = {
		{0, 1, 2},
		{3, 4, 5}
    };
    cout << &array[0][0] << " " << &array[0][1] << " " << &array[0][2] << endl;
    cout << &array[1][0] << " " << &array[1][1] << " " << &array[1][2] << endl;
}

int main() {
    test_arr();
}
```

```c++

```


##