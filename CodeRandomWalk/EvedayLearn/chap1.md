# 算法性能度量

## $O(n)$
**任何开发计算机程序员的软件工程师都应该能够估计这个程序的运行时间是一秒钟还是一年**
$O$是指上确界，例如$O(n)$ , $O(n^2)$, $O(n\log n)$
做个测试实验，看一下自己电脑1s大概可以处理的数据量，尽管电脑提示还运行着其他任务，并且各个指令不尽相同。
```C++
#include<iostream>
#include<chrono>
#include<thread>
using namespace std;
using namespace chrono;

void fN(long long n){
    long long i=0;
    while(i++<n);
}

void fN2(long long n){
	long long k=0;
	for(long long i=0; i<n; i++){
		for(long long j=0; j<n; j++){
			++k;
		}
	}
}

void fNlgN(long long n){
	long long k=0;
	for(long long i=0; i<n; i++){
		for(long long j=1; j<n; j *= 2){
			++k;
		}
	}
}

int msCatch(long long n, void (*func)(long long n)){
	milliseconds startTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	func(n);
	milliseconds endTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	return milliseconds(endTime).count() - milliseconds(startTime).count();	
}

int main(){
	int n, choice, t;
	while(true){
		cin >> n >> choice;
		if(n<=0 || choice<=0)break;
		if(choice==1)
			t = msCatch(n, fN); // 2e9
		else if(choice==2)
			t = msCatch(n, fN2); // 42500 约 4e4
		else
			t= msCatch(n, fNlgN); // 6e7
		cout << "time: " << t << endl;
	}
    return 0;
}

```
在本机i5处理器上，n大约10^9, n2大约4e4, nlgn大约6e7

## 递归算法的时间复杂度
同一道题目，同样使用递归算法，有的同学会写出了$O(n)$的代码，有的同学就写出了$O(\log n)$的代码
例如：求x的n次方
最直观的方式应该就是，一个for循环求出结果
```C++
// O(n)
int fun1(int x, int n){
    int res = 1;
    for(int i=0; i<n; i++){
        res *= x;
    }
    return res;
}
```
如果此时没有思路，不要说：我不会，我不知道了等等。可以和面试官探讨一下，询问：“可不可以给点提示”。面试官提示：“考虑一下递归算法”。一些同学可能一看到递归就想到了O(log n)，其实并不是这样，递归算法的时间复杂度本质上是要看: **递归的次数 * 每次递归中的操作次数**。
How about this one?
```c++
int func3(int x, int n) {
    if (n == 0) {
        return 1;
    }
    if (n % 2 == 1) {
        return function3(x, n / 2) * function3(x, n / 2)*x;
    }
    return function3(x, n / 2) * function3(x, n / 2);
}
```
实际上，上述代码的运行效率仍然是O(n)， 若作出实际递归树，每个分叉都算到了。所以应当想办法减少递归调用次数，如下为O(lg n)的解法
```c++
long recOptimize(int x, int y){
	if(y == 1)return x;
	else {
		int t = recOptimize(x, y/2);
		return y%2==1? t*t*x : t*t;
	}
}
```

## 空间复杂度
1. 空间复杂度是考虑程序运行时占用内存的大小
2. 不要以为空间复杂度就已经精准的掌握了程序的内存使用大小，很多因素会影响程序真正内存使用大小，例如编译器的内存对齐，编程语言容器的底层实现等等这些都会影响到程序内存的开销

## 斐波那契数列的计算
两种实现
1. 原始定义
```C++
long fib(int n){
    if (n==1 || n==2)return 1;
    return fib(n-1)+fib(n-2);
}
// 每计算一个点都要向下分两个叉，每个分支都要分别计算，递归深度为n
// 因而复杂度为指数级O(2^n)
```
2. 如何减少一个递归调用？
```C++
long fib3(int f, int s, int n){
    if(n==1 || n==2)return 1;
    if(n==3)return f+s;
    return fib3(s, f+s, n-1);
}
// fib(n) = fib3(1, 1, n)
```
都知道二分查找的时间复杂度是O(logn)，那么递归二分查找的空间复杂度是多少呢？

我们依然看 每次递归的空间复杂度和递归的深度
再来看递归的深度，二分查找的递归深度是logn ，递归深度就是调用栈的长度，那么这段代码的空间复杂度为 1 * logn = O(logn)。


## Referrence code

![](https://img-blog.csdnimg.cn/20201208231559175.png)
$// TODO: 2$
```c++
// 版本二
int fibonacci(int first, int second, int n) {
    if (n <= 0) {
        return 0;
    }
    if (n < 3) {
        return 1;
    }
    else if (n == 3) {
        return first + second;
    }
    else {
        return fibonacci(second, first + second, n - 1);
    }
}

## 不同语言的内存管理
* C/C++这种内存堆空间的申请和释放完全靠自己管理
* Java 依赖JVM来做内存管理，不了解jvm内存管理的机制，很可能会因一些错误的代码写法而导致内存泄漏或内存溢出
* Python内存管理是由私有堆空间管理的，所有的python对象和数据结构都存储在私有堆空间中。程序员没有访问堆的权限，只有解释器才能操作。

例如Python万物皆对象，并且将内存操作封装的很好，所以**python的基本数据类型所用的内存会要远大于存放纯数据类型所占的内存**.

**程序运行堆栈**
![](https://img-blog.csdnimg.cn/20210309165950660.png)

**内存对齐**
编译器一般都会做内存对齐的优化操作，也就是说当考虑程序真正占用的内存大小的时候，也需要认识到内存对齐的影响。
```C++
struct node{
   int num;
   char cha;
}st; // 8
```
