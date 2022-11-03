# set, map的使用与区别
## set
set底层实现类似于二叉搜索树, set容器内存放的元素只包含一个key值, 各个元素不重复且有序, key值不允许修改。
### 插入insert
```c++
set<int> s;
s.insert(2);
s.insert(1);
s.insert(4);
s.insert(5);
s.insert(3);
s.insert(5);
for (auto e : s)
    cout << e << " ";
// 1 2 3 4 5
```
### 查找find, 删除erase, 清空clear
```c++
//时间复杂度：O(logN)----底层是搜索树
set<int>::iterator pos = s.find(3);
if (pos != s.end()){
	cout << "找到了" << endl;
    s.erase(pos);//找到了我就删，没找到要删的话会报错
    /*
    采用s.erase(3);这种操作如果没有3并不会报错，如果有3则会删除这个结点。
    找到pos位置，采用s.erase(pos);这种操作如果没有3则会报错，如果有3则会删除这个结点
    */
}
s.clear();//清掉所有数据
```
### 遍历
```c++
//新式for循环
for (auto e : s){
	cout << e << " ";
}
cout << endl;

//迭代器遍历
set<int>::iterator sit = s.begin();
while (sit != s.end()){
	cout << *sit << " ";
	sit++;
}
cout << endl;
```
### multiset
允许键值冗余, 接口与set相同, 不允许删除键值
### 初始化
```c++
set<T> s;
set<T> s(b, e);
// 其中，b和e分别为迭代器的开始和结束的标记（数组多为数据源）。
// arr,arr+sizeof(arr)/sizeof(*arr)
```
## map
有别于set的是，map是一种key(键),value(值)的形式，用来保存键和值组成的集合，键必须是唯一的，但值可以不唯一。里面的元素可以根据键进行自动排序，由于map是key_value的形式，所以map里的所有元素都是pair类型。pair里面的first被称为key(键），second被称为value(值）。它可以通过关键字查找映射关联信息value，同时根据key值进行排序。
### 插入insert
有三种方法
```c++
map<string, string> dict;
dict.insert(pair<string, string>("string", "字符串"));//模板类型pair：构造了一个匿名对象插入到map
dict.insert(make_pair("apple", "苹果"));//模板函数make_pair：偷懒了，实际调的是pair
dict.insert({ "left", "左边" });
dict.insert({ "left", "剩余" });//插入不进去了，因为key值已经有了
```
### 遍历
```c++
//新式for循环
for (const auto &e : dict){
	cout << e.first << ":" << e.second << endl;
}
cout << endl;
//迭代器遍历
map<string, string>::iterator mit = dict.begin();
while (mit != dict.end()){
	cout << mit->first << ":" << mit->second << endl;
	cout << (*mit).first << ":" << (*mit).second << endl;
	mit++;
}
```
### operator[ ]
`[]`可以通过key值访问对应的value值, 当key值不存在时, 就插入这条数据, 否则将对应的value修改为新的值
```c++
dict["left"] = "剩余"; // 修改 左边 为 剩余
```
### multimap
允许键值冗余
## set与map的比较
set是一种关联式容器，其特性如下：

* set以RBTree作为底层容器
* 所得元素的只有key没有value，value就是key
* 不允许出现键值重复
* 所有的元素都会被自动排序
* 不能通过迭代器来改变set的值，因为set的值就是键
* map和set一样是关联式容器，它们的底层容器都是红黑树，区别就在于map的值不作为键，键和值是分开的。它的特性如下：

* map以RBTree作为底层容器
* 所有元素都是键+值存在
* 不允许键重复
* 所有元素是通过键进行自动排序的
* map的键是不能修改的，但是其键对应的值是可以修改的
