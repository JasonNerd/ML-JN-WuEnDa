## 349. 两个数组的交集
题意：给定两个数组，编写一个函数来计算它们的交集。
说明： 输出结果中的每个元素一定是唯一的。 我们可以不考虑输出结果的顺序。
但是要注意，使用数组来做哈希的题目，是因为题目都限制了数值的大小。

而这道题目没有限制数值的大小，就无法使用数组来做哈希表了。

而且如果哈希值比较少、特别分散、跨度非常大，使用数组就造成空间的极大浪费。

此时就要使用另一种结构体了，set ，关于set，C++ 给提供了如下三种可用的数据结构：
`std::set`
`std::multiset`
`std::unordered_set`
std::set和std::multiset底层实现都是红黑树，std::unordered_set的底层实现是哈希表， 使用unordered_set 读写效率是最高的，并不需要对数据进行排序，而且还不要让数据重复，所以选择unordered_set。
### 题解
```c++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> result_set; // 存放结果，之所以用set是为了给结果集去重
        unordered_set<int> nums_set(nums1.begin(), nums1.end());
        for (int num : nums2) {
            // 发现nums2的元素 在nums_set里又出现过
            if (nums_set.find(num) != nums_set.end()) {
                result_set.insert(num);
            }
        }
        return vector<int>(result_set.begin(), result_set.end());
    }
};
```

## 349. 两个数组的交集2
求两个数组的交集，但不去重，若不一致，以较少的为准
```
示例 1：

输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
示例 2:

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
```