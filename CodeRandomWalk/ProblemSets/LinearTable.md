# 线性表题目集

### 标签
```git
[数组]  [二分法]  [双指针]  [滑动窗口]  [链表]
[排序]  [哈希算法]
```

### [001. 在排序数组中查找元素的第一个和最后一个位置](https://programmercarl.com/0034.%E5%9C%A8%E6%8E%92%E5%BA%8F%E6%95%B0%E7%BB%84%E4%B8%AD%E6%9F%A5%E6%89%BE%E5%85%83%E7%B4%A0%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E5%92%8C%E6%9C%80%E5%90%8E%E4%B8%80%E4%B8%AA%E4%BD%8D%E7%BD%AE.html)
> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。你可以假设数组中无重复元素。
```tag
[数组]  [二分法]
```
```c++
/**
 * 总结起来，一共三种方法
 * 1. O(n)遍历查找
 * 2. O(logN)二分查找，找不到直接返回{-1, -1}，否则依据index左右移动窗口
 * 3. 拆解为左边界查找和右边界查找，使用二分法，对于查找结果进行讨论
*/
/** 算法笔记说法
 * 情况一：target 在数组范围的右边或者左边，例如数组{3, 4, 5}，
 * target为2; 或者数组{3, 4, 5},target为6，此时应该返回{-1, -1}.
 * 情况二：target 在数组范围中，且数组中不存在target，
 * 例如数组{3,6,7},target为5，此时应该返回{-1, -1}.
 * 情况三：target 在数组范围中，且数组中存在target，例如数组{3,6,7},
 * target为6，此时应该返回{1, 1}
*/
```
```c++
vector<int> nums = {2, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7, 7};

int searchL(int target){
    // [l, r]
    int l=0, r=nums.size()-1;
    int left = -2;
    while(l <= r){
        int m = l + (r-l)/2;
        if(nums[m] < target){
            l = m + 1;  // [m+1, r]
        }else {
            r = m - 1; // [l, m]
            left = r;
        }
    }
    return left; // 左边界的左边一个位置
}

int searchR(int target){
    int l=0, r=nums.size()-1;
    int right = -2;
    while(l <= r){
        int m = l + (r-l)/2;
        if(nums[m] > target){
            r = m - 1;  // [l, m-1]
        }else {
            l = m + 1;  // [m+1, r]
            right = l;
        }
    }
    return right;
}

vector<int> getFLOlogN(int target){
    // 关于logN的算法，考虑二分法，此时需要注意有重复元素的可能
    int left = searchL(target); // 左边界的左一个位置
    int right = searchR(target); // 右边界的右一个位置
    if(left==-2 || right==-2)return {-1, -1}; // 情况一、元素位于列表两边
    if(right-left>1)return {left+1, right-1}; // 情况一、元素位于列表中
    return {-1, -1}; // 情况一、元素应位于列表中
}

```

### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


### Title
> Description
> Description
```tag
[数组]  [二分法]
```
```c++
// 解题思路
```
```c++
reference code
```


