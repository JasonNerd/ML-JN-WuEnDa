# 2-2、删除元素
## 引例
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。元素的顺序可以改变。
你不需要考虑数组中超出新长度后面的元素。
```c
示例 1: 给定 nums = [3,2,2,3], val = 3, 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。 你不需要考虑数组中超出新长度后面的元素。
示例 2: 给定 nums = [0,1,2,2,3,0,4,2], val = 2, 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。你不需要考虑数组中超出新长度后面的元素。
```

## 讲解
要知道数组的元素在内存地址中是连续的，不能单独删除数组中的某个元素，只能覆盖。
1. 暴力解法
这个题目暴力的解法就是两层for循环，一个for循环遍历数组元素 ，第二个for循环更新数组。
很明显暴力解法的时间复杂度是O(n^2)，这道题目暴力解法在leetcode上是可以过的。
2. 快慢指针
***双指针法***（快慢指针法）： 通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。定义快慢指针
快指针：寻找新数组的元素 ，新数组就是不含有目标元素的数组
慢指针：指向更新 新数组下标的位置
![](https://tva1.sinaimg.cn/large/008eGmZEly1gntrds6r59g30du09mnpd.gif)
**双指针法（快慢指针法）在数组和链表的操作中是非常常见的，很多考察数组、链表、字符串等操作的面试题，都使用双指针法。**

## 例题代码
```c++
// 时间复杂度：O(n^2)
// 空间复杂度：O(1)
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int size = nums.size();
        for (int i = 0; i < size; i++) {
            if (nums[i] == val) { // 发现需要移除的元素，就将数组集体向前移动一位
                for (int j = i + 1; j < size; j++) {
                    nums[j - 1] = nums[j];
                }
                i--; // 因为下标i以后的数值都向前移动了一位，所以i也向前移动一位
                size--; // 此时数组的大小-1
            }
        }
        return size;

    }
};
```

```c++
// 时间复杂度：O(n)
// 空间复杂度：O(1)
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slowIndex = 0;
        for (int fastIndex = 0; fastIndex < nums.size(); fastIndex++) {
            if (val != nums[fastIndex]) {
                nums[slowIndex++] = nums[fastIndex];
            }
        }
        return slowIndex;
    }
};
```

## 练习

1. [26.删除排序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/solution/shuang-zhi-zhen-shan-chu-zhong-fu-xiang-dai-you-hu/)
   给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么 nums 的前 k 个元素应该保存最终结果。将最终结果插入 nums 的前 k 个位置后返回 k 。不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
2. [283.移动零](https://leetcode.cn/problems/move-zeroes/solution/yi-dong-ling-by-leetcode-solution/)
   给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。请注意 ，必须在不复制数组的情况下原地对数组进行操作。
3. [844.比较含退格的字符串](https://leetcode.cn/problems/backspace-string-compare/solution/bi-jiao-han-tui-ge-de-zi-fu-chuan-by-leetcode-solu/)
   给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。注意：如果对空文本输入退格字符，文本继续为空。
4. [977.有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/solution/you-xu-shu-zu-de-ping-fang-by-leetcode-solution/)
   给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
   ```c
   输入：nums = [-4,-1,0,3,10]
   输出：[0,1,9,16,100]
   解释：平方后，数组变为 [16,1,0,9,100]
   排序后，数组变为 [0,1,9,16,100]
   ```
   
## 题解记录
1.