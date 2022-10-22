// @FileName:     C013MS.cpp
// @CreateTime:   2022/10/21 16:57:36
// @Author:       Rainbow River
/*
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。
如果不存在符合条件的子数组，返回 0。
示例：
输入：s = 7, nums = [2,3,1,2,4,3] 输出：2 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
考察点: 滑动窗口
*/

#include <iostream>
#include <vector>
using namespace std;

int minSubArrLen(int s, vector<int> nums){
    int i = 0, sum = 0, j = 0, n = nums.size();
    int res = n;
    for (j=0; j<n; j++){
        sum += nums[j];
        while (sum >= s){
            int sbl = j - i + 1;
            res = sbl > res ? res : sbl;
            sum -= nums[i++];
        }
    }
    return res;
}

int main(){
   vector<int> nums = {2, 4, 4, 2, 3, 1, 5, 7, 2, 4, 3, 3};
   cout << minSubArrLen(15, nums) << endl;
   return 0;
}
