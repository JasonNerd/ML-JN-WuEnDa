// @FileName:     C01DEE01.cpp
// @CreateTime:   2022/10/21 14:17:22
// @Author:       Rainbow River
/*
给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
*/
#include <iostream>
#include <vector>
using namespace std;

int ridRedundant(vector<int> &nums){
    // nums是升序数组, 设慢指针i指向新数组的最后一个元素x, 快指针j用于遍历数组
    // 那么总有nums[j] >= nums[i], nums[j]>nums[i]时指针i移动, 接受新的元素, 否则i不动
    int n = nums.size();
    if (n==0) return n; // 单独讨论空数组
    int i=0, j=0;
    for(j = 0; j<n; j++){
        if(nums[j] > nums[i]){
            nums[++i] = nums[j];
        }
    }
    return i+1;
}

int main(){
   vector<int> nums = {1, 1, 2, 3, 3, 3, 5, 6, 6, 6};
   int m = ridRedundant(nums);
   for (int i = 0; i < m; i++){
        cout << nums[i] << " ";
   }
   
   return 0;
}
