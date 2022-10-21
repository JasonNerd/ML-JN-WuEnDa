// @FileName:     C01DEE02.cpp
// @CreateTime:   2022/10/21 14:18:20
// @Author:       Rainbow River
/*
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。请注意 ，必须在不复制数组的情况下原地对数组进行操作
*/
#include <iostream>
#include <vector>
using namespace std;

void movZero(vector<int> &nums){
    // 数组的元素只有两种情况: 是0 以及 不是0.
    // 难度在于这里不是删除，不能覆盖，要保持非0元素的相对顺序
    // 因而 覆盖 这一操作由 交换 替代
    int i = 0, j = 0, n = nums.size();
    for(j=0; j<n; j++){     // j是快指针, 用于遍历nums的每一个元素
        // i总是指向第一个0元素
        while(i<n && nums[i] != 0) ++i;
        if (nums[j] != 0){ // 非0元素则与0交换，此时nums[j]恰好移动到非0元素末尾, 0则被移到后面
            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
        }
    }
}

int main(){
   vector<int> nums = {0, 0, 1, 2, 3, 0, 4, 0, 0, 5, 0, 0};
   movZero(nums);
   for (int i = 0; i < nums.size(); i++) {
        cout << nums[i] << " ";
   }
   return 0;
}
