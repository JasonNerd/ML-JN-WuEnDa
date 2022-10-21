// @FileName:     C01DE.cpp
// @CreateTime:   2022/10/21 10:30:35
// @Author:       Rainbow River

// 给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，
// 并返回移除后数组的新长度。元素的顺序可以改变。
#include <iostream>
#include <vector>
using namespace std;

int delX(int x, vector<int>& nums){
    int n = nums.size();
    int i = 0, j = 0;
    // j 是遍历指针, i 总是指示新数组的末位的下一位
    while (j < n) {
        // 思路是: 每个不等于x的元素都是新数组的元素
        if (nums[j] != x){
            nums[i] = nums[j];
            i++;
        }
        j++;
    }
    return i;
}

int main(){
    int x = 5;
    vector<int> nums = {0, 1, 2, 2, 3, 0, 4, 2};
    int m = delX(x, nums);
    for (int i = 0; i < m; i++) {
        cout << nums[i] << " ";
    }    
    return 0;
}
/**
 * 实际上也可以从快慢指针的角度看, j是快指针寻找新数组的元素
 * 新数组就是不含有目标元素的数组, i是慢指针, 指向更新 新数组下标的位置
*/