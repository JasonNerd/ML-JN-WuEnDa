/**
 * 给定一个按照升序排列的整数数组 nums，和一个目标值 target。
 * 找出给定目标值在数组中的开始位置和结束位置。
 * 如果数组中不存在目标值 target，返回 [-1, -1]。
 * 你可以设计并实现时间复杂度为 O(logn) 的算法解决此问题吗？
*/
/** 算法笔记说法
 * 情况一：target 在数组范围的右边或者左边，例如数组{3, 4, 5}，
 * target为2; 或者数组{3, 4, 5},target为6，此时应该返回{-1, -1}.
 * 情况二：target 在数组范围中，且数组中不存在target，
 * 例如数组{3,6,7},target为5，此时应该返回{-1, -1}.
 * 情况三：target 在数组范围中，且数组中存在target，例如数组{3,6,7},
 * target为6，此时应该返回{1, 1}
*/
#include <vector>
#include <iostream>
using namespace std;
// 数组元素可重复
vector<int> nums = {2, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7, 7};

vector<int> getFirstLastIndexOn(int target){
    vector<int> res = {-1, -1};
    int n = nums.size();
    for(int i=0; i<n; i++){
        if(nums[i] == target)
            res[1] = i;
    }
    for(int i=n-1; i>=0; i--){
        if(nums[i] == target)
            res[0] = i;
    }
    return res;
}

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

int main(){
    int target;
    cin >> target; // 1 2 3 4 7 8
    vector<int> res = getFLOlogN(target);
    cout << res[0] << ", " << res[1] << endl;
    return 0;
}
/**
 * 总结起来，一共三种方法
 * 1. O(n)遍历查找
 * 2. O(logN)二分查找，找不到直接返回{-1, -1}，否则依据index左右移动窗口
 * 3. O(logN)二分查找左右边界
*/

    // 首先，按正常二分法查找target的index
    // index = -1, 不在列表中，返回[-1, -1]
    // index = m(>=0), 在列表中, 不妨分开考虑:
    // 1. 先二分查找右边界r
    // 2. 再二分查找左边界l
    /* 实际分开了存不存在，所以情况一和情况二分开的原因是？  */