// @FileName:     C01DEE04.cpp
// @CreateTime:   2022/10/21 14:19:22
// @Author:       Rainbow River
/*
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
*/

#include <iostream>
#include <vector>

using namespace std;
vector<int> getSquare(vector<int> nums){
    int n = nums.size();
    vector<int> res(n);
    // 设定2个指针, 指向头部和尾部
    int i = 0, j = nums.size()-1, k = res.size()-1;
    while(k >= 0){
        int jsq = nums[j]*nums[j], isq = nums[i]*nums[i];
        if(jsq > isq){
            res[k--] = jsq;
            j--;
        } else{
            res[k--] = isq;
            i++;
        }
    }
    return res;
}

int main(){
   vector<int> a = {-5, -3, -2, 0, 1, 4, 5, 6};
   vector<int> b = getSquare(a);
   for (int i = 0; i < b.size(); i++) {
        cout << b[i] << " ";
   }
   return 0;
}
