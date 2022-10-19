/**
 * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
 * 如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
 * 你可以假设数组中无重复元素。
 * 2022-10-18-20:08
*/
#include<vector>
#include<iostream>

using namespace std;
vector<int> data = {3, 5, 6, 11, 23, 26, 30, 33, 40};

int insertIndex(int target){
    int l = 0, r = data.size(); // [l, r)
    while(l < r){
        int m = l + (r-l)/2;
        if(data[m] < target){
            l = m + 1;  // [m+1, r)
        }else if(data[m] > target){
            r = m;  // [l, m)
        }else return m; // m
    }
    if(target > data[data.size()-1])return data.size();
    return r; // 注意不是return m; 因为最后l==r时m来不及更新
}

int main(){
    int target;
    cin >> target;
    int index = insertIndex(target);
    cout << "You can find or insert ";
    cout << target << " at data[" << index << "]" << endl;
    return 0;
}

