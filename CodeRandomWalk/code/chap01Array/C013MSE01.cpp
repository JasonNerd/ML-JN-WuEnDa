// @FileName:     C013MSE01.cpp
// @CreateTime:   2022/10/21 17:08:11
// @Author:       Rainbow River

/*
你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树上的水果 种类 。
你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：
你只有 两个 篮子，并且每个篮子只能装 单一类型 的水果。每个篮子能够装的水果总量没有限制。
你可以选择任意一棵树开始采摘，你必须从 每棵 树（包括开始采摘的树）上 恰好摘一个水果 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
给你一个整数数组 fruits ，返回你可以收集的水果的 最大 数目.

输入：fruits = [0,1,2,2]
输出：3
解释：可以采摘 [1,2,2] 这三棵树。
如果从第一棵树开始采摘，则只能采摘 [0,1] 这两棵树。

*/
#include <iostream>
#include <vector>
using namespace std;
int pickFruits(vector<int> fruits){
    int i=0, j=0, n=fruits.size(), res=0; // i是指从第i棵树开始采摘
    int f0=-1, f1=-1; // 记录两种水果的种类, 或者两个篮子, -1表示空
    for (j=0; j<n; j++){
        if(f0 == -1){
            // 遇到了一种新的果子
            f0 = fruits[j]; // 放进第一个篮子
        }
        if(f0 != -1 && fruits[j] != f0 && f1 == -1){
            // 已经遇到了一种果子, 遇到了第二种果子
            f1 = fruits[j]; // 那么放进另一个篮子
        }
        if (f0 != -1 && f1 != -1 && fruits[j] != f0 && fruits[j] != f1){
            // 遇到了第三种果子，并且篮子已经装了两种果子，停止采摘
            int t = j - i;
            res = t > res? t : res;

        }
    }
}

int main(){
   vector<int> nums = {2, 3, 2, 1, 2, 3, 0, 2, 2, 3, 1, 2, 1, 2, 3};

   return 0;
}
/*
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        int n = fruits.size();
        unordered_map<int, int> cnt;

        int left = 0, ans = 0;
        for (int right = 0; right < n; ++right) {
            ++cnt[fruits[right]];
            while (cnt.size() > 2) {
                auto it = cnt.find(fruits[left]);
                --it->second;
                if (it->second == 0) {
                    cnt.erase(it);
                }
                ++left;
            }
            ans = max(ans, right - left + 1);
        }
        return ans;
    }
};

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/fruit-into-baskets/solution/shui-guo-cheng-lan-by-leetcode-solution-1uyu/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

*/