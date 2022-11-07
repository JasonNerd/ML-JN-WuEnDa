// @FileName:     arrand.cpp
// @CreateTime:   2022/11/04 16:25:47
// @Author:       Rainbow River
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
using namespace std;


void print_vec(vector<int> v){
    int n = v.size();
    for (int i = 0; i < n; i++){
        cout << v[i];
        if (i != n-1) cout << " ";
        else cout << endl;
    }
    
}
void print_map(unordered_map<int, int> mii){
    for (auto e : mii){
        cout << e.first <<": " << e.second <<endl;
    }
}

vector<int> in_sig(vector<int>& nums1, vector<int>& nums2){
    // 求nums1与nums2的交集, 无重复
    unordered_set<int> us1, us2;
    for (auto u: nums1)
        us1.insert(u);
    for (auto u: nums2)
        us2.insert(u);
    vector<int> res;
    for(auto u: us2){
        unordered_set<int>::iterator it = us1.find(u);
        if (it != us1.end())
            res.push_back(u);
    }
    return res;
}

vector<int> in_abl(vector<int>& nums1, vector<int>& nums2){
    // 求nums1与nums2的交集, 允许重复, 重复量为最小值
    unordered_map<int, int> m1, m2;
    unordered_map<int, int>::iterator umiit;
    for(auto u: nums1){
        umiit = m1.find(u);
        if(umiit == m1.end())
            m1[u] = 1;
        else m1[u]++;
    }
    // print_map(m1);
    // cout<<"-----"<<endl;
    for(auto u: nums2){
        umiit = m2.find(u);
        if(umiit == m1.end())
            m2[u] = 1;
        else m2[u]++;
    }
    // print_map(m2);
    vector<int> res;
    for(const auto &e: m1){
        umiit = m2.find(e.first);
        if (umiit != m2.end()){
            int mc = e.second > umiit->second ? umiit->second : e.second;
            for(int i=0; i<mc; i++)
                res.push_back(e.first);
        }
    }
    return res;
}

int main(){
    vector<int> nums1 = {3, 3, 2, 1, 5, 1, 3, 4, 1};
    vector<int> nums2 = {2, 2, 3, 5, 6, 3, 5, 2};
    vector<int> r1 = in_sig(nums1, nums2); // [3, 2, 5]
    vector<int> r2 = in_abl(nums1, nums2); // [2, 3, 3, 5]
    print_vec(r1);
    print_vec(r2);
    return 0;
}
