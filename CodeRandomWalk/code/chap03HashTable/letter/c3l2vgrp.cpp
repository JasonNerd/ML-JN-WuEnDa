// @FileName:     c3l2vgrp.cpp
// @CreateTime:   2022/11/04 16:27:25
// @Author:       Rainbow River
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
/*
给你一个字符串数组，请你将 字母异位词 组合在一起。
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
*/
using namespace std;
vector<vector<string>> groupAnagrams(vector<string>& strs){
    unordered_map<string, vector<string>> ht;
    vector<vector<string>> res;
    for (auto s: strs){
        string key(s);
        sort(key.begin(), key.end());
        ht[key].push_back(s);
    }
    for (auto v: ht){
        res.push_back(v.second);
    }
    return res;
}

int main(){
    vector<string> strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
    vector<vector<string>> res = groupAnagrams(strs);
    for(int i=0; i<res.size(); i++){
        vector<string> grp = res[i];
        for(int j=0; j<grp.size(); j++){
            cout << grp[j] << " ";
        }
        cout << endl;
    }
    return 0;
}
