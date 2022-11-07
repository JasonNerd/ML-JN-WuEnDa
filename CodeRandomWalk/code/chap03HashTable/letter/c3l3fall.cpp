// @FileName:     c3l3fall.cpp
// @CreateTime:   2022/11/04 16:28:00
// @Author:       Rainbow River
/*
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。
不考虑答案输出的顺序。假设字符串只包含小写字母
*/
#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<int> findAnagrams(string s, string p){
    if(s.size() < p.size())return vector<int>();
    vector<int> pmap(26), window(26);
    // 首先计算固定模式p和s[0, p.size()-1]各字母出现次数
    for(int i=0; i<p.size(); i++){
        pmap[p[i]-'a']++;
        window[s[i]-'a']++;
    }
    vector<int> res;
    if(pmap == window) // == 操作符重载
        res.push_back(0);
    for(int i=0, j=p.size()-1 ; j<s.size()-1; i++, j++){
        window[s[i]-'a']--;
        window[s[j+1]-'a']++;
        if(window == pmap)
            res.push_back(i+1);
    }
    return res;
}
int main(){
    string s = "acbcab", p = "abc";
    vector<int> res = findAnagrams(s, p);
    for (auto r : res)
        cout << r << " ";
    return 0;
}
