// @FileName:     c3l1Val.cpp
// @CreateTime:   2022/11/04 16:26:54
// @Author:       Rainbow River
#include <iostream>
#include <string>
using namespace std;
/*
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
你可以假设字符串只包含小写字母。
示例 1: 输入: s = "anagram", t = "nagaram" 输出: true
*/
const int LETN = 26;
int hash_table[LETN];
int hash_func(char c){
    return c-'a';
}

bool isAnagram(string s, string t){
    if(s.size() != t.size()) return false;
    for (auto c: s)
        hash_table[hash_func(c)]++;
    for (auto c: t)
        hash_table[hash_func(c)]--;
    for(auto c: hash_table)
        if (c != 0)
            return false;
    return true;
}
int main(){
    string s = "nanlanan";
    string t = "lananana";
    cout << isAnagram(s, t) << endl;
    return 0;
}
