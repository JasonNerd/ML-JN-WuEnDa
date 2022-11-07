// @FileName:     c3l5ransom.cpp
// @CreateTime:   2022/11/04 16:29:21
// @Author:       Rainbow River
/*
给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串
 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
你可以假设两个字符串均只含有小写字母
*/
#include <iostream>
#include <string>
#include <vector>
using namespace std;
bool canConstruct(string ransomNote, string magazine) {
    int n = 26, rn = ransomNote.size(), mn = magazine.size();
    vector<int> r(n), m(n);
    for(int i=0; i<rn; i++)
        r[ransomNote[i]-'a']++;
    for(int i=0; i<mn; i++)
        m[magazine[i]-'a']++;
    for(int i=0; i<n; i++)
        if(m[i] - r[i] < 0)return false;
    return true;
}
int main(){
    cout << canConstruct("a", "b");
    cout << canConstruct("aa", "ab");
    cout << canConstruct("aa", "aba");
    return 0;
}
