// @FileName:     c3l4build.cpp
// @CreateTime:   2022/11/04 16:28:34
// @Author:       Rainbow River
/*
给你两个长度相等的字符串 s 和 t。每一个步骤中，你可以选择将 t 中的 
任一字符 替换为 另一个字符。
*/
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int minSteps(string s, string t){
    vector<int> sh(26), th(26);
    int l = s.size();
    for(int i=0; i<l; i++){
        sh[s[i]-'a']++;
        th[t[i]-'a']++;
    }
    int res = 0;
    for(int i=0; i<26; i++){
        // printf("%c: %d %d \n", i+'a', sh[i], th[i]);
        int d = sh[i] - th[i];
        res += d>0? d: -d;
    }
    return res/2;
}
int main(){
    string s = "leetcode", t = "practice";
    int stepNum = minSteps(s, t);
    cout << stepNum << endl;
    return 0;
}
