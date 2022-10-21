// @FileName:     C01DEE03.cpp
// @CreateTime:   2022/10/21 14:18:37
// @Author:       Rainbow River

/*
给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。注意：如果对空文本输入退格字符，文本继续为空。
*/

#include <iostream>
#include <string>
using namespace std;

int clearStr(string &s){
    /* s为含退格的字符串, 函数将返回(修改使成为)不含退格的字符串, */
    // 字符包含两种情况: 不是#, 此时新增字符, 是#， 此时删除字符
    int i = -1, j = 0, n = s.size();
    for(j = 0; j < n; j++){
        if(s[j] != '#'){
            s[++i] = s[j];
        }else {
            if(i > -1){ // 新的s保证非空
                --i; // 退一格
            }
        }
    }
    return i+1; // 经过处理得到的正常字符串包含的字符数
}

bool cmpStr(string a, string b){
    return a.substr(0, clearStr(a)).compare(b.substr(0, clearStr(b))) == 0;
}

int main(){
   string a = "#####"; // bd
   string b = "#d##d#"; // bd
   cout << cmpStr(a, b) << endl;
   return 0;
}

/*
"####"
""
true <- 均为空

"aa#bc#d"
"##abdd#"
true <- 均为abd

"a#bbc##d"; // bd
"#ba#dc##d"; // bd
*/