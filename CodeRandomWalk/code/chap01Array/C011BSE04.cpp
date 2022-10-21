// @FileName:     C01BSE04.cpp
// @CreateTime:   2022/10/21 10:58:23
// @Author:       Rainbow River
// 给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，
// 则返回 true ，否则返回 false 。

#include <iostream>

using namespace std;

bool isSquare(int x){
    int l = 0, h = x;
    while(l < h){
        int m = l + (h-l)/2;
        if (m*m < x){
            if((m+1)*(m+1) > x) return false;
            l = m + 1;
        }else if(m*m > x)
            h = m - 1;
        else 
            return true;
    }
}

int main(){
   while (true){
        int x;
        cin >> x;
        if(x < 0) return 0;
        cout << isSquare(x) << endl;
    }
   return 0;
}
