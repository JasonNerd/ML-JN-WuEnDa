// @FileName:     C01BSE03.cpp
// @CreateTime:   2022/10/21 11:01:48
// @Author:       Rainbow River
/**
给你一个非负整数 x, 计算并返回 x 的 算术平方根 。由于返回类型是整数，结果只保留整数部分, 小数部分将被舍去 。
*/

#include <iostream>

using namespace std;

int getSqrt(int x){
    // 假设结果为a, 则必然有 a*a <= x, 令 b = a+1, b*b > x, 其中 0 <= a < x
    // 问题转化为在 [0, x] 内搜索一个数字 a 满足以上特点
    int l = 0, h = x;
    while(l < h){
        int m = l + (h-l)/2;
        if(m*m < x){
            if((m+1)*(m+1) > x) return m; // 整数平方根
            l = m + 1; // 说明(m+1)^2 <= x, 可排除 m
        }
        else if(m*m > x) h = m - 1;
        else return m; // 说明 x 恰为完全平方数
        
    }
    return l;
}

int main(){
    while (true){
        int x;
        cin >> x;
        if(x < 0) return 0;
        cout << getSqrt(x) << endl;
    }
    return 0;
}
