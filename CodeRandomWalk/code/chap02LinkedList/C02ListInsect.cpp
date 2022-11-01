// @FileName:     C02ListInsect.cpp
// @CreateTime:   2022/11/01 14:55:47
// @Author:       Rainbow River
#include <cstdio>

using namespace std;
struct node{
    int val;
    node* next;
    node(int val){
        this->val = val;
        this->next = nullptr;
    }
};

int main(){
    // 手动创造两个相交的链表
    node *headA, *headB;
    node a1(2), a2(4), b1(1), b2(3), b3(5), c1(6), c2(7), c3(8);
    a1.next = &a2;
    a2.next = &c1;
    b1.next = &b2;
    b2.next = &b3;
    b3.next = &c1;
    c1.next = &c2;
    c2.next = &c3;
    c3.next = nullptr;
    headA = &a1;
    headB = &b1;
    // 先遍历得到长度
    node *p=headA, *q=headB;
    int lenA=0, lenB=0;
    while(p){
        p = p->next;
        lenA++;
    }
    while(q){
        q = q->next;
        lenB++;
    }
    printf("lenA=%d, lenB=%d\n", lenA, lenB);
    // 比较长度, 走到同一个相对位置
    p = headA;
    q = headB;
    if (lenA > lenB){
        int step = lenA-lenB;
        while(step > 0){
            step--;
            p = p->next;
        }
    }
    if (lenB > lenA){
        int step = lenB-lenA;
        while(step > 0){
            step--;
            q = q->next;
        }
    }
    while(p != nullptr && q != nullptr){
        if(p == q){
            break;
        }
        p = p->next;
        q = q->next;
    }
    printf("p == %p\n", p);
    if(p != nullptr)printf("p->val=%d\n", p->val);
    return 0;
}
