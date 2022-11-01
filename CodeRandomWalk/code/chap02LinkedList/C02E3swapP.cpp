// @FileName:     C02E3swapP.cpp
// @CreateTime:   2022/11/01 15:29:46
// @Author:       Rainbow River
// 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
// 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

#include <iostream>
#include <vector>
#include "C02LinkedList.h"
using namespace std;
// g++ C02LinkedList.h C02LinkedList.cpp C02E3swapP.cpp -o a
void MyLinkedList::swapPairs(){
    if(!head) return ; // 空表直接返回
    node *p=head, *q=p->next, *dum, *r;
    dum->next = head; // 虚拟头节点
    r = dum;
    // 交换实际涉及到指向该节点的指针r, 节点指针p, 与待交换的下一节点指针q
    while(q != nullptr){
        p->next = q->next; // 更新操作
        q->next = p;
        r->next = q;
        if(head == p) // 判断头尾指针
            head = q;
        if(tail == q)
            tail == p;
        r = p;
        p = p->next;
        if(!p) return ; // 已经到终点了
        q = p->next;
        if(!q) return ; // 已经到终点了
    }
    delete dum;
}

int main(){
    MyLinkedList l;
    vector<int> a = {0};
    l.initArr(a);
    l.prtLinkedList();
    l.swapPairs();
    l.prtLinkedList();
    return 0;
}
