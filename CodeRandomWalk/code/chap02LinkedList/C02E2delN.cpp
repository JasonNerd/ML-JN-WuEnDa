// @FileName:     C02E2delN.cpp
// @CreateTime:   2022/11/01 15:21:08
// @Author:       Rainbow River
// 给定一个链表，删除倒数第N个节点

#include <iostream>
#include "C02LinkedList.h"
#include <vector>

using namespace std;

void MyLinkedList::removeNthFromEnd(int n){
    if (!head) return ; // 空链表直接返回
    if (n <=0 ) return ; // n不合法直接返回
    node *slow=head, *quick=head;
    int i=1;
    while(i<n && quick){
        quick = quick->next;
        ++i;
    }
    if(!quick) return ; // n 过大
    if(quick == tail){
        // 说明要删的是头节点
        deleteAtIndex(0);
        return ;
    }
    while(quick->next != tail){
        // slow指向待删除节点的前一个节点
        quick=quick->next;
        slow=slow->next;
    }
    node* p = slow->next; // 待删除节点
    slow->next = p->next;
    if(p == tail)
        tail = slow;
    delete p;
    list_len--;
}
// g++ C02LinkedList.h C02LinkedList.cpp C02E2delN.cpp -o a && a
int main(){
    vector<int> a = {0, 1, 2, 3, 4, 5, 6};
    MyLinkedList l;
    l.initArr(a);
    l.removeNthFromEnd(1);
    l.prtLinkedList();
    l.removeNthFromEnd(-1);
    l.prtLinkedList();
    l.removeNthFromEnd(8);
    l.prtLinkedList();
    l.removeNthFromEnd(6);
    l.prtLinkedList();
    l.removeNthFromEnd(3);
    l.prtLinkedList();
    return 0;
}
