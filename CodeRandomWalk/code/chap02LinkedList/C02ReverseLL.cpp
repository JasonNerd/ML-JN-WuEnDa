// @FileName:     C02ReverseLL.cpp
// @CreateTime:   2022/10/27 10:21:47
// @Author:       Rainbow River

#include <iostream>
#include "C02LinkedList.h"
#include<vector>

using namespace std;
void MyLinkedList::reverse(){
    node * dummyHead = new node(0);
    if (list_len <= 1) return;
    node *p = head, *q = head->next;
    while (q){
        node* r = q->next;
        q->next = p;
        p = q;
        q = r;
    }
    head->next = nullptr;
    node * r = head;
    head = tail;
    tail = r;
}

int main(){
    vector<int> a = {2, 3, 5, 7, 6, 8};
    MyLinkedList l; 
    l.initArr(a);
    l.prtLinkedList();
    l.reverse();
    l.prtLinkedList();
    return 0;
}
