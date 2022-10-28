// @FileName:     C02LinkedList.cpp
// @CreateTime:   2022/10/26 21:07:05
// @Author:       Rainbow River

#include <iostream>
#include "C02LinkedList.h"
using namespace std;

int MyLinkedList::get(int index){
    node* pi = getNodeAtIndex(index);
    if(pi)return pi->val;
    else return -1;
}
void MyLinkedList::addAtHead(int val){
    node* a = new node(val);
    if(head == nullptr){ // 说明是首次头部插入
        tail = a; // 尾指针指向新节点
    }
    a->next = head;
    head = a;
    list_len++;
}
void MyLinkedList::addAtTail(int val){
    if(head == nullptr){
        addAtHead(val); // 链表为空表，等价于addAtHead
        return ;
    }
    node* a = new node(val);
    tail->next = a;
    tail = a;
    list_len++;
}
void MyLinkedList::addAtIndex(int index, int val){ // 注意是前插
    if(index <= 0){
        addAtHead(val);
    }else if(index == list_len){
        addAtTail(val);
    }else if(index > list_len){
        return ;
    }else {
        node* pre = getNodeAtIndex(index-1); // pre绝不为空
        node* a = new node(val);
        a->next = pre->next;
        pre->next = a;
        list_len++;
    }
}
void MyLinkedList::deleteAtIndex(int index){
    // 需要考虑到删除的节点是不是头节点，是不是尾节点
    // 删除后链表是不是空
    if(index<0 || index>=list_len) return ;
    if(head == nullptr) return ;
    if (index == 0){ // 删除第一个节点
        node *p = head;
        // 考虑只有一个节点, head更新后为Null
        head = head->next; // 仍然正确
        // 但此时若直接释放p, 当节点只有一个时，tail发生错误
        if(list_len == 1)
            tail = head;
        delete p;
    }else {
        node* prev = getNodeAtIndex(index-1);
        node* p = prev->next;
        prev->next = p->next;
        // 删除尾节点, 注意更新一下tail
        if(index == list_len-1)
            tail = prev;
        delete p;
    }
    list_len--;
}

void MyLinkedList::deleteByVal(int val){
    // 设置一个虚拟头节点
    node* dummyHead = new node(-1);
    dummyHead->next = head;
    node  *p = dummyHead, *q = head;
    while(q){
        if (q->val == val){
            p->next = q->next;
            if(q == tail)
                tail = p;
            delete q;
            q = p->next;
            list_len--;
        }else{
            p = p->next;
            q = q->next;
        }
    }
    head = dummyHead->next;
    delete dummyHead;
}

void MyLinkedList::prtLinkedList(){
    node *p = head;
    cout<<"MyLinkedList(";
    for(int i=0; i<list_len; i++, p=p->next){
        cout << p->val;
        if(i<list_len-1)
            cout << " > ";
    }
    cout << ")" << endl;
}

void MyLinkedList::initArr(vector<int> a){
    for(int i=0; i<a.size(); i++)
        addAtTail(a[i]);
}

// int main(){
//     MyLinkedList mylist;
//     mylist.deleteAtIndex(0);
//     mylist.prtLinkedList(); //
//     mylist.addAtIndex(3, -2);
//     mylist.deleteByVal(2);
//     mylist.prtLinkedList(); //
//     mylist.addAtTail(3);
//     mylist.addAtHead(1);
//     mylist.addAtTail(2);
//     mylist.addAtHead(2);
//     mylist.prtLinkedList(); //
//     mylist.addAtIndex(3, -1);
//     mylist.prtLinkedList(); //
//     mylist.addAtHead(3);
//     mylist.addAtTail(4);
//     mylist.addAtHead(2);
//     mylist.prtLinkedList(); //
//     mylist.addAtIndex(2, -2);
//     mylist.prtLinkedList(); //
//     mylist.addAtIndex(-1, -1);
//     mylist.addAtIndex(12, -1);
//     mylist.prtLinkedList(); //
//     mylist.deleteAtIndex(0);
//     mylist.deleteAtIndex(0);
//     mylist.prtLinkedList(); //
//     mylist.deleteAtIndex(-1);
//     mylist.deleteAtIndex(1);
//     mylist.prtLinkedList(); //
//     mylist.deleteAtIndex(8);
//     mylist.deleteAtIndex(7);
//     mylist.prtLinkedList(); //
//     mylist.deleteByVal(2);
//     mylist.prtLinkedList();
//     mylist.deleteByVal(1);
//     mylist.prtLinkedList();
//     mylist.deleteByVal(3);
//     mylist.prtLinkedList();
//     mylist.deleteByVal(-1);
//     mylist.prtLinkedList();
//     mylist.deleteByVal(4);
//     mylist.prtLinkedList();
//     system("pause");
//     return 0;
// }
