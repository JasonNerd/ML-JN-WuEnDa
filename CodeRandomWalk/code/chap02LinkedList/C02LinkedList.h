#ifndef C02LINKEDLSIT_H
#define C02LINKEDLIST_H
#include <vector>
using namespace std;
class MyLinkedList{
    private:
        struct node{ // 单链表指针的设立
            int val;
            struct node* next;
            node() {}
            node(int val){this->val=val; this->next=nullptr;}
        };
        int list_len = 0; // 链表长度
        node* head = nullptr; // 链表头指针
        node* tail = nullptr; // 链表尾指针
    
        node* getNodeAtIndex(int index){ // index从0开始
            if(index<0 || index>=list_len) return nullptr;
            node* p = head;
            for(int i=0; i<index; i++)
                p = p->next;
            return p;
        }
    public:
        MyLinkedList() { }
        int get(int index);
        void addAtHead(int val);
        void addAtTail(int val);
        void addAtIndex(int index, int val); // 
        void deleteAtIndex(int index); // 将位置为index的节点删除
        void deleteByVal(int val); // 删除链表中值为val的节点
        void prtLinkedList(); // 打印链表
        void reverse(); // 反转链表
        void initArr(vector<int> a); // 使用a初始化链表
        void swapPairs(); // 两两交换
        void removeNthFromEnd(int n); // 删除倒数第n个节点
        void demostrate(); // 演示系统
};

#endif