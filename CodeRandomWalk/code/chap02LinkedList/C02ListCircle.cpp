// @FileName:     C02ListCircle.cpp
// @CreateTime:   2022/11/01 20:18:29
// @Author:       Rainbow River

#include <cstdio>

using namespace std;
struct node{
    int val;
    node*next;
    node(int val){
        this->val = val;
    }
};

node* entrance(node* list, node* meet){
    node *p=list, *q = meet;
    while (true){
        printf("p=%d, q=%d\n", p->val, q->val);
        if (p == q)
            return p;
        else {
            p = p->next;
            q = q->next;
        }
    }
    
}

int main(){
    // 生成一个circle list
    node *list;
    node l1(1), l2(2), l3(3), l4(4), l5(5), l6(6), l7(7);
    list = &l1;
    l1.next = &l2;
    l2.next = &l3;
    l3.next = &l4;
    l4.next = &l5;
    l5.next = &l6;
    l6.next = &l7;
    l7.next = &l4;
    // 判断是否有环
    node *slow=list, *quick=list, *meet;
    while(quick != nullptr){
        slow = slow->next;
        quick = quick->next;
        if(quick != nullptr)
            quick = quick->next;
        else{
            meet = nullptr; // 说明没有环
            break;
        } 
        if(quick == slow){
            meet = quick; // 相遇了, 有环
            break;
        }
    }
    // 继续 看入口
    if(meet != nullptr){
        printf("quick and slow meet at node %d\n", meet->val);
        node * ent = entrance(list, meet);
        printf("The entrance of circle is node %d\n", ent->val);
    }else{
        printf("There is no circle in it ... ...\n");
    }
    return 0;
}
