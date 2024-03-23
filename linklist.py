# 链表
## 链表是由一系列节点组成的元素集合
## 每个节点包括两部分：当前节点的值和指向下一个节点的指针
## 创建链表可以用头插法和尾插法
### （一）头插法：新节点插入到链表的头部
### （二）尾插法：新节点插入到链表的尾部
## 链表插入操作
### p.next = curent.next
### current.next = p
## 链表删除操作
### p = current.next
### current.next = current.next.next
### del p

class Node:
    def __init__(self, item):
        self.item = item
        self.next = None

def create_linklist_head(li):
    """
    头插法创建链表 : 倒序结果
    """
    head = Node(li[0])
    for i in li[1:]:
        node = Node(i)
        node.next = head
        head = node
    return head
def create_linklist_tail(li):
    """
    尾插法创建链表 : 正序结果
    """
    head = Node(li[0])
    tail = head
    for i in li[1:]:
        node = Node(i)
        tail.next = node
        tail = node
    return head

def print_linklist(lk):
    """
    打印链表
    """
    while lk:
        print(lk.item, end=' ')
        lk = lk.next
    print()



if __name__ == '__main__':
    import random
    a = Node(1)
    b = Node(2)
    c = Node(3)
    a.next = b
    b.next = c
    print(a.next.next.item)# a->b->c
    lk = create_linklist_head([i for i in range(20)])
    lk2 = create_linklist_tail([i for i in range(20)])
    print_linklist(lk)
    print_linklist(lk2)
