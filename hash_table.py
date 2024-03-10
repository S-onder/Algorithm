class LinkList:
    """链表列"""
    class Node:
        """链表的节点类"""
        def __init__(self, item=None):
            self.item = item
            self.next = None

    class LinkListIterator:
        """链表的迭代器"""
        def __init__(self, node):
            self.node = node
        def __next__(self):
            if self.node:
                cur_node = self.node
                self.node = self.node.next
                return cur_node.item
            else:
                raise StopIteration

    def __init__(self, iterable=None):
        self.head = None
        self.tail = None #头尾节点都是空
        if iterable:
            self.extend(iterable) #如果有迭代器(list)，就调用extend方法

    def extend(self,iterable):
        #可以插入
        for item in iterable:
            self.append(item)
    def append(self, item):
        s = LinkList.Node(item) #先创建一个节点
        if not self.head:
            #没有头节点
            self.head = s
            self.tail = s
        else:
            #将当前节点插入到最后
            self.tail.next = s
            self.tail = s
    def find(self,item):
        #可以查找
        for node in self:
            #self是一个可迭代对象
            if node == item:
                return True
        else:
            return False

    def __iter__(self):
        #将类变成可迭代对象
        return self.LinkListIterator(self.head)
    
    def __repr__(self) -> str:
        return "<<" + ",".join(map(str,self)) + ">>"
    

class HashTable:
    """哈希表"""
    def __init__(self, size = 1000):
        self.size = size
        self.T = [LinkList() for _ in range(size)] #用拉链法解决冲突(即哈希表的每个元素都是一个链表)
    def h(self, k):
        """哈希函数"""
        return k % self.size #取余数,找到哈希表的位置
    def insert(self, k):
        """插入元素"""
        i = self.h(k)
        if self.T[i].find(k):
            print("元素已经存在")
        else:
            self.T[i].append(k) #插入元素

    def get(self, k):
        """查找元素"""
        i = self.h(k) #找到哈希表对应位置
        return self.T[i]
    

if __name__ == '__main__':
    lk = LinkList(range(10))
    # print(lk.find(2))
    ht = HashTable()
    ht.insert(10)
    ht.insert(12)
    ht.insert(2)
    ht.insert(1002)
    ht.insert(2002)
    ht.insert(3002)
    ht.insert(4002)
    print(ht.T)

    