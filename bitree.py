# 二叉树
## 二叉树是一种数据结构，每个节点会分枝成两个节点，因此称为二叉树
## 将二叉树的节点定义为一个对象，节点之间通过类似链表方式来链接

# 1.创建一个二叉树
class BiTreeNode:
    """用于创建二叉树的节点"""
    def __init__(self, data):
        self.data = data
        self.lchild = None #左节点
        self.rchild = None #右节点

#创建每个叶子结点
a = BiTreeNode('A')
b = BiTreeNode('B')
c = BiTreeNode('C')
d = BiTreeNode('D')
e = BiTreeNode('E')
f = BiTreeNode('F')
g = BiTreeNode('G')

#将每个节点链接在一起
e.lchild = a
e.rchild = g
a.rchild = c
c.lchild = b
c.rchild = d
g.rchild = f

root = e #根节点
# print(root)  #形成一颗树的形状

# 2.二叉树的遍历
## (1)前序遍历 : EACBDGF
## 从根节点出发，不断的遍历左节点直到最后，接着遍历右节点。从头至尾遍历

def pre_order(root):
    """前序遍历二叉树"""
    if root:
        print(root.data, end = ",")
        pre_order(root.lchild)
        pre_order(root.rchild)

 
## (2)中序遍历 : ABCDEGF
## 从根节点出发，左节点-自己-右节点

def in_order(root):
    """中序遍历二叉树"""
    if root:
        in_order(root.lchild)
        print(root.data, end=',')
        in_order(root.rchild)
## (3)后序遍历 : BDCAFGE
## 从根节点出发，左节点-右节点-自己

def post_oder(root):
    """后序遍历二叉树"""
    if root:
        post_oder(root.lchild)
        post_oder(root.rchild)
        print(root.data, end = ',')

## (4)层次遍历 : EAGCFBD
## 根据树的层次来遍历
        
from collections import deque
def level_order(root):
    """层次遍历二叉树"""
    queue = deque() # 创建一个队列
    queue.append(root) 
    while len(queue) >0: #只要队不空
        node = queue.popleft() #
        print(node.data, end = ',')
        if node.lchild:
            queue.append(node.lchild)
        if node.rchild:
            queue.append(node.rchild)


if __name__ == '__main__':
    print(root.data) #根节点
    print(root.rchild.rchild.data) #某一部分节点
    pre_order(root) #前序遍历二叉树
    in_order(root) #中序遍历二叉树
    post_oder(root) #后序遍历二叉树
    level_order(root) #层次遍历二叉树