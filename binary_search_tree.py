# 二叉搜索树
## 满足x是一个节点。若y是x的左子节点，则y.key <= x.key；若y是x的右子节点，则y.key >= x.key.
## (左子节点比自身小,右子节点比自身大)
## 常见的操作有：查询/插入/删除

class BiTreeNode:
    """二叉树节点类"""
    def __init__(self, data):
        self.val = data
        self.left = None
        self.right = None
        self.parent = None

class BST:
    """Bi Search Tree"""
    def __init__(self, li = None):
        self.root = None
        if li:
            for val in li:
                self.insert_(val)

    
    def insert(self, node, val):
        """
        插入操作(递归的想法)
        node : 递归的插入到哪个节点
        val : 要插入的值
        """
        if not node:
            #如果该节点为空：到最底下了
            node = BiTreeNode(val) #创建该节点
        elif val < node.data:
            # 要插入的值，小于当前插入节点的值
            node.left = self.insert(node.left, val) # 希望能插入到左叶子结点去
            node.left.parent = node
        elif val > node.data:
            # 要插入的值，大于当前插入节点的值
            node.right = self.insert(node.right, val) #希望能插入到右叶子结点去
            node.right.parent = node
        return node
    
    def insert_(self,val):
        """
        插入操作(非递归)
        """
        p = self.root 
        if not p: #空树特殊处理一下
            self.root = BiTreeNode(val)
            return
        
        while True: #return会跳出循环
            if val < p.val:
                #期望往左节点走
                if p.left:#左叶子结点存在
                    p = p.left #将判断结点移动至左叶子节点
                else: #左叶子结点不存在
                    p.left = BiTreeNode(val)
                    p.left.parent = p
                    return
            elif val > p.val:
                # 期望往右节点走
                if p.right:#右叶子结点存在
                    p = p.right #将判断结点移动至右叶子结点
                else:
                    p.right = BiTreeNode(val)
                    p.right.parent = p
                    return
            else:
                return

    def pre_order(self, root):
        """前序遍历"""
        if root:
            print(root.val, end = ',')
            self.pre_order(root.left)
            self.pre_order(root.right)

    def in_order(self,root):
        """中序遍历"""
        if root:
            self.in_order(root.left)
            print(root.val, end = ',')
            self.in_order(root.right)

    def post_order(self, root):
        """后序遍历"""
        if root:
            self.post_oder(root.left)
            self.post_oder(root.right)
            print(root.val, end = ',')
    
    def query(self, node, val):
        """
        查询操作(递归思想)
        node : 当前查询的结点
        val : 要查询的值
        """
        if not node:#找不到结点了
            return None
        if node.val < val:#当前查询的结点小于要查询的值,往右节点找
            return self.query(node.right, val)
        elif node.val > val:
            return self.query(node.left, val)
        else:
            return node
        
    def query_(self, val):
        """
        查询操作(非递归思想)
        val : 要查询的值
        """
        p = self.root #先制定p为根节点
        while p:
            if p.val < val:
                #小于要查询的值
                p = p.right
            elif p.val > val:
                p = p.left
            else:
                return p.val
        #若p是空的，返回None
        return None
    
    def __remove_node_1(self, node):
        """
        情况（1）删除是叶子节点的节点
        node : 要删除的节点
        """
        if not node.parent:
            #该节点无父节点：只有一个节点的树
            self.root = None
        if node == node.parent.left: #当前节点是其父亲节点的左节点
            node.parent.left = None
        else: #当前节点是其父亲节点的右节点
            node.parent.right = None

    def __remove_node_2_1(self, node):
        """
        情况（2）删除有一个叶子节点的节点(只有一个左孩子节点)
        node : 要删除的节点
        """
        if not node.parent:
            #当前节点是根节点
            self.root = node.left
            node.left.parent = None
        elif node == node.parent.left: #当前节点是其父节点的左孩子节点
            node.parent.left = node.left
            node.left.parent = node.parent
        else:#当前节点是其父节点的右孩子节点
            node.parent.right = node.left
            node.left.parent = node.parent

    def __remove_node_2_2(self, node):
        """
        情况（2）删除有一个叶子节点的节点(只有一个右孩子节点)
        node : 要删除的节点
        """
        if not node.parent:
            self.root = node.right
            node.right.parent = None
        elif node == node.parent.left:
            node.parent.left = node.right
            node.right.parent = node.parent
        else:
            node.parent.right = node.right
            node.right.parent = node.parent
            

    def delete(self, val):
        """
        删除操作(先找节点，再删除)
        分为三种情况：
        （1）删除是叶子节点的节点
        （2）删除有一个叶子节点的节点
        （3）删除有两个叶子节点的节点
        """
        if self.root: #不是空树
            node = self.query_(val) #找到要删除值的节点
            if not node: #不存在这样的值
                return False
            if not node.left and not node.right:
                #情况（1）是叶子节点
                self.__remove_node_1(node)
            # 情况（2）有一个叶子节点的节点
            elif not node.right:#只有左孩子节点
                self.__remove_node_2_1(node)
            elif not node.left:#只有右孩子节点
                self.__remove_node_2_2(node)
            else:
                # 情况（3）有两个叶子节点的节点
                pass



            
if __name__ == '__main__':
    # tree = BST([4, 6, 7, 9, 2, 1, 3, 5, 8]) #根节点是4
    # print(tree.root.val)
    # tree.pre_order(tree.root)
    # print('')
    # tree.in_order(tree.root) #中序遍历二叉搜索树,相当于将其排序
    li = list(range(0, 30, 2))
    tree = BST(li)
    # print(tree.root.val)
    # tree.in_order(tree.root)
    print(tree.query_(3))