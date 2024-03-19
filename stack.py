# 栈(Stack)
## 栈是一种数据集合，只能在一端进行插入或删除操作，这一端称为栈顶，另一端称为栈底。
## 栈的特点是后进先出，即最后入栈的元素最先出栈。
## 栈顶：列表最后一位元素
## 栈底：列表第一位元素
## 栈的操作：
### 入栈(push)：将元素压入栈顶
### 出栈(pop)：将栈顶元素弹出
### 取栈顶元素(top)：获取栈顶元素，不对栈做任何修改
## 一般用列表来实现栈，列表的append()和pop()方法可以实现栈的入栈和出栈操作

class Stack:
    def __init__(self):
        self.stack = []
    def push(self,element):
        self.stack.append(element)
    def pop(self):
        return self.stack.pop()
    def top(self):
        if self.stack:
            return self.stack[-1]
        else:
            return None
        
if __name__ == '__main__':
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(stack.pop()) #后进先出
