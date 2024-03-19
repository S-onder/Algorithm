# 队列(Queue)
## 仅允许在列表的一端进行插入，另一端进行删除
## 进行插入的一端称为队尾(rear)，插入动作称为进队或入队
## 进行删除的一端称为队头(front)，删除动作称为出队
## 先进先出(FIFO)原则
## 队列实现方式：环形队列、链式队列
## 队列指定长度

class Queue:
    """自己实现队列"""
    def __init__(self, size = 100):
        self.queue = [0 for _ in range(size)] #队列长度
        self.rear = 0 #队尾(进队)
        self.front = 0 #队首(出队)
        self.size = size
    
    def push(self, val):
        """进队"""
        if not self.is_filled():
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = val
        else:
            raise IndexError("push to a filled queue")

    def pop(self):
        """出队"""
        if not self.is_empty():
            self.front = (self.front + 1) % self.size
            return self.queue[self.front]
        else:
            raise IndexError("pop from an empty queue")
    
    def is_empty(self):
        """判断队列是否为空"""
        return self.front == self.rear

    def is_filled(self):
        """判断队列是否已满"""
        return (self.rear + 1) % self.size == self.front
    

# 用现成函数
# 双向队列
from collections import deque #双向队列
q = deque()
q.append(2024) #队尾进队
print(q.popleft())#队首出队
q.appendleft(2023) #队首进队
print(q.pop()) #队尾出队

    
# if __name__ == '__main__':
#     q = Queue(10)
#     for i in range(1,10):
#         q.push(i)
#     print(q.pop())
    
