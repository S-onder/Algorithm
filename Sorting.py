# 排序算法

# (一) 冒泡排序(Bubble Sort)
## 时间复杂度：O(n^2)
def BubbleSort(li):
    """
    冒泡排序
    总共会排序n-1趟，每次排序n-1-i次
    """
    for i in range(len(li) - 1):
        # 第i趟排序
        for j in range(len(li)-i-1):
            if li[j] > li[j+1]:
                # 指针大于后面的数
                li[j], li[j+1] = li[j+1], li[j]
    return li

# (二) 插入排序(Insertion Sort)
## 时间复杂度：O(n^2)
def InsertionSort(li):
    """
    插入排序
    有序指针i往右移动，无序指针j往左移动
    """
    for i in range(1, len(li)):
        # i表示无序序列第一个
        tmp = li[i]
        j = i - 1 #j表示有序序列最后一个
        while j >=0 and li[j] > tmp:
            li[j+1] = li[j]
            j -= 1
        li[j+1] = tmp #指针移动结束，插入tmp
    return li

if __name__ == "__main__":
    import random
    li = [random.randint(0,30) for _ in range(30)]
    print(li)
    # print("冒泡排序")
    # print(BubbleSort(li))
    print("插入排序")
    print(InsertionSort(li))
