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

# （三）快速排序(Quick Sort)
## 时间复杂度：O(nlogn)
## 快排框架
# def QuickSort(data, left, right):
#     """
#     data : 数据
#     left : 左指针
#     right : 右指针
#     """
#     if left < right:
#         #保证列表存在2个值
#         mid = partition(data, left, right) #归位过程
#         QuickSort(data, left, mid-1) #左边递归归位
#         QuickSort(data, mid+1, right) #右边递归归位


def partition(li, left, right):
    tmp = li[left] #从左边开始拿出一个数字进行归为
    while left < right:
        # 只要当前两个指针范围存在不止一个数，就进行
        while left < right and li[right] >= tmp:
            # 当右指针指向的数大于tmp时，右指针左移
            right -= 1
            # print('right:', right)
        li[left] = li[right] #将右指针指向的数放到左空位置
        while left < right and li[left] <= tmp:
            # 当左指针指向的数小于tmp时，左指针右移
            left += 1
            # print('left:', left)
        li[right] = li[left] #将左指针指向的数放到右空位置
    li[left] = tmp #将tmp放到归位
    return left #返回归位的位置


def QuickSort(li,left, right):
    if left < right:
        mid = partition(li, left, right)
        QuickSort(li, left, mid-1)
        QuickSort(li, mid+1, right)


        
if __name__ == "__main__":
    import random
    # li = [random.randint(0,30) for _ in range(30)]
    # print(li)
    # # print("冒泡排序")
    # # print(BubbleSort(li))
    # print("插入排序")
    # print(InsertionSort(li))
    l1 = [5,7,4,6,3,1,2,9,8]
    # partition(l1, 0, len(l1)-1)
    # print(l1)
    QuickSort(l1, 0, len(l1)-1)
    print(l1)