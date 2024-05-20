# 排序算法

# (零) 二分查找

def binary_search(nums, target):
    """
    二分查找
    nums : 有序列表
    target : 目标值
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        #当左指针小于等于右指针时
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1 #找不到的情况


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


def partition(nums, left, right):
    """
    快速排序的归位过程
    """
    tmp = nums[left] #将左面第一个数字进行归为，暂存第一个数字
    while left < right:
        #指针不重合就进行
        while left < right and nums[right] >= tmp:
            right -= 1 #从右面开始找，直到找到一个小于tmp的位置
        nums[left] = nums[right] #将该位置放到左面
        while left < right and nums[left] <= tmp:
            left += 1 #接着从左面找，直到找到一个大于tmp的位置
        nums[right] = nums[left] #将该位置放到右面  
    nums[left] = tmp #将tmp放到中间位置 当left = right时，将tmp放到中间位置
    return left



def QuickSort(nums,left, right):
    if left < right:
        mid = partition(nums, left, right)
        QuickSort(nums, left, mid-1)
        QuickSort(nums, mid+1, right)


        
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