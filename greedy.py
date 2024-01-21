# 贪心算法

#引例：找零问题
t = [100, 20, 5, 1] #零钱种类
def change_money(t,n):
    """
    给定具体数额进行找零
    贪心思想：先从最大的开始找零
    """
    m = [0 for _ in range(len(t))] #每种零钱多少张
    for i, money in enumerate(t):
        m[i] = n // money #整出，算有多少张
        n = n % money #余数，算还剩多少钱
    return m,n

# 背包问题
# 0-1背包问题：商品是离散的，只能选择拿或者不拿
# 分数背包问题：商品是连续的，可以选择拿一部分

goods = [(60, 10), (100, 20), (120, 30)] #每个商品元组表示(价格，重量)且假设已经排好序
# goods.sort(key=lambda x:x[0]/x[1], reverse=True) #排序的逻辑

def fractional_backpack(goods, w):
    """
    分数背包问题
    贪心思想：先从单位价值最高的商品开始拿
    goods : 候选商品的元组
    w : 背包容量
    return : m(每种商品取走的比例),total_value(拿走商品的总价值)
    """
    m = [0 for _ in range(len(goods))]
    total_value = 0
    for i, (good, weight) in enumerate(goods):
        # 判断背包容量是否足够该商品
        if w >= weight:
            #背包超过了商品重量
            m[i] = 1 #该商品可以全部拿走
            w -= weight #背包容量减少
            total_value += good
        else:
            m[i] = w / weight #否则只能拿走部分
            total_value += good * m[i]
            w = 0
            break
    return m,total_value


# 拼接最大数字问题
## 将数字按照字符串方式拼接，求最大的数字
### 比较麻烦的是 ：’128‘，’1286‘，要先拼接成’1281286‘和’1286128‘两者比较，取最大的。
li = [32, 94, 128, 1286, 6, 71]
from functools import cmp_to_key
def cmp(x,y):
    if x+y < y+x:
        return 1
    elif x+y > y+x:
        return -1
    else:
        return 0
    

def number_join(li):
    """
    将数字按照字符串方式拼接，求最大的数字
    贪心思想：比较两个数x，y拼接的不同两种方式：x+y和y+x
    """
    li = list(map(str, li)) #将数字转换成str
    li.sort(key=cmp_to_key(cmp))
    return "".join(li) #拼接list
    
# 活动选择问题
## 选出一个最大的互相兼容的活动集合


#活动集合list:每个元组包含了该活动的起始时间和结束时间且保证活动按照结束时间排好序
activaties = [(1,4), (3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)] 

def activaties_selection(activaties):
    """
    活动选择问题
    贪心思想：最先结束的活动一定是最优解的一部分
    activaties : 活动集合list(一定要按照结束时间排好序)
    res : 输出结果的活动list
    """
    res = [activaties[0]] #第一个活动一定排好序了
    for i in range(len(activaties)):
        if activaties[i][0] >= res[-1][1]:#循环到的活动起始时间大于输出列表中最后一个活动的结束时间，证明活动不冲突
            res.append(activaties[i]) #添加当前活动
    return res

if __name__ == '__main__':
    #找零问题
    # print(change_money(t,8324))
    # print(goods)
    # print(fractional_backpack(goods,40))
    # print(number_join(li))
    print(activaties_selection(activaties))