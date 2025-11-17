import numpy as np
import time

# 自定义绝对值距离函数
def dis_abs(x, y):
    return np.abs(x - y)[0]  # 处理单个元素数组

# 动态时间规整（DTW）核心算法实现
def estimate_twf(A, B, dis_func=dis_abs):

    a = time.time()
    N_A = len(A)
    N_B = len(B)

    # 初始化累积距离矩阵
    D = np.zeros([N_A, N_B])
    D[0, 0] = dis_func(A[0], B[0])  # 起点

    # 填充第一列（只能垂直移动）
    for i in range(1, N_A):
        D[i, 0] = D[i - 1, 0] + dis_func(A[i], B[0])

    # 填充第一行（只能水平移动）
    for j in range(1, N_B):
        D[0, j] = D[0, j - 1] + dis_func(A[0], B[j])

    # 填充剩余矩阵（动态规划核心）
    for i in range(1, N_A):
        for j in range(1, N_B):
            D[i, j] = dis_func(A[i], B[j]) + min(D[i - 1, j],  # 来自上方
                                                 D[i, j - 1],  # 来自左方
                                                 D[i - 1, j - 1])  # 来自对角线

    # 路径回溯部分
    i = N_A - 1  # 从终点开始回溯
    j = N_B - 1
    count = 0
    d = np.zeros(max(N_A, N_B) * 3)  # 存储路径上的局部距离
    path = []  # 存储路径坐标

    while True:
        if i > 0 and j > 0:  # 中间区域
            path.append((i, j))
            m = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

            # 优先选择对角线方向
            if m == D[i - 1, j - 1]:
                d[count] = D[i, j] - D[i - 1, j - 1]
                i = i - 1
                j = j - 1
            elif m == D[i, j - 1]:  # 水平方向
                d[count] = D[i, j] - D[i, j - 1]
                j = j - 1
            elif m == D[i - 1, j]:  # 垂直方向
                d[count] = D[i, j] - D[i - 1, j]
                i = i - 1
            count += 1

        elif i == 0 and j == 0:  # 到达起点
            path.append((i, j))
            d[count] = D[i, j]
            count += 1
            break

        elif i == 0:  # 只能水平移动
            path.append((i, j))
            d[count] = D[i, j] - D[i, j - 1]
            j = j - 1
            count += 1

        elif j == 0:  # 只能垂直移动
            path.append((i, j))
            d[count] = D[i, j] - D[i - 1, j]
            i = i - 1
            count += 1

    # 计算平均对齐距离
    mean = np.sum(d) / count
    b = time.time()
    c = b-a
    return mean, path[::-1], D,c

def find_best_match(ts1, ts2):

    len_ts1 = len(ts1)
    len_ts2 = len(ts2)
    min_distance = float('inf')
    best_start = 0

    # 转换为numpy数组以提高性能
    ts1_np = np.array(ts1)
    ts2_np = np.array(ts2)

    # 假设有6列信号
    ts2_np = ts2_np[::2]  # 如果需要采样或处理某一列，可以进行相应操作
    c_total = 0
    # 每次滑动2048步
    for start in range(0, len_ts1 - len_ts2 + 1, 4096):  # 步长为2048
        window = ts1_np[start: start + len_ts2]
        window = window[::2]
        distance, _, _,c1 = estimate_twf(window, ts2_np)
        c_total += c1
        # 如果找到更小的DTW距离，更新最优匹配
        if distance < min_distance:
            min_distance = distance
            best_start = start

    # 提取最优匹配的子序列
    zuiyou = ts1[best_start:best_start+len_ts2]

    return zuiyou, best_start,c_total


def find_best_match_raw(ts1, ts2):

    len_ts1 = len(ts1)
    len_ts2 = len(ts2)
    min_distance = float('inf')
    best_start = 0

    # 转换为numpy数组以提高性能
    ts1_np = np.array(ts1)
    ts2_np = np.array(ts2)

    c_total = 0
    # 每次滑动2048步
    for start in range(0, len_ts1 - len_ts2 + 1, 4096):  # 步长为2048
        window = ts1_np[start: start + len_ts2]
        distance, _, _,c2 = estimate_twf(window, ts2_np)
        c_total += c2
        # 如果找到更小的DTW距离，更新最优匹配
        if distance < min_distance:
            min_distance = distance
            best_start = start

    # 提取最优匹配的子序列
    zuiyou = ts1[best_start:best_start+len_ts2]

    return zuiyou, best_start,c_total



