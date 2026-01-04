import matplotlib.pyplot as plt
import numpy as np
import random

# 準備數據
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

# 定義模型與損失函數
def predict(a, xt):
    return a[0] + a[1] * xt

def MSE(a, x, y):
    total = 0
    for i in range(len(x)):
        total += (y[i] - predict(a, x[i]))**2
    return total

def loss(p):
    return MSE(p, x, y)

# 爬山演算法

# 產生鄰近的隨機點
def neighbor(p, h=0.01):
    p1 = list(p) # 複製目前的點
    for i in range(len(p)):
        d = random.uniform(-h, h) # 隨機產生偏移量
        p1[i] = p[i] + d
    return p1

def hillClimbing(f, p, h=0.01):
    failCount = 0
    fnow = f(p)
    
    # 設定失敗容忍次數
    while (failCount < 10000):
        p1 = neighbor(p, h)
        f1 = f(p1)
        
        # 找低點，最小化誤差
        if f1 < fnow:
            fnow = f1
            p = p1
            failCount = 0  # 重置失敗計數
            print(f'Improvement: p={p}, loss={fnow:.4f}') 
        else:
            failCount += 1 # 累積失敗次數
            
    return p, fnow

# 執行與繪圖
# 初始猜測 [截距, 斜率]
start_p = [0.0, 0.0] 

# 執行爬山演算法
best_p, min_loss = hillClimbing(loss, start_p, h=0.05)

print(f'最終結果: p={best_p}, loss={min_loss}')

# 繪圖
y_predicted = list(map(lambda t: best_p[0] + best_p[1]*t, x))
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line (Hill Climbing)')
plt.legend()
plt.show()