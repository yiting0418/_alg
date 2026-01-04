import matplotlib.pyplot as plt
import numpy as np
'''
策略：
在當前位置，試探上、下、左、右 四個方向（截距與斜率的加減）。
計算這四個新位置的 Loss，直接移動到 Loss 最小的那個位置。
如果四個方向都沒有比較好，就縮小步伐 (step size) 繼續找，直到步伐極小為止。
'''
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

# 貪婪演算法 (Greedy Search)
def greedy_regression(f, start_p, start_h=0.1, min_h=0.0001):
    p = list(start_p)
    h = start_h
    current_loss = f(p)
    
    print(f"Start: p={p}, loss={current_loss:.4f}")
    
    while h > min_h:
        best_p = None
        best_loss = current_loss
        
        # 定義四個鄰居方向：(截距+h, 斜率), (截距-h, 斜率), (截距, 斜率+h), (截距, 斜率-h)
        # 貪婪策略：窮舉所有鄰居，找出最好的一個
        candidates = [
            [p[0] + h, p[1]], # Intercept +
            [p[0] - h, p[1]], # Intercept -
            [p[0], p[1] + h], # Slope +
            [p[0], p[1] - h]  # Slope -
        ]
        
        found_better = False
        for cand in candidates:
            l = f(cand)
            # 如果發現比現在更好的點 (且比目前找到最好的鄰居還好)
            if l < best_loss:
                best_loss = l
                best_p = cand
                found_better = True
        
        if found_better:
            # 貪婪地移動到最好的那個位置
            p = best_p
            current_loss = best_loss
            # print(f"Moved to: {p}, loss={current_loss:.4f}") 
        else:
            # 如果四周都沒比較好，代表在目前解析度下已經是谷底
            # 縮小步伐 (h)，進行更精細的搜尋
            h = h * 0.5
            # print(f"Narrowing step to {h}")

    return p, current_loss

# 執行與繪圖
start_p = [0.0, 0.0] 
best_p, min_loss = greedy_regression(loss, start_p)

print(f'最終結果 (Greedy): p={best_p}, loss={min_loss:.4f}')

y_predicted = list(map(lambda t: best_p[0] + best_p[1]*t, x))
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line (Greedy)')
plt.legend()
plt.show()