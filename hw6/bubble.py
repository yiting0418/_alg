import matplotlib.pyplot as plt
import numpy as np

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

# 氣泡改良法
def bubble_improvement(f, p, step=0.1, min_step=0.0001):
    p = list(p) 
    current_loss = f(p)
    
    print(f"Initial: p={p}, loss={current_loss:.4f}")
    
    # 多次遍歷
    # 當step還夠大時繼續執行
    while step > min_step:
        changed = False # 標記這一輪是否有變更
        
        # 遍歷每一個參數
        # i=0 -> 截距; i=1 -> 斜率
        for i in range(len(p)):
            
            # 嘗試增加step或減少step
            candidates = [
                (p[i] + step, 'plus'),
                (p[i] - step, 'minus')
            ]
            
            original_val = p[i]
            
            for val, direction in candidates:
                p[i] = val # 試探性修改
                new_loss = f(p)
                
                # 如果誤差變小
                if new_loss < current_loss:
                    current_loss = new_loss
                    changed = True # 標記有變動
                    break 
                else:
                    # 沒變好，還原回去
                    p[i] = original_val
        
        # 如果這一輪掃描下來，所有參數都沒變動，代表這個step已經無法更好了
        # 這時候縮小step再試試看
        if not changed:
            step = step * 0.5
            
    return p, current_loss

# 執行與繪圖
start_p = [0.0, 0.0] 
best_p, min_loss = bubble_improvement(loss, start_p)

print(f'最終結果 (Bubble Improvement): p={best_p}, loss={min_loss:.4f}')

# 繪圖
y_predicted = list(map(lambda t: best_p[0] + best_p[1]*t, x))
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line (Bubble Style)')
plt.legend()
plt.show()