import numpy as np
import math

#設定np輸出格式
np.set_printoptions(precision=4, suppress=True)

#避免log(0)
def log2(x):
    return math.log(max(x, 1e-15), 2)

def cross_entropy(p, q):
    r = 0
    for i in range(len(p)):
        r += p[i] * log2(1 / q[i])
    return r

def entropy(p):
    return cross_entropy(p, p)

def greedy(p_target, q_init, loops=2000, learning_rate=0.5):
    q = q_init.copy()
    losses = []
    
    for i in range(loops):
        loss = cross_entropy(p_target, q)
        losses.append(loss)
        gains = p_target / q
        
        idx_best = np.argmax(gains)  #找出最高效益，p大/q小，需要加
        idx_worst = np.argmin(gains) #p小/q大，需要減
        
        diff = gains[idx_best] - gains[idx_worst]
        #收斂判斷
        if diff < 1e-7:
            print(f"在第 {i} 次迭代時收斂")
            break
        
        #決定加減量
        amount = min(learning_rate * diff * q[idx_worst], q[idx_worst])
        q[idx_best] += amount
        q[idx_worst] -= amount
    return q

if __name__ == "__main__":
    #目標p
    p = np.array([1/2, 1/4, 1/4])
    #初始猜測q
    q_start = np.array([1/3, 1/3, 1/3])
    
    print(f"Target p: {p}")
    print(f"Target Min Loss (Entropy): {entropy(p):.5f}\n")
    
    q_final = greedy(p, q_start, loops=20000, learning_rate=0.5)
    
    print("-" * 60)
    print("Final Result:")
    
    if q_final is not None:
        print(f"Optimized q : {q_final}")
        print(f"Target    p : {p}")
        print(f"Final Loss  : {cross_entropy(p, q_final):.5f}")
        
        #驗證
        error = np.sum(np.abs(q_final - p))
        if error < 0.01:
            print("\n✅ 驗證成功：使用貪婪搜尋法找到了 q=p")
        else:
            print("\n⚠️ 未完全收斂")
    else:
        print("❌ 錯誤：函式沒有回傳值 (None)")
