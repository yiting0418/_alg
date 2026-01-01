import numpy as np
import matplotlib.pyplot as plt

# --- 1. 資料讀取 (保持原樣) ---
def load_data(filename):
    data, labels = [], []
    # 確保 map 對應正確
    map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('%', '@')): continue
                
                parts = line.split(',')
                if len(parts) == 5:
                    data.append([float(x) for x in parts[:4]])
                    labels.append(map.get(parts[4], -1))
        return np.array(data), np.array(labels)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {filename}，請確保檔案在正確路徑。")
        return np.array([]), np.array([])

# --- 2. K-Means 演算法實作 (新增部分) ---
def kmeans(data, k=3, max_iters=100, tol=1e-4, n_init=5):
    """
    增強版 K-Means：執行 n_init 次，回傳最好的一次結果
    """
    best_labels = None
    best_centroids = None
    best_inertia = float('inf')  # 紀錄最小誤差，初始設為無限大
    
    # 多跑幾次，避免運氣不好選到爛的起始點
    for i in range(n_init):
        # --- 單次 K-Means 流程 ---
        n_samples, _ = data.shape
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[random_indices]
        current_labels = np.zeros(n_samples)
        
        for _ in range(max_iters):
            # 1. 計算距離
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # 2. 更新質心
            new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
            
            # 處理空群集的情況 (極少見，但若隨機點選得太差可能發生)
            if np.any(np.isnan(new_centroids)):
                new_centroids = centroids # 若出錯則回退，或重新隨機
                break

            if np.linalg.norm(new_centroids - centroids) < tol:
                break
            
            centroids = new_centroids
            current_labels = labels
        
        # --- 計算這一次分群的品質 (Inertia: 每個點到其質心的距離平方和) ---
        # 距離越小，代表聚類越緊密，效果越好
        final_distances = np.min(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        inertia = np.sum(final_distances ** 2)
        
        # 如果這次的結果比之前的都好，就保留這次的結果
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = current_labels
            best_centroids = centroids
            
    return best_labels, best_centroids

# --- 3. 標籤對齊函式 (新增部分：為了解決 K-Means 標籤隨機性的問題) ---
def align_labels(true_labels, cluster_labels):
    aligned_labels = np.zeros_like(cluster_labels)
    # 對每個群集，找出它最常對應到的真實標籤
    for i in range(3):
        mask = (cluster_labels == i)
        if np.sum(mask) > 0:
            # 找出該群中出現最多次的真實標籤
            true_in_cluster = true_labels[mask]
            counts = np.bincount(true_in_cluster.astype(int))
            most_frequent = np.argmax(counts)
            aligned_labels[mask] = most_frequent
    return aligned_labels

# --- 主程式 ---

# 載入資料
x, y = load_data('iris.arff')

if len(x) > 0:
    # --- 資料切分 (KNN 使用) ---
    idx = np.arange(len(x))
    np.random.seed(42) 
    np.random.shuffle(idx)

    split_idx = int(len(x) * 0.8)
    train_idx, test_idx = idx[:split_idx], idx[split_idx:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # --- KNN 部分 (保持原邏輯) ---
    def euc(a, b):
        return np.sqrt(np.sum((a - b)**2))

    def knn(x_train, y_train, test_point, k=5):
        distances = []
        for i, train_point in enumerate(x_train):
            dist = euc(test_point, train_point)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        k_labels = [label for _, label in k_nearest]
        # 處理沒有眾數的情況，預設取第一個
        if not k_labels: return 0
        prediction = max(set(k_labels), key=k_labels.count)
        return prediction

    # KNN 預測
    k_neighbors = 5
    knn_correct = 0
    knn_predictions = []

    for i, point in enumerate(x_test):
        pred = knn(x_train, y_train, point, k=k_neighbors)
        knn_predictions.append(pred)
        if pred == y_test[i]:
            knn_correct += 1

    knn_acc = (knn_correct / len(x_test)) * 100

    # --- K-Means 部分 (執行並計算準確度) ---
    # K-Means 通常是對"所有資料"或"訓練資料"進行分群，這裡我們對全部 x 進行分群來觀察結構
    raw_cluster_labels, centroids = kmeans(x, k=3)
    
    # 對齊標籤以便比較 (因為 K-Means 標籤 0 不一定等於 Setosa)
    kmeans_aligned_labels = align_labels(y, raw_cluster_labels)
    
    # 計算 K-Means 的"分群純度/準確率"
    kmeans_acc = np.mean(kmeans_aligned_labels == y) * 100

    # --- 結果輸出 ---
    print("="*30)
    print(f"KNN (監督式) 測試集準確率: {knn_acc:.2f}%")
    print(f"K-Means (非監督式) 全體分群準確率: {kmeans_acc:.2f}%")
    print("="*30)

    # --- 視覺化比較 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 圖 1: KNN 結果 (針對 Test Set)
    axes[0].scatter(x_train[:, 2], x_train[:, 3], c=y_train, label='Train Data', alpha=0.3, cmap='viridis')
    scatter1 = axes[0].scatter(x_test[:, 2], x_test[:, 3], c=knn_predictions, marker='x', s=80, label='KNN Pred', cmap='viridis')
    axes[0].set_title(f'KNN Prediction (Test Acc={knn_acc:.1f}%)')
    axes[0].set_xlabel('Petal Length')
    axes[0].set_ylabel('Petal Width')
    axes[0].legend()

    # 圖 2: K-Means 結果 (針對全體數據)
    # 畫出所有點的分群結果
    axes[1].scatter(x[:, 2], x[:, 3], c=kmeans_aligned_labels, alpha=0.6, cmap='viridis', label='Clustered Points')
    # 畫出質心
    axes[1].scatter(centroids[:, 2], centroids[:, 3], c='red', marker='*', s=200, label='Centroids')
    axes[1].set_title(f'K-Means Clustering (Global Acc={kmeans_acc:.1f}%)')
    axes[1].set_xlabel('Petal Length')
    axes[1].set_ylabel('Petal Width')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
