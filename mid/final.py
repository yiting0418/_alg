import numpy as np
import matplotlib.pyplot as plt
import random

def load_data(filename):
    data, labels = [], []
    map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('%', '@')): continue
            
            parts = line.split(',')
            if len(parts) == 5:
                data.append([float(x) for x in parts[:4]])
                labels.append(map.get(parts[4], -1))
    
    return np.array(data), np.array(labels)

x, y = load_data('iris.arff')

#資料切分
idx = np.arange(len(x))
np.random.seed(42) 
np.random.shuffle(idx)

split_idx = int(len(x) * 0.8)
train_idx, test_idx = idx[:split_idx], idx[split_idx:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

#計算歐式距離
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def knn_predict(x_train, y_train, test_point, k=5):
    distances = []
    for i, train_point in enumerate(x_train):
        dist = euclidean_distance(test_point, train_point)
        distances.append((dist, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    k_labels = [label for _, label in k_nearest]
    prediction = max(set(k_labels), key=k_labels.count)
    
    return prediction

#預測與評估
k_value = 5
correct_cnt = 0
predictions = []

for i, point in enumerate(x_test):
    pred = knn_predict(x_train, y_train, point, k=k_value)
    predictions.append(pred)
    
    if pred == y_test[i]:
        correct_cnt += 1

acc = (correct_cnt / len(x_test)) * 100
print(f"樣本數: {len(x_test)}")
print(f"正確數: {correct_cnt}")
print(f"準確率: {acc:.2f}%")

plt.figure(figsize=(8, 6))

plt.scatter(x_train[:, 2], x_train[:, 3], c=y_train, label='Train', alpha=0.6)
plt.scatter(x_test[:, 2], x_test[:, 3], c=predictions, marker='x', label='Test Predict', s=50)

plt.title(f'KNN Result (Acc={acc:.1f}%)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()