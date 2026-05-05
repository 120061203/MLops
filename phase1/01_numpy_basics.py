# ============================================================
# 01_numpy_basics.py
# 主題：NumPy 核心操作
#
# NumPy 是 ML 的數學基礎，幾乎所有 ML 套件底層都依賴它。
# 核心概念：把資料表示成「陣列（array）」，然後對整個陣列做運算，
# 而不是用 for loop 一個一個處理。
# ============================================================

import numpy as np

# ─────────────────────────────────────────
# 1. 建立陣列
# ─────────────────────────────────────────

# 從 Python list 建立
a = np.array([1, 2, 3, 4, 5])
print("1D array:", a)           # [1 2 3 4 5]
print("dtype:", a.dtype)        # int64（NumPy 自動推斷型別）
print("shape:", a.shape)        # (5,) → 5 個元素的 1D 陣列

# 2D 陣列（矩陣）：list of lists
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print("\n2D array:\n", b)
print("shape:", b.shape)        # (2, 3) → 2 列、3 行

# 常用的建立方式
zeros = np.zeros((3, 3))        # 全 0 的 3x3 矩陣
ones  = np.ones((2, 4))         # 全 1 的 2x4 矩陣
eye   = np.eye(3)               # 3x3 單位矩陣（對角線為 1）
rand  = np.random.randn(3, 3)   # 標準常態分佈的隨機值

print("\nzeros:\n", zeros)
print("ones:\n", ones)
print("identity:\n", eye)

# ─────────────────────────────────────────
# 2. 基本運算（element-wise）
# ─────────────────────────────────────────

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

# 對應元素相加、相乘（不需要 for loop）
print("\nx + y =", x + y)       # [5. 7. 9.]
print("x * y =", x * y)        # [4. 10. 18.]
print("x ** 2 =", x ** 2)      # [1. 4. 9.]
print("sqrt(x) =", np.sqrt(x)) # [1. 1.414 1.732]

# ─────────────────────────────────────────
# 3. 矩陣乘法
# ─────────────────────────────────────────
# 注意：* 是 element-wise，矩陣乘法要用 @ 或 np.dot()

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("\nA * B (element-wise):\n", A * B)   # 對應元素相乘
print("\nA @ B (矩陣乘法):\n", A @ B)       # 真正的矩陣乘法

# ─────────────────────────────────────────
# 4. Broadcasting（廣播機制）
# ─────────────────────────────────────────
# Broadcasting 讓不同 shape 的陣列可以直接運算，
# NumPy 會自動「擴展」較小的陣列來配合較大的陣列。

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])   # shape: (3, 3)

row = np.array([10, 20, 30])     # shape: (3,)

# row 會自動擴展成 3x3，每一列都加上 [10, 20, 30]
result = matrix + row
print("\nmatrix + row (broadcasting):\n", result)

scalar = 100
print("\nmatrix + 100 (scalar broadcasting):\n", matrix + scalar)

# ─────────────────────────────────────────
# 5. 向量化 vs for loop（速度比較）
# ─────────────────────────────────────────
# ML 訓練資料量大，for loop 太慢，要用向量化操作。

import time

size = 1_000_000
a = np.random.randn(size)
b = np.random.randn(size)

# for loop 版本
start = time.time()
result_loop = [a[i] * b[i] for i in range(size)]
print(f"\nfor loop:    {time.time() - start:.4f} 秒")

# 向量化版本
start = time.time()
result_vec = a * b
print(f"vectorized:  {time.time() - start:.4f} 秒")
# 向量化通常快 10~100 倍

# ─────────────────────────────────────────
# 6. 常用統計操作
# ─────────────────────────────────────────

data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

print("\n--- 統計操作 ---")
print("mean:  ", data.mean())    # 平均值
print("std:   ", data.std())     # 標準差
print("min:   ", data.min())     # 最小值
print("max:   ", data.max())     # 最大值
print("sum:   ", data.sum())     # 總和
print("argmax:", data.argmax())  # 最大值的索引（分類任務常用）

# axis 參數：指定沿哪個維度做運算
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("\nmatrix sum along axis=0 (每欄加總):", matrix.sum(axis=0))  # [5 7 9]
print("matrix sum along axis=1 (每列加總):", matrix.sum(axis=1))   # [6 15]

# ─────────────────────────────────────────
# 7. Shape 操作（ML 中非常常用）
# ─────────────────────────────────────────

x = np.arange(12)               # [0, 1, 2, ..., 11]
print("\noriginal shape:", x.shape)   # (12,)

# reshape：重新排列成不同形狀，元素總數不變
x_2d = x.reshape(3, 4)
print("reshape(3,4):\n", x_2d)

x_3d = x.reshape(2, 2, 3)
print("reshape(2,2,3):\n", x_3d)

# -1 讓 NumPy 自動計算該維度大小
x_auto = x.reshape(4, -1)       # 4 列，行數自動算出為 3
print("reshape(4,-1):\n", x_auto)
