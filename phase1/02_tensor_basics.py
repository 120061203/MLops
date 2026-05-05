# ============================================================
# 02_tensor_basics.py
# 主題：PyTorch Tensor 基礎
#
# Tensor 是 PyTorch 的核心資料結構，概念上和 NumPy array 幾乎一樣，
# 但多了兩個關鍵能力：
#   1. 可以放到 GPU 上加速運算
#   2. 支援自動微分（下一個檔案會講）
# ============================================================

import torch
import numpy as np

# ─────────────────────────────────────────
# 1. 建立 Tensor
# ─────────────────────────────────────────

# 從 Python list 建立
a = torch.tensor([1.0, 2.0, 3.0])
print("tensor:", a)
print("dtype:", a.dtype)    # torch.float32（預設浮點數）
print("shape:", a.shape)    # torch.Size([3])

# 從 list of lists 建立 2D tensor（矩陣）
b = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print("\n2D tensor:\n", b)
print("shape:", b.shape)    # torch.Size([2, 2])

# 常用建立方式
zeros = torch.zeros(3, 3)           # 全 0
ones  = torch.ones(2, 4)            # 全 1
rand  = torch.randn(3, 3)           # 標準常態分佈隨機值
eye   = torch.eye(3)                # 單位矩陣
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]，類似 range()

print("\nzeros:\n", zeros)
print("arange:", arange)

# ─────────────────────────────────────────
# 2. 資料型別（dtype）
# ─────────────────────────────────────────
# ML 中常用的型別：
#   torch.float32 (float) → 模型參數的預設型別
#   torch.float16 (half)  → 省記憶體，訓練加速用
#   torch.int64   (long)  → 分類任務的標籤（class index）

float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
int_tensor   = torch.tensor([1, 2, 3],  dtype=torch.int64)

# 轉換型別
converted = float_tensor.to(torch.float16)
print("\nfloat32:", float_tensor.dtype)
print("轉成 float16:", converted.dtype)

# ─────────────────────────────────────────
# 3. Shape 操作
# ─────────────────────────────────────────

x = torch.arange(12).float()   # [0., 1., ..., 11.]
print("\noriginal:", x.shape)  # torch.Size([12])

# reshape：重新排列（和 NumPy 一樣）
x_2d = x.reshape(3, 4)
print("reshape(3,4):\n", x_2d)

# view：和 reshape 類似，但要求記憶體連續
x_view = x.view(4, 3)
print("view(4,3):\n", x_view)

# unsqueeze：增加一個維度（常用於讓單一樣本符合 batch 格式）
x_1d = torch.tensor([1.0, 2.0, 3.0])   # shape: (3,)
x_2d = x_1d.unsqueeze(0)                # shape: (1, 3)，新增 batch 維度
x_2d_col = x_1d.unsqueeze(1)            # shape: (3, 1)
print("\nunsqueeze(0):", x_2d.shape)
print("unsqueeze(1):", x_2d_col.shape)

# squeeze：移除大小為 1 的維度
print("squeeze back:", x_2d.squeeze().shape)  # 回到 (3,)

# ─────────────────────────────────────────
# 4. 基本運算
# ─────────────────────────────────────────

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\n--- 基本運算 ---")
print("a + b:", a + b)
print("a * b:", a * b)          # element-wise
print("a @ b:", a @ b)          # 內積（dot product）

# 矩陣乘法
A = torch.ones(2, 3)
B = torch.ones(3, 4)
C = A @ B                       # (2,3) @ (3,4) → (2,4)
print("\n矩陣乘法 shape:", C.shape)

# 常用數學函式
print("\ntorch.exp:", torch.exp(a))
print("torch.log:", torch.log(a))
print("torch.sum:", torch.sum(a))
print("torch.mean:", torch.mean(a))
print("torch.max:", torch.max(a))

# ─────────────────────────────────────────
# 5. 與 NumPy 互相轉換
# ─────────────────────────────────────────
# 兩者共享記憶體（CPU 上），修改一個會影響另一個

np_array = np.array([1.0, 2.0, 3.0])

# NumPy → Tensor
tensor_from_np = torch.from_numpy(np_array)
print("\nNumPy → Tensor:", tensor_from_np)

# Tensor → NumPy
tensor = torch.tensor([4.0, 5.0, 6.0])
np_from_tensor = tensor.numpy()
print("Tensor → NumPy:", np_from_tensor)

# ─────────────────────────────────────────
# 6. Device：CPU vs GPU
# ─────────────────────────────────────────
# GPU 可以大幅加速矩陣運算。在 AWS 訓練時通常使用 GPU instance。
# 訓練時要確保 model 和資料都在同一個 device 上。

# 偵測是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n使用裝置:", device)

# 建立 tensor 並移到指定 device
x = torch.randn(3, 3)
x = x.to(device)                   # 移到 GPU（若有）
print("tensor device:", x.device)

# 也可以在建立時直接指定
y = torch.ones(3, 3, device=device)
print("y device:", y.device)

# ─────────────────────────────────────────
# 7. 取值與索引（和 NumPy 相同語法）
# ─────────────────────────────────────────

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])

print("\n--- 索引 ---")
print("t[0]:", t[0])           # 第一列
print("t[1, 2]:", t[1, 2])     # 第二列第三行 → 6.0
print("t[:, 0]:", t[:, 0])     # 第一行所有列

# 取出純 Python 數值（訓練 loop 中記錄 loss 時常用）
loss_value = t[0, 0].item()
print("item():", loss_value, type(loss_value))  # float
