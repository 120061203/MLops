# ============================================================
# 01_neural_network.py
# 主題：神經網路原理
#
# Phase 1 學的是最簡單的線性模型（y = wx + b）。
# 但現實問題通常不是線性的，例如：
#   - 圖片分類（貓 vs 狗）
#   - 文字情感分析（正面 vs 負面）
#
# 解法：把多個線性層疊起來，中間加上「激活函式」製造非線性，
# 讓模型能學習更複雜的關係。
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────
# 1. 為什麼需要激活函式？
# ─────────────────────────────────────────
# 問題：多個線性層疊在一起，數學上等於一個線性層，沒有意義。
#
# Linear(Linear(x)) = W2 * (W1*x + b1) + b2
#                   = (W2*W1)*x + (W2*b1 + b2)
#                   = W*x + b   ← 還是一個線性層！
#
# 解法：在每層之間加入非線性的激活函式，打破這個限制。

# ─────────────────────────────────────────
# 2. 常用激活函式
# ─────────────────────────────────────────

x = torch.linspace(-3, 3, 7)
print("輸入 x:", x)

# ReLU：最常用，負數變 0，正數不變
relu_out = F.relu(x)
print("\nReLU(x):", relu_out)
# [-3,-2,-1, 0, 1, 2, 3] → [0, 0, 0, 0, 1, 2, 3]

# Sigmoid：把任意數壓到 (0, 1)，常用於二元分類輸出層
sigmoid_out = torch.sigmoid(x)
print("Sigmoid(x):", sigmoid_out.round(decimals=2))

# Softmax：把一組數字變成「機率分佈」，總和為 1，多分類輸出層用
logits = torch.tensor([2.0, 1.0, 0.5])   # 模型原始輸出（logits）
softmax_out = F.softmax(logits, dim=0)
print("\nSoftmax([2.0, 1.0, 0.5]):", softmax_out)
print("總和:", softmax_out.sum())          # 永遠等於 1.0

# ─────────────────────────────────────────
# 3. 多層神經網路（MLP）做分類
# ─────────────────────────────────────────
# 任務：用 2 個特徵分類 3 種類別（多分類）

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 輸入層 → 隱藏層
            nn.ReLU(),                           # 激活函式（非線性）
            nn.Linear(hidden_dim, hidden_dim),  # 隱藏層 → 隱藏層
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes), # 隱藏層 → 輸出層
            # 注意：最後一層不加 Softmax，CrossEntropyLoss 會自動處理
        )

    def forward(self, x):
        return self.network(x)   # 輸出 logits，shape: (batch, num_classes)

model = Classifier(input_dim=2, hidden_dim=64, num_classes=3)
print("\n模型結構:")
print(model)

# ─────────────────────────────────────────
# 4. CrossEntropy Loss（分類任務的標準 loss）
# ─────────────────────────────────────────
# 衡量模型預測的機率分佈和真實標籤有多遠
# 輸入：logits（模型原始輸出），labels（正確類別的 index）

loss_fn = nn.CrossEntropyLoss()

# 模擬一個 batch：4 筆資料，每筆 2 個特徵
X_batch = torch.randn(4, 2)
# 真實標籤：每筆資料屬於哪個類別（0, 1, 2）
y_batch = torch.tensor([0, 2, 1, 0])

# 前向傳播
logits = model(X_batch)
print("\nlogits shape:", logits.shape)   # (4, 3)：4 筆資料，3 個類別的分數
print("logits:\n", logits)

# 計算 loss
loss = loss_fn(logits, y_batch)
print("\nCrossEntropy Loss:", loss.item())

# 取出預測類別（機率最大的那個）
predictions = logits.argmax(dim=1)
print("預測類別:", predictions)
print("真實類別:", y_batch)

# 計算準確率
accuracy = (predictions == y_batch).float().mean()
print(f"準確率: {accuracy.item():.0%}")

# ─────────────────────────────────────────
# 5. 完整訓練：用 MLP 分類合成資料
# ─────────────────────────────────────────

torch.manual_seed(42)

# 產生三類合成資料
def make_data(n=300):
    X, y = [], []
    for cls in range(3):
        # 每類資料集中在不同中心點
        center = torch.tensor([cls * 2.0, cls * 2.0])
        points = center + torch.randn(n // 3, 2)
        X.append(points)
        y.append(torch.full((n // 3,), cls, dtype=torch.long))
    return torch.cat(X), torch.cat(y)

X, y = make_data(300)
print(f"\n資料: {X.shape}, 標籤: {y.shape}")

# 切分訓練/驗證集
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = Classifier(input_dim=2, hidden_dim=64, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    # 訓練
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 驗證
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean()
        print(f"Epoch [{epoch+1:3d}/100] "
              f"Loss: {loss.item():.4f}  "
              f"Val Loss: {val_loss.item():.4f}  "
              f"Val Acc: {val_acc.item():.0%}")

# ─────────────────────────────────────────
# 6. 重點整理
# ─────────────────────────────────────────
print("""
=== 重點整理 ===

激活函式：
  ReLU    → 隱藏層的標準選擇，計算快
  Sigmoid → 二元分類輸出層（輸出 0~1 的機率）
  Softmax → 多分類輸出層（輸出加總為 1 的機率分佈）

Loss Function：
  MSELoss          → 迴歸任務（預測數值）
  CrossEntropyLoss → 分類任務（預測類別）

輸出層設計：
  迴歸：Linear(hidden, 1) + 無激活函式
  二元分類：Linear(hidden, 1) + Sigmoid
  多分類：Linear(hidden, num_classes) + 不加（CrossEntropy 內建）
""")
