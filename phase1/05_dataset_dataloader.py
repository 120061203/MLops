# ============================================================
# 05_dataset_dataloader.py
# 主題：Dataset 與 DataLoader
#
# 現實中資料量很大，不能一次全部丟進模型。
# PyTorch 提供兩個工具來管理資料：
#   Dataset:    定義「如何取得第 i 筆資料」
#   DataLoader: 負責「批次（batch）切分、shuffle、多執行緒載入」
#
# 訓練時通常不是一次用全部資料，而是每次用一個 batch（例如 32 筆）。
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ─────────────────────────────────────────
# 1. 自定義 Dataset
# ─────────────────────────────────────────
# 繼承 Dataset 並實作三個方法：
#   __init__:   初始化，載入或準備資料
#   __len__:    回傳資料集大小
#   __getitem__: 回傳第 i 筆資料

class LinearDataset(Dataset):
    def __init__(self, n_samples=200):
        """產生 y = 3x + 2 的合成資料"""
        torch.manual_seed(42)
        self.X = torch.randn(n_samples, 1)          # 隨機輸入
        self.y = 3 * self.X + 2 + 0.1 * torch.randn(n_samples, 1)  # 含雜訊

    def __len__(self):
        """回傳資料集大小，DataLoader 需要這個"""
        return len(self.X)

    def __getitem__(self, idx):
        """回傳第 idx 筆資料，DataLoader 用這個取資料"""
        return self.X[idx], self.y[idx]

dataset = LinearDataset(n_samples=200)
print("資料集大小:", len(dataset))

# 取出單筆資料
x_sample, y_sample = dataset[0]
print("第 0 筆 X:", x_sample, "y:", y_sample)

# ─────────────────────────────────────────
# 2. DataLoader
# ─────────────────────────────────────────

loader = DataLoader(
    dataset,
    batch_size=32,      # 每次取 32 筆資料
    shuffle=True,       # 每個 epoch 打亂順序（訓練時建議開啟）
    num_workers=0,      # 用幾個子執行緒載入資料（Mac 上建議先設 0）
)

print(f"\n共 {len(loader)} 個 batch（{len(dataset)} 筆 / {32} = {len(loader)} 批）")

# 查看一個 batch 的 shape
for X_batch, y_batch in loader:
    print("X_batch shape:", X_batch.shape)  # (32, 1)
    print("y_batch shape:", y_batch.shape)  # (32, 1)
    break   # 只看第一個 batch

# ─────────────────────────────────────────
# 3. Train / Validation 分割
# ─────────────────────────────────────────
# 好的訓練流程需要 validation set 來監控是否過擬合（overfitting）

from torch.utils.data import random_split

full_dataset = LinearDataset(n_samples=1000)

# 80% 訓練，20% 驗證
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)  # 驗證不需要 shuffle

print(f"\n訓練集: {len(train_dataset)} 筆，驗證集: {len(val_dataset)} 筆")

# ─────────────────────────────────────────
# 4. 完整訓練流程（含 Validation）
# ─────────────────────────────────────────

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10

for epoch in range(num_epochs):

    # ── 訓練階段 ──
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        # Forward
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ── 驗證階段 ──
    model.eval()
    val_loss = 0.0

    with torch.no_grad():   # 驗證不需要梯度
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}")

# ─────────────────────────────────────────
# 5. 自定義真實場景的 Dataset：從檔案讀取
# ─────────────────────────────────────────
# 實際工作中資料通常來自 CSV、JSON、圖片等

class CSVDataset(Dataset):
    """模擬從 CSV 讀取資料的 Dataset"""

    def __init__(self, filepath=None):
        # 這裡用假資料模擬，實際應用換成 pandas 讀 CSV
        # import pandas as pd
        # df = pd.read_csv(filepath)
        # self.X = torch.tensor(df[['feature1', 'feature2']].values, dtype=torch.float32)
        # self.y = torch.tensor(df['label'].values, dtype=torch.long)

        # 模擬資料
        self.X = torch.randn(100, 2)       # 100 筆，每筆 2 個特徵
        self.y = (self.X[:, 0] > 0).long()  # 二元分類標籤 0 或 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

csv_ds = CSVDataset()
csv_loader = DataLoader(csv_ds, batch_size=16, shuffle=True)

for X_b, y_b in csv_loader:
    print(f"\nCSV Dataset - X: {X_b.shape}, y: {y_b.shape}, y dtype: {y_b.dtype}")
    break

# ─────────────────────────────────────────
# 6. 重點整理
# ─────────────────────────────────────────
print("""
=== 重點整理 ===

Dataset：
  - 繼承 torch.utils.data.Dataset
  - 實作 __len__ 和 __getitem__
  - 負責「取第 i 筆資料」的邏輯

DataLoader：
  - 傳入 Dataset，指定 batch_size、shuffle
  - 用 for loop 迭代，每次拿一個 batch
  - 訓練集開 shuffle=True，驗證/測試集開 False

訓練流程：
  for epoch in range(epochs):
      model.train()
      for X_batch, y_batch in train_loader:
          pred = model(X_batch)
          loss = loss_fn(pred, y_batch)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      model.eval()
      with torch.no_grad():
          for X_batch, y_batch in val_loader:
              ... 計算 val loss ...
""")
