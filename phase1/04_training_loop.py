# ============================================================
# 04_training_loop.py
# 主題：完整訓練迴圈
#
# 把前面學的全部組合起來：
#   - 定義模型（nn.Module）
#   - 定義 loss function
#   - 定義 optimizer（自動幫我們做梯度下降）
#   - 跑完整的 training loop
#
# 範例任務：用線性迴歸學習 y = 3x + 2 這條線
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. 準備資料
# ─────────────────────────────────────────

torch.manual_seed(42)   # 固定隨機種子，讓結果可重現

# 產生訓練資料：y = 3x + 2 加上一點雜訊
X = torch.linspace(-1, 1, 100).unsqueeze(1)    # shape: (100, 1)
y = 3 * X + 2 + 0.1 * torch.randn_like(X)     # 真實關係 + 雜訊

print("X shape:", X.shape)  # (100, 1)
print("y shape:", y.shape)  # (100, 1)

# ─────────────────────────────────────────
# 2. 定義模型（nn.Module）
# ─────────────────────────────────────────
# 所有 PyTorch 模型都繼承自 nn.Module。
# 需要實作：
#   __init__: 定義模型的層（layers）
#   forward:  定義資料如何流過這些層

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features) 就是 y = Wx + b
        # 輸入 1 個特徵，輸出 1 個值
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # forward 定義資料如何流過模型
        return self.linear(x)

model = LinearModel()
print("\n模型結構:")
print(model)

# 查看模型參數（w 和 b 是隨機初始化的）
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")

# ─────────────────────────────────────────
# 3. 定義 Loss Function 和 Optimizer
# ─────────────────────────────────────────

# Loss Function：衡量預測和真實的差距
# MSELoss = 均方誤差，迴歸任務的標準選擇
loss_fn = nn.MSELoss()

# Optimizer：自動用梯度來更新參數
# SGD（隨機梯度下降）是最基本的 optimizer
# Adam 是實務上最常用的，通常收斂更快
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# model.parameters() 告訴 optimizer 要更新哪些參數

# ─────────────────────────────────────────
# 4. 訓練迴圈（Training Loop）
# ─────────────────────────────────────────
# 這是 ML 訓練的核心結構，幾乎所有訓練程式都長這樣

num_epochs = 1000    # epoch：把整個訓練資料跑一遍算一個 epoch

loss_history = []   # 記錄每個 epoch 的 loss

for epoch in range(num_epochs):

    # ── 4.1 設定為訓練模式 ──
    # 某些層（Dropout、BatchNorm）在訓練和推論時行為不同
    model.train()

    # ── 4.2 Forward Pass：計算預測值 ──
    y_pred = model(X)

    # ── 4.3 計算 Loss ──
    loss = loss_fn(y_pred, y)

    # ── 4.4 清除上一輪的梯度 ──
    optimizer.zero_grad()

    # ── 4.5 Backward Pass：計算梯度 ──
    loss.backward()

    # 前 5 個 epoch 印出梯度，看反向傳播的結果
    if epoch < 5:
        w = model.linear.weight.grad.item()
        b = model.linear.bias.grad.item()
        print(f"Epoch {epoch+1} backward → w.grad: {w:+.4f}  b.grad: {b:+.4f}")

    # ── 4.6 更新參數 ──
    optimizer.step()

    loss_history.append(loss.item())

    # 每 20 個 epoch 印一次進度
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {loss.item():.6f}")

# 繪製 loss 曲線
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")   # log scale 讓早期和晚期的變化都看得清楚
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("\nLoss 曲線已儲存至 loss_curve.png")

# ─────────────────────────────────────────
# 5. 查看訓練結果
# ─────────────────────────────────────────

print("\n--- 訓練結果 ---")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item():.4f}")

# 真實值：w=3.0, b=2.0
# 訓練後應該非常接近

# ─────────────────────────────────────────
# 6. 推論（Inference）
# ─────────────────────────────────────────

model.eval()            # 設定為推論模式

with torch.no_grad():   # 推論時不需要計算梯度
    x_new = torch.tensor([[0.5]])   # 新的輸入
    y_new = model(x_new)
    print(f"\n輸入 x=0.5，預測 y={y_new.item():.4f}")
    print(f"真實值 y = 3*0.5 + 2 = {3*0.5 + 2:.4f}")

# ─────────────────────────────────────────
# 7. 儲存與載入模型
# ─────────────────────────────────────────

# 只存參數（推薦做法）
torch.save(model.state_dict(), "linear_model.pth")
print("\n模型已儲存至 linear_model.pth")

# 載入參數
loaded_model = LinearModel()
loaded_model.load_state_dict(torch.load("linear_model.pth"))
loaded_model.eval()
print("模型載入成功")

# ─────────────────────────────────────────
# 8. 多層神經網路（MLP）範例
# ─────────────────────────────────────────
# 實際任務通常不只一層，這是加深網路的標準方式

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential：按順序疊加層
        self.network = nn.Sequential(
            nn.Linear(1, 64),    # 輸入層 → 隱藏層
            nn.ReLU(),           # 激活函式（非線性）
            nn.Linear(64, 64),   # 隱藏層 → 隱藏層
            nn.ReLU(),
            nn.Linear(64, 1),    # 隱藏層 → 輸出層
        )

    def forward(self, x):
        return self.network(x)

mlp = MLP()
print("\nMLP 結構:")
print(mlp)

# 計算參數總量
total_params = sum(p.numel() for p in mlp.parameters())
print(f"總參數量: {total_params:,}")
