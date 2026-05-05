# ============================================================
# 02_backprop.py
# 主題：反向傳播（Backpropagation）
#
# Phase 1 學了單層模型的梯度計算。
# 這個檔案說明：多層網路中，梯度是怎麼從 loss 一層一層往回傳的。
#
# 核心概念：Chain Rule（連鎖律）
#   如果 loss 依賴 z，z 依賴 w，那麼：
#   d(loss)/d(w) = d(loss)/d(z) * d(z)/d(w)
# ============================================================

import torch
import torch.nn as nn

# ─────────────────────────────────────────
# 1. 梯度怎麼在多層網路中流動
# ─────────────────────────────────────────
# 用一個簡單的兩層網路示範

torch.manual_seed(0)

# 手動建立兩層的參數
W1 = torch.randn(4, 2, requires_grad=True)   # 第一層權重 (4個神經元, 2個輸入)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(1, 4, requires_grad=True)   # 第二層權重 (1個輸出, 4個神經元)
b2 = torch.zeros(1, requires_grad=True)

x = torch.tensor([1.0, 2.0])                 # 輸入
y_true = torch.tensor([1.0])                 # 真實值

# 前向傳播：資料從輸入流向輸出
z1 = W1 @ x + b1          # 第一層線性運算，shape: (4,)
a1 = torch.relu(z1)        # 激活函式
z2 = W2 @ a1 + b2          # 第二層線性運算，shape: (1,)
loss = (z2 - y_true) ** 2  # MSE Loss

print("前向傳播：")
print(f"  z1 (第一層輸出): {z1.detach()}")
print(f"  a1 (ReLU 後):   {a1.detach()}")
print(f"  z2 (預測值):    {z2.item():.4f}")
print(f"  loss:           {loss.item():.4f}")

# 反向傳播：PyTorch 自動計算所有梯度
loss.backward()

print("\n反向傳播後的梯度：")
print(f"  W2.grad: {W2.grad}")
print(f"  W1.grad:\n{W1.grad}")

# ─────────────────────────────────────────
# 2. 觀察梯度消失問題（Vanishing Gradient）
# ─────────────────────────────────────────
# 深層網路中，梯度經過多層連鎖相乘，可能越來越小，
# 導致前面的層幾乎學不到東西。

print("\n=== 梯度大小隨層數的變化 ===")

# 用 Sigmoid 激活函式（容易造成梯度消失）
class DeepNetSigmoid(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(10, 10), nn.Sigmoid()]
        layers.append(nn.Linear(10, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 用 ReLU 激活函式（緩解梯度消失）
class DeepNetReLU(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(10, 10), nn.ReLU()]
        layers.append(nn.Linear(10, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

x = torch.randn(1, 10)
y = torch.tensor([[1.0]])

for NetClass, name in [(DeepNetSigmoid, "Sigmoid"), (DeepNetReLU, "ReLU")]:
    model = NetClass(num_layers=5)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    # 取第一層的梯度大小
    first_layer_grad = model.network[0].weight.grad.abs().mean().item()
    last_layer_grad  = model.network[-1].weight.grad.abs().mean().item()

    print(f"\n{name} 激活函式（5層網路）：")
    print(f"  第一層梯度均值: {first_layer_grad:.6f}")
    print(f"  最後層梯度均值: {last_layer_grad:.6f}")
    print(f"  梯度比例: {first_layer_grad / (last_layer_grad + 1e-10):.4f}")

# Sigmoid 的第一層梯度會遠小於最後層 → 梯度消失
# ReLU 的梯度相對穩定

# ─────────────────────────────────────────
# 3. 觀察每層的梯度（用 hook）
# ─────────────────────────────────────────
# hook 讓我們在前向/反向傳播時「偷看」中間層的值

print("\n=== 各層梯度大小 ===")

model = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1)
)

grad_log = {}

def make_hook(name):
    def hook(grad):
        grad_log[name] = grad.abs().mean().item()
    return hook

# 在每一層的輸出上註冊 hook
x = torch.randn(1, 4)
out1 = model[1](model[0](x));  out1.register_hook(make_hook("Layer1"))
out2 = model[3](model[2](out1)); out2.register_hook(make_hook("Layer2"))
out3 = model[4](out2);           out3.register_hook(make_hook("Layer3(output)"))

loss = out3.mean()
loss.backward()

for name, grad_val in grad_log.items():
    print(f"  {name}: {grad_val:.6f}")

# ─────────────────────────────────────────
# 4. 為什麼 Transformer 能訓練很深的網路？
# ─────────────────────────────────────────
# 兩個關鍵設計：
#   1. Residual Connection（殘差連接）：讓梯度有「捷徑」直接回傳
#   2. Layer Normalization：穩定每層的數值範圍

class ResidualBlock(nn.Module):
    """殘差連接示範：output = F(x) + x"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm   = nn.LayerNorm(dim)
        self.relu   = nn.ReLU()

    def forward(self, x):
        # 殘差連接：把輸入 x 直接加到輸出上
        residual = x
        out = self.linear(x)
        out = self.relu(out)
        out = out + residual    # ← 這就是殘差連接，讓梯度可以直接流過
        out = self.norm(out)    # ← Layer Norm 穩定數值
        return out

block = ResidualBlock(dim=8)
x = torch.randn(1, 8)
out = block(x)
print(f"\n殘差連接：input shape {x.shape} → output shape {out.shape}")

# ─────────────────────────────────────────
# 5. 重點整理
# ─────────────────────────────────────────
print("""
=== 重點整理 ===

反向傳播：
  loss.backward() 會自動用 Chain Rule 計算所有參數的梯度
  梯度從輸出層往輸入層傳播

梯度消失：
  深層網路中，梯度經過多層相乘可能趨近 0
  Sigmoid 容易造成，ReLU 能緩解

Transformer 的解法：
  Residual Connection → 梯度可以跳過某些層直接回傳
  Layer Normalization → 穩定每層的數值，避免梯度爆炸或消失
  這兩個設計讓 Transformer 可以疊到 100+ 層
""")
