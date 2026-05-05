# ============================================================
# 03_autograd.py
# 主題：自動微分（Autograd）
#
# 訓練 ML 模型的核心是「梯度下降」：
#   1. 計算 loss（預測有多差）
#   2. 對每個參數求偏微分（梯度）→ 知道要往哪個方向調整
#   3. 更新參數
#
# PyTorch 的 autograd 幫我們自動完成第 2 步，
# 不需要手動推導數學公式。
# ============================================================

import torch

# ─────────────────────────────────────────
# 1. requires_grad：告訴 PyTorch 要追蹤這個 tensor 的梯度
# ─────────────────────────────────────────

# 一般 tensor（不追蹤梯度）
x = torch.tensor(3.0)
print("requires_grad:", x.requires_grad)  # False

# 模型參數需要追蹤梯度
w = torch.tensor(2.0, requires_grad=True)   # 權重（weight）
b = torch.tensor(1.0, requires_grad=True)   # 偏置（bias）
print("\nw requires_grad:", w.requires_grad)  # True

# ─────────────────────────────────────────
# 2. 前向傳播（Forward Pass）
# ─────────────────────────────────────────
# PyTorch 會記錄所有運算，建立「計算圖（computation graph）」

x = torch.tensor(4.0)          # 輸入資料（不需要梯度）

# 模型：y = w * x + b（最簡單的線性模型）
y_pred = w * x + b             # 2.0 * 4.0 + 1.0 = 9.0
print("\ny_pred:", y_pred)

# y_pred 有 grad_fn，代表它是由有 requires_grad 的 tensor 計算而來
print("grad_fn:", y_pred.grad_fn)  # MulBackward0 之類

# ─────────────────────────────────────────
# 3. 計算 Loss
# ─────────────────────────────────────────

y_true = torch.tensor(10.0)    # 真實答案

# 均方誤差 MSE Loss：(預測 - 真實)^2
loss = (y_pred - y_true) ** 2
print("\nloss:", loss)          # (9 - 10)^2 = 1.0

# ─────────────────────────────────────────
# 4. 反向傳播（Backward Pass）
# ─────────────────────────────────────────
# backward() 讓 PyTorch 沿著計算圖反向計算所有梯度

loss.backward()

# 現在可以存取每個參數的梯度
print("\n--- 梯度 ---")
print("w.grad:", w.grad)   # d(loss)/d(w)
print("b.grad:", b.grad)   # d(loss)/d(b)

# 手動驗算：
# loss = (w*x + b - y_true)^2
# d(loss)/d(w) = 2 * (w*x + b - y_true) * x = 2 * (9-10) * 4 = -8
# d(loss)/d(b) = 2 * (w*x + b - y_true)     = 2 * (9-10)     = -2

# ─────────────────────────────────────────
# 5. 梯度下降更新參數
# ─────────────────────────────────────────
# 用梯度來更新參數：param = param - learning_rate * grad
# learning_rate（學習率）控制每次更新的步伐大小

learning_rate = 0.01

# torch.no_grad()：更新參數時不要追蹤這個操作的梯度
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

print("\n更新後 w:", w.item())
print("更新後 b:", b.item())

# ─────────────────────────────────────────
# 6. 重要：每次 backward 前要清除梯度
# ─────────────────────────────────────────
# 梯度會「累積」（add），不會自動清除。
# 訓練 loop 中每次迭代都要先 zero_grad()。

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

for step in range(3):
    y_pred = w * 4.0 + b
    loss = (y_pred - 10.0) ** 2
    loss.backward()
    print(f"\nStep {step+1} - w.grad: {w.grad}, b.grad: {b.grad}")

# 可以看到梯度不斷累積，這不是我們要的
# 解法：每次迭代後呼叫 .zero_()

print("\n--- 正確做法（每次清除梯度）---")
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

for step in range(3):
    # 清除上一輪的梯度
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    y_pred = w * 4.0 + b
    loss = (y_pred - 10.0) ** 2
    loss.backward()
    print(f"Step {step+1} - w.grad: {w.grad}, b.grad: {b.grad}")

# ─────────────────────────────────────────
# 7. torch.no_grad()：推論時關閉梯度計算
# ─────────────────────────────────────────
# 推論（inference）時不需要計算梯度，關掉可以省記憶體、加快速度

w = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = w * 3.0
    print("\ntorch.no_grad() 內：y.requires_grad =", y.requires_grad)  # False

# ─────────────────────────────────────────
# 8. 整體流程小結
# ─────────────────────────────────────────
print("\n=== 訓練一步的標準流程 ===")
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(4.0)
y_true = torch.tensor(10.0)
lr = 0.01

# Step 1: Forward pass
y_pred = w * x + b

# Step 2: Compute loss
loss = (y_pred - y_true) ** 2
print(f"loss: {loss.item():.4f}")

# Step 3: Backward pass（計算梯度）
loss.backward()

# Step 4: Update parameters
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

# Step 5: Clear gradients
w.grad.zero_()
b.grad.zero_()

print(f"updated w: {w.item():.4f}, b: {b.item():.4f}")
