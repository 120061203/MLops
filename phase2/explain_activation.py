import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────
# 產生一個「線性模型解決不了」的資料
# 內圈是類別 0，外圈是類別 1
# ─────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

def make_circles(n=200):
    theta = np.linspace(0, 2 * np.pi, n // 2)
    # 內圈（類別 0）
    r0 = 0.5 + np.random.randn(n // 2) * 0.05
    X0 = np.stack([r0 * np.cos(theta), r0 * np.sin(theta)], axis=1)
    # 外圈（類別 1）
    r1 = 1.0 + np.random.randn(n // 2) * 0.05
    X1 = np.stack([r1 * np.cos(theta), r1 * np.sin(theta)], axis=1)
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return torch.tensor(X), torch.tensor(y)

X, y = make_circles(200)

# ─────────────────────────────────────────
# 定義兩個模型
# ─────────────────────────────────────────

# 模型 A：沒有激活函式（純線性，不管疊幾層都一樣）
class LinearOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 2),
        )
    def forward(self, x):
        return self.net(x)

# 模型 B：有激活函式（可以學非線性邊界）
class WithReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )
    def forward(self, x):
        return self.net(x)

def train(model, X, y, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model_linear = LinearOnly()
model_relu   = WithReLU()

train(model_linear, X, y)
train(model_relu,   X, y)

# ─────────────────────────────────────────
# 畫出決策邊界
# ─────────────────────────────────────────

def plot_boundary(ax, model, X, y, title):
    # 建立格點
    x_min, x_max = -1.4, 1.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(x_min, x_max, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid).argmax(dim=1).numpy()

    zz = preds.reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap='coolwarm')
    ax.contour(xx, yy, zz, colors='gray', linewidths=1.5)  # 決策邊界

    # 畫資料點
    colors = ['steelblue', 'darkorange']
    labels = ['類別 0（內圈）', '類別 1（外圈）']
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls],
                   label=labels[cls], s=20, alpha=0.8)

    # 計算準確率
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == y).float().mean().item()

    ax.set_title(f"{title}\n準確率：{acc:.0%}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("激活函式的用途：讓模型學會非線性邊界", fontsize=15, fontweight='bold')

plot_boundary(axes[0], model_linear, X, y, "沒有激活函式\n（只能畫直線）")
plot_boundary(axes[1], model_relu,   X, y, "有 ReLU 激活函式\n（可以畫曲線）")

# 加說明文字
axes[0].text(0, -1.3, "不管疊幾層，本質上還是一條直線\n永遠無法把內圈和外圈分開",
             ha='center', fontsize=10, color='darkred',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

axes[1].text(0, -1.3, "ReLU 讓模型能畫出圓形邊界\n成功把兩個類別分開",
             ha='center', fontsize=10, color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig("/Users/songlin.chen/Documents/MLops/phase2/explain_activation.png", dpi=150, bbox_inches='tight')
print("圖片儲存完成")
