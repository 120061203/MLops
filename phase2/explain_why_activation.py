import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("為什麼 ReLU 能產生非線性？", fontsize=15, fontweight='bold')

x = np.linspace(-3, 3, 400)

# ─────────────────────────────────────────
# 圖 1：線性層疊加還是線性
# ─────────────────────────────────────────
ax = axes[0]

y_linear1 = 0.5 * x + 1          # 第一層
y_linear2 = 0.8 * y_linear1 - 0.5 # 第二層（線性組合）
y_linear3 = -0.6 * y_linear2 + 2  # 第三層

ax.plot(x, y_linear1, '--', color='steelblue',  lw=2, label='第 1 層輸出')
ax.plot(x, y_linear2, '--', color='darkorange', lw=2, label='第 2 層輸出')
ax.plot(x, y_linear3, '-',  color='darkred',    lw=3, label='第 3 層輸出（還是直線！）')
ax.axhline(0, color='gray', lw=0.8, linestyle=':')
ax.set_title("線性層不斷疊加\n→ 結果永遠是直線", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlabel("輸入 x")
ax.set_ylabel("輸出")
ax.grid(True, alpha=0.3)
ax.text(0, -2, "W3*(W2*(W1*x+b1)+b2)+b3\n= 還是 W*x + b",
        ha='center', fontsize=10, color='darkred',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ─────────────────────────────────────────
# 圖 2：ReLU 如何「折」出形狀
# ─────────────────────────────────────────
ax = axes[1]

def relu(x): return np.maximum(0, x)

# 每條線是一個神經元：線性 + ReLU
neuron1 =  relu(1.5 * x - 1.0)   # 從 x=0.67 開始往上
neuron2 =  relu(-1.5 * x - 1.0)  # 從 x=-0.67 開始往上（反向）
neuron3 =  relu(x + 0.5)          # 從 x=-0.5 開始

ax.plot(x, neuron1, color='steelblue',  lw=2, label='神經元 1：relu(1.5x-1)')
ax.plot(x, neuron2, color='darkorange', lw=2, label='神經元 2：relu(-1.5x-1)')
ax.plot(x, neuron3, color='green',      lw=2, label='神經元 3：relu(x+0.5)')

# 組合：用三個神經元的加權和
combined = 1.0*neuron1 + 1.0*neuron2 - 0.8*neuron3
ax.plot(x, combined, 'k-', lw=3, label='組合後的輸出')

ax.axhline(0, color='gray', lw=0.8, linestyle=':')
ax.set_title("多個 ReLU 神經元\n→ 可以組合出任意折線", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlabel("輸入 x")
ax.set_ylabel("輸出")
ax.grid(True, alpha=0.3)

# ─────────────────────────────────────────
# 圖 3：折線越多，越接近任意曲線
# ─────────────────────────────────────────
ax = axes[2]

# 目標：用 ReLU 神經元逼近 sin(x)
target = np.sin(x)
ax.plot(x, target, 'k--', lw=2, label='目標曲線 sin(x)', zorder=5)

# 用不同數量的神經元逼近
for n_neurons, color, label in [(2, 'red', '2 個神經元'),
                                  (5, 'orange', '5 個神經元'),
                                  (20, 'steelblue', '20 個神經元')]:
    # 用均勻分佈的折點來逼近
    breakpoints = np.linspace(-3, 3, n_neurons + 2)[1:-1]
    approx = np.zeros_like(x)
    for bp in breakpoints:
        slope = np.cos(bp)  # sin 的導數
        approx += relu(slope * (x - bp)) * (1.0 / n_neurons)
        approx -= relu(-slope * (x - bp)) * (1.0 / n_neurons)
    approx = approx + np.sin(breakpoints).mean() * 0
    # 簡單示意：直接用分段線性逼近
    y_approx = np.interp(x,
                          np.linspace(-3, 3, n_neurons + 2),
                          np.sin(np.linspace(-3, 3, n_neurons + 2)))
    ax.plot(x, y_approx, color=color, lw=2, label=label, alpha=0.85)

ax.set_title("神經元越多\n→ 越接近任意曲線", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlabel("輸入 x")
ax.set_ylabel("輸出")
ax.grid(True, alpha=0.3)
ax.text(0, -1.3,
        "理論上，足夠多的 ReLU 神經元\n可以逼近任何函數（萬能逼近定理）",
        ha='center', fontsize=10, color='navy',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig("/Users/songlin.chen/Documents/MLops/phase2/explain_why_activation.png",
            dpi=150, bbox_inches='tight')
print("圖片儲存完成")
