import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Softmax 與 CrossEntropy 數學原理", fontsize=15, fontweight='bold')

# ─────────────────────────────────────────
# 圖 1：e^x 放大差距
# ─────────────────────────────────────────
ax = axes[0]

logits_sets = [
    ([2.0, 1.0, 0.5], '原始 logits\n[2.0, 1.0, 0.5]'),
    ([4.0, 2.0, 1.0], '放大 2 倍的 logits\n[4.0, 2.0, 1.0]'),
]

x = np.array([0, 1, 2])
width = 0.35

for i, (logits, label) in enumerate(logits_sets):
    logits = np.array(logits)
    exp_vals = np.exp(logits)
    softmax = exp_vals / exp_vals.sum()
    bars = ax.bar(x + i * width, softmax, width,
                  label=label, alpha=0.85,
                  color=['steelblue', 'darkorange'][i])
    for bar, val in zip(bars, softmax):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.0%}', ha='center', fontsize=9,
                color=['steelblue', 'darkorange'][i], fontweight='bold')

ax.set_xticks(x + width/2)
ax.set_xticklabels(['類別 0', '類別 1', '類別 2'])
ax.set_title("e^x 放大分數差距\n→ 讓模型更有把握的類別機率更高", fontsize=11, fontweight='bold')
ax.set_ylabel("Softmax 機率")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')
ax.text(1, 0.85, 'e^x 讓最大值\n與其他值差距更大',
        ha='center', fontsize=9, color='darkred',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ─────────────────────────────────────────
# 圖 2：-log(x) 曲線
# ─────────────────────────────────────────
ax = axes[1]

p = np.linspace(0.01, 1.0, 300)
loss = -np.log(p)

ax.plot(p, loss, color='darkred', lw=3)
ax.fill_between(p, loss, alpha=0.1, color='red')

# 標記幾個關鍵點
key_points = [(0.9, 'steelblue'), (0.5, 'orange'), (0.1, 'red'), (0.01, 'darkred')]
for prob, color in key_points:
    l = -np.log(prob)
    ax.plot(prob, l, 'o', color=color, markersize=9, zorder=5)
    ax.annotate(f'機率={prob:.0%}\nloss={l:.2f}',
                xy=(prob, l), xytext=(prob + 0.08, l + 0.3),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.set_xlim(0, 1.05)
ax.set_ylim(-0.1, 5.5)
ax.set_xlabel("正確類別的預測機率")
ax.set_ylabel("Loss = -log(機率)")
ax.set_title("-log(x) 曲線\n猜越錯，懲罰越重", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

ax.annotate('機率=100%\nloss=0\n（完美預測）',
            xy=(1.0, 0), xytext=(0.7, 0.8),
            fontsize=9, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# ─────────────────────────────────────────
# 圖 3：完整流程
# ─────────────────────────────────────────
ax = axes[2]
ax.axis('off')

flow = [
    ("模型輸出（logits）", "[2.0,  1.0,  0.5]", 'steelblue', 0.92),
    ("Step 1：取 e^x", "[7.39, 2.72, 1.65]", 'darkorange', 0.74),
    ("Step 2：除以總和 11.76", "[63%,  23%,  14%]", 'green', 0.56),
    ("真實答案 = 類別 0", "→ 取類別 0 的機率：63%", 'purple', 0.38),
    ("Step 3：CrossEntropy", "loss = -log(0.63) = 0.46", 'darkred', 0.20),
]

ax.text(0.5, 0.99, "完整計算流程", ha='center', va='top',
        fontsize=13, fontweight='bold', transform=ax.transAxes)

for label, value, color, y in flow:
    ax.text(0.05, y + 0.07, label, ha='left', va='center',
            fontsize=10, color='gray', transform=ax.transAxes)
    ax.text(0.05, y, value, ha='left', va='center',
            fontsize=12, color=color, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=1.5))
    if y > 0.20:
        ax.annotate('', xy=(0.15, y - 0.05), xytext=(0.15, y - 0.01),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.text(0.5, 0.03,
        "loss 越小 → 預測越準\n訓練目標 = 讓 loss 不斷下降",
        ha='center', va='bottom', fontsize=10,
        color='darkred', fontstyle='italic',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig("/Users/songlin.chen/Documents/MLops/phase2/explain_softmax_cross.png",
            dpi=150, bbox_inches='tight')
print("圖片儲存完成")
