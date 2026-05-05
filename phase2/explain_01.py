import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# 設定中文字型
font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Phase 2 - 01 核心概念", fontsize=16, fontweight='bold')

# ─────────────────────────────────────────
# 圖 1：激活函式（ReLU vs Sigmoid）
# ─────────────────────────────────────────
ax1 = axes[0]
x = torch.linspace(-4, 4, 200)
relu = F.relu(x).numpy()
sigmoid = torch.sigmoid(x).numpy()
x_np = x.numpy()

ax1.plot(x_np, relu,    color='steelblue',  linewidth=2.5, label='ReLU')
ax1.plot(x_np, sigmoid, color='darkorange', linewidth=2.5, label='Sigmoid')
ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax1.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax1.set_title("激活函式", fontsize=13, fontweight='bold')
ax1.set_xlabel("輸入值")
ax1.set_ylabel("輸出值")
ax1.legend(fontsize=11)
ax1.set_ylim(-0.3, 4.2)
ax1.annotate("負數 → 0\n（ReLU）", xy=(-2, 0.05), fontsize=9, color='steelblue')
ax1.annotate("壓縮到 0~1\n（Sigmoid）", xy=(1.2, 0.35), fontsize=9, color='darkorange')
ax1.grid(True, alpha=0.3)

# ─────────────────────────────────────────
# 圖 2：Softmax 把分數變成機率
# ─────────────────────────────────────────
ax2 = axes[1]

categories = ['類別 0\n(貓)', '類別 1\n(狗)', '類別 2\n(鳥)']
logits  = np.array([2.0, 1.0, 0.5])
softmax = np.exp(logits) / np.exp(logits).sum()

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, logits,  width, label='Logits（原始分數）', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, softmax, width, label='Softmax（機率）',   color='darkorange', alpha=0.8)

# 加上數值標籤
for bar, val in zip(bars1, logits):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10, color='steelblue')
for bar, val in zip(bars2, softmax):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.0%}', ha='center', va='bottom', fontsize=10, color='darkorange')

ax2.set_title("Softmax：分數 → 機率", fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories)
ax2.set_ylabel("數值")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

total = sum(softmax)
ax2.text(0.5, -0.18, f'機率總和 = {total:.0%}（永遠等於 100%）',
         transform=ax2.transAxes, ha='center', fontsize=10, color='gray')

# ─────────────────────────────────────────
# 圖 3：CrossEntropy Loss
# ─────────────────────────────────────────
ax3 = axes[2]

# 真實答案是類別 0（貓）
scenarios = ['預測 90%\n是貓\n（幾乎猜對）', '預測 50%\n是貓\n（不確定）', '預測 10%\n是貓\n（猜錯）']
probs  = [0.90, 0.50, 0.10]   # 預測類別 0（正確類別）的機率
losses = [-np.log(p) for p in probs]  # CrossEntropy = -log(正確類別的機率)
colors = ['green', 'orange', 'red']

bars = ax3.bar(scenarios, losses, color=colors, alpha=0.8, width=0.5)

for bar, loss, prob in zip(bars, losses, probs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'Loss = {loss:.2f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

ax3.set_title("CrossEntropy Loss\n（真實答案 = 類別0/貓）", fontsize=13, fontweight='bold')
ax3.set_ylabel("Loss 值")
ax3.set_ylim(0, 3.0)
ax3.grid(True, alpha=0.3, axis='y')

patches = [
    mpatches.Patch(color='green',  label='猜對 → Loss 小'),
    mpatches.Patch(color='orange', label='不確定 → Loss 中'),
    mpatches.Patch(color='red',    label='猜錯 → Loss 大'),
]
ax3.legend(handles=patches, fontsize=10)

plt.tight_layout()
plt.savefig("/Users/songlin.chen/Documents/MLops/phase2/explain_01.png", dpi=150, bbox_inches='tight')
print("圖片已儲存至 phase2/explain_01.png")
