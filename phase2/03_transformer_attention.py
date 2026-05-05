# ============================================================
# 03_transformer_attention.py
# 主題：Self-Attention 機制
#
# Attention 是 Transformer 的核心，解決了一個問題：
#   「處理一個詞時，要怎麼知道句子中其他詞的重要性？」
#
# 例如：「銀行旁邊有條河」vs「我去銀行存錢」
#   「銀行」這個詞，意思取決於周圍的詞。
#   Attention 讓模型動態決定要「關注」哪些詞。
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────
# 1. Q、K、V 是什麼？
# ─────────────────────────────────────────
# 每個 token（詞）會被轉成三個向量：
#   Q (Query)：「我想找什麼資訊？」
#   K (Key)：  「我有什麼資訊？」
#   V (Value)：「我實際提供的內容」
#
# 計算流程：
#   1. Q 和每個 K 做內積 → 得到「相關性分數」
#   2. 用 Softmax 把分數變成機率（Attention Weight）
#   3. 用這個機率對所有 V 做加權平均 → 最終輸出

# ─────────────────────────────────────────
# 2. 手動實作 Scaled Dot-Product Attention
# ─────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]

    # Step 1: Q @ K^T → 相關性分數，shape: (batch, seq_len, seq_len)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    # 除以 sqrt(d_k) 是為了避免內積值太大導致 Softmax 梯度消失

    # Step 2: 若有 mask，把被遮住的位置設為 -inf（Softmax 後變 0）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax → Attention Weight（每列加總為 1）
    attn_weights = F.softmax(scores, dim=-1)

    # Step 4: 加權平均 V → 輸出
    output = attn_weights @ V

    return output, attn_weights

# 示範：3 個 token，每個向量維度為 4
batch_size = 1
seq_len = 3
d_k = 4

torch.manual_seed(42)
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("=== Scaled Dot-Product Attention ===")
print(f"Q shape: {Q.shape}")
print(f"Attention Weights:\n{attn_weights[0].detach()}")
print("每列加總:", attn_weights[0].sum(dim=-1).detach())   # 每列都是 1
print(f"Output shape: {output.shape}")

# ─────────────────────────────────────────
# 3. Multi-Head Attention
# ─────────────────────────────────────────
# 單一 Attention 只能關注一種關係。
# Multi-Head 讓模型同時從多個角度理解詞之間的關係：
#   Head 1：可能關注語法關係（主詞-動詞）
#   Head 2：可能關注語義關係（同義詞）
#   Head 3：可能關注位置關係（前後鄰近）

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每個 head 的維度

        # 把輸入線性投影成 Q、K、V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # 把所有 head 的輸出合併後再投影
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """把 (batch, seq, d_model) 切成 (batch, heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq, d_k)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape

        # 線性投影
        Q = self.split_heads(self.W_q(x))   # (batch, heads, seq, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # 每個 head 各自做 attention
        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 合併所有 head：(batch, heads, seq, d_k) → (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        # 最終線性投影
        out = self.W_o(out)
        return out, attn_weights

d_model = 32
num_heads = 4
seq_len = 5
batch_size = 2

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
x = torch.randn(batch_size, seq_len, d_model)
out, attn = mha(x)

print("\n=== Multi-Head Attention ===")
print(f"輸入 shape:  {x.shape}")
print(f"輸出 shape:  {out.shape}")   # 和輸入一樣
print(f"Attention shape: {attn.shape}")  # (batch, heads, seq, seq)
print(f"Head 1 的 Attention Weights:\n{attn[0, 0].detach().round(decimals=2)}")

# ─────────────────────────────────────────
# 4. Positional Encoding（位置編碼）
# ─────────────────────────────────────────
# Attention 本身不知道詞的順序（「我打你」和「你打我」會得到一樣的結果）
# 需要額外加入位置資訊

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶數維度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇數維度用 cos
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # 把位置編碼加到輸入上
        return x + self.pe[:, :x.size(1)]

pe = PositionalEncoding(d_model=32)
x = torch.randn(2, 5, 32)
x_with_pos = pe(x)
print(f"\n=== Positional Encoding ===")
print(f"加入位置編碼前: {x.shape}")
print(f"加入位置編碼後: {x_with_pos.shape}")  # shape 不變，但每個位置的值不同

# ─────────────────────────────────────────
# 5. 重點整理
# ─────────────────────────────────────────
print("""
=== 重點整理 ===

Attention 的直覺：
  每個詞問「我應該關注哪些詞？」
  Q（我要找什麼）和 K（別人有什麼）做比對
  根據相關性對 V（別人的內容）做加權平均

Q / K / V：
  都是輸入 x 經過不同線性層得到的
  同一個輸入可以同時作為 Q、K、V（Self-Attention）

Multi-Head：
  多個 head 同時從不同角度做 attention
  最後合併，讓模型捕捉多種語言關係

Positional Encoding：
  彌補 attention 沒有順序概念的缺陷
  用 sin/cos 函式把位置資訊編進向量中
""")
