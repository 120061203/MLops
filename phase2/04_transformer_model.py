# ============================================================
# 04_transformer_model.py
# 主題：完整 Transformer Encoder 架構
#
# 把前面學的組合起來：
#   Multi-Head Attention + Feed Forward + Layer Norm + Residual
#
# Encoder 架構（BERT 用的）：
#   輸入文字 → Embedding → 多層 EncoderLayer → 輸出向量表示
#
# Decoder 架構（GPT 用的）多了一個 Masked Attention，
# 這裡先專注在 Encoder。
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────
# 1. 基本元件（從上一個檔案複製過來）
# ─────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, seq, d_model = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        batch, seq, d_model = x.shape
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        return self.W_o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ─────────────────────────────────────────
# 2. Feed Forward Network（FFN）
# ─────────────────────────────────────────
# 每個 Transformer 層都有一個 FFN：
#   兩層線性層 + ReLU，對每個 token 獨立處理
#   作用：在 attention 整合資訊後，進一步轉換表示

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # d_ff 通常是 d_model 的 4 倍（慣例）
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# ─────────────────────────────────────────
# 3. Encoder Layer（一層的完整結構）
# ─────────────────────────────────────────
# 每層結構：
#   x → MultiHeadAttention → 殘差 + LayerNorm → FFN → 殘差 + LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff        = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1：Multi-Head Attention + 殘差連接 + LayerNorm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))   # 殘差連接

        # Sub-layer 2：Feed Forward + 殘差連接 + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))      # 殘差連接

        return x

# ─────────────────────────────────────────
# 4. 完整 Transformer Encoder
# ─────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=512, dropout=0.1):
        super().__init__()
        # Embedding：把 token id 轉成向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # 多層 EncoderLayer 疊起來
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len)，裡面是 token id

        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 依序通過每一層
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)   # shape: (batch, seq_len, d_model)

# ─────────────────────────────────────────
# 5. 實際跑看看
# ─────────────────────────────────────────

# 模型設定（小型版本，方便示範）
vocab_size = 1000   # 詞彙量
d_model    = 64     # 每個 token 的向量維度
num_heads  = 4      # Attention head 數量
d_ff       = 256    # FFN 中間層維度（通常 4 * d_model）
num_layers = 2      # 疊幾層 Encoder

model = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
)

print("=== Transformer Encoder ===")
print(model)

# 計算參數量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n總參數量: {total_params:,}")

# 模擬輸入：2 個句子，每句 10 個 token
batch_size = 2
seq_len = 10
token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
print(f"\n輸入 token_ids shape: {token_ids.shape}")

output = model(token_ids)
print(f"輸出 shape: {output.shape}")
# (batch=2, seq=10, d_model=64)：每個 token 都得到一個 64 維的向量表示

# ─────────────────────────────────────────
# 6. 加上分類頭：做文字分類任務
# ─────────────────────────────────────────
# Encoder 輸出的是每個 token 的向量。
# 分類任務通常取第一個 token（[CLS]）的向量來做分類。

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers
        )
        # 分類頭：取 [CLS] token 的向量 → 類別數
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        enc_out = self.encoder(x)       # (batch, seq, d_model)
        cls_token = enc_out[:, 0, :]    # 取第一個 token，shape: (batch, d_model)
        return self.classifier(cls_token)  # (batch, num_classes)

classifier = TextClassifier(
    vocab_size=1000,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=2,
    num_classes=3,   # 例如：正面、負面、中性
)

logits = classifier(token_ids)
print(f"\n文字分類輸出 shape: {logits.shape}")  # (2, 3)
print(f"預測類別: {logits.argmax(dim=1)}")

# ─────────────────────────────────────────
# 7. BERT vs GPT 架構差異
# ─────────────────────────────────────────
print("""
=== BERT vs GPT 架構差異 ===

BERT（Encoder only）：
  - 每個 token 可以同時看到左右兩邊的詞
  - 適合：文字分類、問答、命名實體辨識
  - 訓練方式：遮住某些詞，預測被遮住的詞（MLM）

GPT（Decoder only）：
  - 每個 token 只能看到左邊（之前）的詞
  - 適合：文字生成、對話
  - 訓練方式：預測下一個詞（CLM）

Decoder 比 Encoder 多了 Causal Mask：
  防止 token 看到未來的詞
  [[1, 0, 0],
   [1, 1, 0],   ← token 2 只能看到 token 1、2
   [1, 1, 1]]
""")
