# ============================================================
# 05_huggingface_intro.py
# 主題：HuggingFace transformers 入門
#
# 前面幾個檔案是從零實作 Transformer，理解底層原理。
# 實際工作中，我們不會自己實作，而是用 HuggingFace 提供的
# 現成 pre-trained model。
#
# HuggingFace 提供：
#   - 數萬個預訓練模型（BERT、GPT、LLaMA 等）
#   - 統一的 API：AutoTokenizer、AutoModel
#   - Pipeline：幾行程式碼完成常見 NLP 任務
# ============================================================

# 安裝（如果還沒裝）：
# pip install transformers

from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# ─────────────────────────────────────────
# 1. Tokenizer：把文字轉成 token id
# ─────────────────────────────────────────
# 模型無法直接處理文字，需要先轉成數字（token id）
# 每個模型都有自己的 tokenizer，要配對使用

print("=== 1. Tokenizer ===")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 第一次執行會從 HuggingFace Hub 下載模型，之後會快取在本地

text = "Hello, I am learning machine learning!"

# 基本 tokenize：把文字切成 token
tokens = tokenizer.tokenize(text)
print(f"原始文字: {text}")
print(f"切成 tokens: {tokens}")
# BERT 用 WordPiece：不常見的詞會被切成子詞（subword）
# 例如 "learning" → ["learning"]，"unrecognizable" → ["un", "##rec", "##og", ...]

# 轉成 token id
token_ids = tokenizer.encode(text)
print(f"\nToken IDs: {token_ids}")
# 注意：BERT 會自動加上 [CLS]（101）和 [SEP]（102）

# 解碼回文字
decoded = tokenizer.decode(token_ids)
print(f"解碼回來: {decoded}")

# ─────────────────────────────────────────
# 2. 批次處理（實際訓練時的用法）
# ─────────────────────────────────────────
print("\n=== 2. 批次 Tokenize ===")

texts = [
    "I love machine learning.",
    "Deep learning is amazing!",
    "Transformers changed NLP.",
]

# padding=True：短句子補到同樣長度
# truncation=True：超過 max_length 的句子截斷
# return_tensors="pt"：直接回傳 PyTorch tensor
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt",
)

print(f"input_ids shape:      {encoded['input_ids'].shape}")
print(f"attention_mask shape: {encoded['attention_mask'].shape}")
print(f"\ninput_ids:\n{encoded['input_ids']}")
print(f"\nattention_mask:\n{encoded['attention_mask']}")
# attention_mask：1 表示真實 token，0 表示 padding（模型要忽略這些位置）

# ─────────────────────────────────────────
# 3. AutoModel：載入預訓練模型
# ─────────────────────────────────────────
print("\n=== 3. AutoModel 推論 ===")

model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
# BERT-base 有 1.1 億個參數

with torch.no_grad():
    outputs = model(**encoded)

# last_hidden_state：每個 token 的向量表示
print(f"\nlast_hidden_state shape: {outputs.last_hidden_state.shape}")
# (batch=3, seq_len, hidden_size=768)
# 每個 token 都得到一個 768 維的向量

# CLS token（第一個 token）代表整個句子
cls_embeddings = outputs.last_hidden_state[:, 0, :]
print(f"CLS embedding shape: {cls_embeddings.shape}")   # (3, 768)

# ─────────────────────────────────────────
# 4. Pipeline：最快速的使用方式
# ─────────────────────────────────────────
# Pipeline 幫你把 tokenize → model → 後處理 全部包起來

print("\n=== 4. Pipeline API ===")

# 情感分析
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

texts_to_analyze = [
    "I really love this product!",
    "This is terrible, I hate it.",
    "It's okay, nothing special.",
]

results = sentiment_pipeline(texts_to_analyze)
for text, result in zip(texts_to_analyze, results):
    print(f"  '{text}'")
    print(f"  → {result['label']} (confidence: {result['score']:.2%})\n")

# ─────────────────────────────────────────
# 5. 文字生成（GPT 系列）
# ─────────────────────────────────────────
print("=== 5. 文字生成 ===")

generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=30,
    truncation=True,
)

prompt = "Machine learning is"
generated = generator(prompt, num_return_sequences=1)
print(f"Prompt: '{prompt}'")
print(f"Generated: {generated[0]['generated_text']}")

# ─────────────────────────────────────────
# 6. 用 BERT 做 sentence embedding（語意搜尋的基礎）
# ─────────────────────────────────────────
print("\n=== 6. Sentence Embedding ===")

def mean_pooling(model_output, attention_mask):
    """對所有 token 的向量做平均（忽略 padding）"""
    token_embeddings = model_output.last_hidden_state
    # 只對真實 token 做平均（attention_mask=1 的位置）
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

sentences = [
    "A dog is running in the park.",
    "A puppy is playing outside.",
    "Machine learning is fascinating.",
]

encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoded)

embeddings = mean_pooling(outputs, encoded['attention_mask'])
print(f"Sentence embeddings shape: {embeddings.shape}")  # (3, 768)

# 計算句子之間的相似度（cosine similarity）
def cosine_similarity(a, b):
    return (a @ b.T) / (a.norm(dim=1, keepdim=True) * b.norm(dim=1, keepdim=True).T)

sim = cosine_similarity(embeddings, embeddings)
print("\n句子相似度矩陣:")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        print(f"  [{i}] vs [{j}]: {sim[i, j].item():.4f}")
# 前兩句（都在說狗）相似度應該高於第三句（機器學習）

# ─────────────────────────────────────────
# 7. 重點整理
# ─────────────────────────────────────────
print("""
=== 重點整理 ===

HuggingFace 核心元件：
  AutoTokenizer  → 文字轉 token id，每個模型有自己的 tokenizer
  AutoModel      → 載入預訓練模型，輸出每個 token 的向量
  pipeline       → 高階 API，直接完成常見任務

Tokenizer 輸出：
  input_ids      → token id 序列
  attention_mask → 1=真實 token，0=padding

Model 輸出：
  last_hidden_state → (batch, seq_len, hidden_size)，每個 token 的向量

下一步（Phase 3）：
  Fine-tuning = 拿這個預訓練模型，針對你的任務再訓練幾輪
  不用從頭訓練，只需要少量資料和時間
""")
