# ============================================================
# 03_evaluate_and_save.py
# 主題：評估指標、儲存與載入模型
#
# 訓練完不代表完成，還需要：
#   1. 用正確的指標評估模型好不好
#   2. 儲存模型，之後可以直接載入不用重新訓練
#   3. 驗證載入的模型和原本一樣
#
# 評估指標：
#   Accuracy：準確率（適合類別平衡）
#   Precision / Recall / F1：適合類別不平衡
#   Confusion Matrix：看清楚哪些類別容易搞混
# ============================================================

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# 1. 載入 fine-tuned 模型
# ─────────────────────────────────────────
print("=== 1. 載入 Fine-tuned 模型 ===")

MODEL_PATH = "./phase3/finetuned_model"   # 01 儲存的路徑

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print(f"成功載入模型: {MODEL_PATH}")
except Exception:
    # 如果還沒有 fine-tuned 模型，用原始預訓練模型示範
    print("找不到 fine-tuned 模型，使用 distilbert-base-uncased 示範")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ─────────────────────────────────────────
# 2. 準備測試資料
# ─────────────────────────────────────────
print("\n=== 2. 準備測試資料 ===")

dataset   = load_dataset("imdb")
test_data = dataset["test"].shuffle(seed=42).select(range(200))

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_test = test_data.map(tokenize_fn, batched=True)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_loader = DataLoader(tokenized_test, batch_size=32)

print(f"測試筆數: {len(test_data)}")

# ─────────────────────────────────────────
# 3. 取得預測結果
# ─────────────────────────────────────────
print("\n=== 3. 推論 ===")

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = outputs.logits.argmax(dim=-1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ─────────────────────────────────────────
# 4. 各種評估指標
# ─────────────────────────────────────────
print("\n=== 4. 評估指標 ===")

# Accuracy
accuracy = (all_preds == all_labels).mean()
print(f"Accuracy: {accuracy:.4f}")

# Precision / Recall / F1（用 sklearn）
# 輸出每個類別的詳細指標
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=["負面(0)", "正面(1)"]
))

# Confusion Matrix
# 格式：
#   [[TN, FP],
#    [FN, TP]]
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(f"  預測負面  預測正面")
print(f"實際負面  {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"實際正面  {cm[1][0]:5d}  {cm[1][1]:5d}")

# ─────────────────────────────────────────
# 5. 儲存與載入（驗證）
# ─────────────────────────────────────────
print("\n=== 5. 儲存與重新載入驗證 ===")

SAVE_PATH = "./phase3/model_evaluated"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"模型儲存至: {SAVE_PATH}")

# 重新載入
reloaded_model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)
reloaded_model.to(device)
reloaded_model.eval()
print("重新載入成功")

# 驗證：對同一筆輸入，兩個模型輸出應完全相同
sample_text = "This movie was absolutely fantastic!"
inputs = tokenizer(sample_text, return_tensors="pt").to(device)

with torch.no_grad():
    out1 = model(**inputs).logits
    out2 = reloaded_model(**inputs).logits

print(f"\n原始模型 logits:  {out1.cpu().numpy()}")
print(f"重載模型 logits:  {out2.cpu().numpy()}")
print(f"結果一致: {torch.allclose(out1, out2)}")

# ─────────────────────────────────────────
# 6. 用 pipeline 快速推論
# ─────────────────────────────────────────
print("\n=== 6. Pipeline 快速推論 ===")

clf_pipeline = pipeline(
    "text-classification",
    model=reloaded_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

test_sentences = [
    "The acting was superb and the story was deeply moving.",
    "I wasted two hours of my life on this garbage.",
    "It was fine, nothing extraordinary.",
]

for sentence in test_sentences:
    result = clf_pipeline(sentence)[0]
    label  = "正面" if result["label"] == "LABEL_1" else "負面"
    print(f"  {sentence[:50]}")
    print(f"  → {label} (confidence: {result['score']:.2%})\n")

print("""
=== 重點整理 ===

評估指標選擇：
  Accuracy   → 類別平衡時用（正/負各半）
  F1 score   → 類別不平衡時更可靠（例如詐騙偵測）
  Confusion Matrix → 看出哪種錯誤最多（FP vs FN）

儲存/載入：
  model.save_pretrained(path)     → 儲存權重 + config
  tokenizer.save_pretrained(path) → 儲存 tokenizer 設定
  AutoModel.from_pretrained(path) → 本地路徑或 HuggingFace Hub 都可以

下一步：把模型包成推論服務（inference pipeline）
""")
