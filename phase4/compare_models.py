# ============================================================
# compare_models.py
# 主題：比較原始 DistilBERT 和 fine-tuned 模型的預測差異
# ============================================================

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────
# 載入兩個模型
# ─────────────────────────────────────────
print("載入模型中...")

# 原始預訓練模型（沒有 fine-tune）
tokenizer_orig = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_orig     = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model_orig.eval()

# Fine-tuned 模型
tokenizer_ft = AutoTokenizer.from_pretrained("./phase3/finetuned_model")
model_ft     = AutoModelForSequenceClassification.from_pretrained("./phase3/finetuned_model")
model_ft.eval()

print("載入完成\n")

# ─────────────────────────────────────────
# 預測函式
# ─────────────────────────────────────────
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = F.softmax(logits, dim=-1).squeeze()
    label_id = probs.argmax().item()
    label    = "正面" if label_id == 1 else "負面"
    return label, probs[label_id].item()

# ─────────────────────────────────────────
# 測試句子
# ─────────────────────────────────────────
test_sentences = [
    "This movie was absolutely amazing, I loved every minute!",
    "Terrible film, complete waste of time and money.",
    "It was okay, nothing special.",
    "The acting was superb and the story deeply moving.",
    "I fell asleep halfway through, so boring.",
    "Best movie I have seen in years!",
    "The plot made no sense at all.",
    "Not great, not terrible, just mediocre.",
]

print(f"{'句子':<50} {'原始模型':^15} {'Fine-tuned':^15} {'是否一致'}")
print("─" * 95)

agree = 0
for text in test_sentences:
    label_orig, conf_orig = predict(model_orig, tokenizer_orig, text)
    label_ft,   conf_ft   = predict(model_ft,   tokenizer_ft,   text)
    match = "✓" if label_orig == label_ft else "✗ 不同"
    if label_orig == label_ft:
        agree += 1
    orig_str = f"{label_orig}({conf_orig:.0%})"
    ft_str   = f"{label_ft}({conf_ft:.0%})"
    print(f"{text[:48]:<50} {orig_str:^15} {ft_str:^15} {match}")

print("─" * 95)
print(f"預測一致: {agree}/{len(test_sentences)} 句")
print(f"\n重點：")
print(f"  原始模型的分類頭是隨機初始化的，信心度接近 50% 代表在亂猜")
print(f"  Fine-tuned 模型學過資料，信心度應該更高、方向更穩定")
