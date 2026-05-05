# ============================================================
# 04_inference_pipeline.py
# 主題：把 fine-tuned 模型包成推論服務
#
# 訓練完模型後，實際使用的方式：
#   - 單筆推論（使用者送一句話）
#   - 批次推論（一次處理多筆）
#   - 信心分數閾值過濾
#   - 模擬 REST API 接收 JSON 輸入
#
# 這是部署到 AWS 前的最後一步：
#   確認模型可以正確接收輸入、輸出結構化結果
# ============================================================

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ─────────────────────────────────────────
# 1. 載入模型
# ─────────────────────────────────────────
print("=== 1. 載入模型 ===")

MODEL_PATH = "./phase3/finetuned_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print(f"載入 fine-tuned 模型: {MODEL_PATH}")
except Exception:
    print("找不到 fine-tuned 模型，使用預訓練模型示範")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"裝置: {device}")

# ─────────────────────────────────────────
# 2. 單筆推論函式
# ─────────────────────────────────────────

LABELS = {0: "負面", 1: "正面"}

def predict_one(text: str) -> dict:
    """
    輸入一段文字，回傳預測結果與信心分數。

    Args:
        text: 要分類的文字

    Returns:
        {
            "text": 原始文字,
            "label": "正面" 或 "負面",
            "label_id": 0 或 1,
            "confidence": 0.0 ~ 1.0,
            "scores": {"負面": float, "正面": float}
        }
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits   # shape: (1, 2)

    # softmax 轉成機率
    probs    = F.softmax(logits, dim=-1).squeeze().cpu()
    label_id = probs.argmax().item()

    return {
        "text":       text,
        "label":      LABELS[label_id],
        "label_id":   label_id,
        "confidence": round(probs[label_id].item(), 4),
        "scores": {
            "負面": round(probs[0].item(), 4),
            "正面": round(probs[1].item(), 4),
        },
    }

# ─────────────────────────────────────────
# 3. 批次推論函式
# ─────────────────────────────────────────

def predict_batch(texts: list[str], batch_size: int = 16) -> list[dict]:
    """
    批次推論，比逐筆快很多。

    Args:
        texts:      文字列表
        batch_size: 每批處理幾筆

    Returns:
        list of predict_one 格式的 dict
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits   # shape: (batch, 2)

        probs    = F.softmax(logits, dim=-1).cpu()
        label_ids = probs.argmax(dim=-1).tolist()

        for text, label_id, prob in zip(batch_texts, label_ids, probs):
            results.append({
                "text":       text,
                "label":      LABELS[label_id],
                "label_id":   label_id,
                "confidence": round(prob[label_id].item(), 4),
                "scores": {
                    "負面": round(prob[0].item(), 4),
                    "正面": round(prob[1].item(), 4),
                },
            })

    return results

# ─────────────────────────────────────────
# 4. 單筆推論示範
# ─────────────────────────────────────────
print("\n=== 2. 單筆推論 ===")

samples = [
    "This movie was absolutely amazing, I loved every minute!",
    "Terrible film, complete waste of time and money.",
    "It was okay, some good parts but also some bad.",
]

for text in samples:
    result = predict_one(text)
    print(f"  文字: {result['text'][:60]}")
    print(f"  結果: {result['label']}  信心度: {result['confidence']:.2%}")
    print(f"  分數: 負面={result['scores']['負面']}  正面={result['scores']['正面']}")
    print()

# ─────────────────────────────────────────
# 5. 批次推論示範
# ─────────────────────────────────────────
print("=== 3. 批次推論 ===")

batch_texts = [
    "Incredible performances by all actors.",
    "The plot made no sense whatsoever.",
    "A decent film with a few memorable scenes.",
    "One of the worst movies I have ever seen.",
    "Visually stunning and emotionally powerful.",
]

batch_results = predict_batch(batch_texts)
for r in batch_results:
    print(f"  [{r['label']}] ({r['confidence']:.2%}) {r['text'][:55]}")

# ─────────────────────────────────────────
# 6. 信心分數閾值過濾
# ─────────────────────────────────────────
print("\n=== 4. 信心分數閾值過濾 ===")
# 低於閾值的預測標記為「不確定」，避免錯誤分類影響下游系統

THRESHOLD = 0.80

def predict_with_threshold(text: str, threshold: float = THRESHOLD) -> dict:
    result = predict_one(text)
    if result["confidence"] < threshold:
        result["label"]    = "不確定"
        result["label_id"] = -1
    return result

uncertain_samples = [
    "It had its moments, but overall just average.",
    "Not great, not terrible, just... there.",
    "I expected more, but it wasn't completely bad either.",
]

print(f"閾值: {THRESHOLD}")
for text in uncertain_samples:
    r = predict_with_threshold(text)
    print(f"  [{r['label']:4s}] ({r['confidence']:.2%}) {text[:55]}")

# ─────────────────────────────────────────
# 7. 模擬 REST API 請求處理
# ─────────────────────────────────────────
print("\n=== 5. 模擬 REST API 請求 ===")

# 模擬從 HTTP 收到的 JSON body
api_request_json = '''
{
    "texts": [
        "Best film of the decade!",
        "I fell asleep halfway through.",
        "Surprisingly enjoyable for a low budget film."
    ],
    "threshold": 0.75
}
'''

def handle_api_request(json_body: str) -> str:
    """模擬 Lambda / FastAPI handler"""
    body = json.loads(json_body)
    texts     = body.get("texts", [])
    threshold = body.get("threshold", 0.80)

    raw_results = predict_batch(texts)

    # 套用閾值
    response_items = []
    for r in raw_results:
        if r["confidence"] < threshold:
            r["label"]    = "不確定"
            r["label_id"] = -1
        response_items.append({
            "text":       r["text"],
            "label":      r["label"],
            "confidence": r["confidence"],
        })

    response = {
        "status": "ok",
        "count":   len(response_items),
        "results": response_items,
    }
    return json.dumps(response, ensure_ascii=False, indent=2)

response_json = handle_api_request(api_request_json)
print("API 回應:")
print(response_json)

print("""
=== 重點整理 ===

推論流程：
  文字 → tokenizer → input_ids / attention_mask
       → model → logits → softmax → 機率 → argmax → 類別

批次推論 vs 逐筆推論：
  批次：GPU 並行處理，速度快很多（尤其有 GPU 時）
  逐筆：適合即時單筆查詢

信心分數閾值：
  避免模型在不確定時給出強硬答案
  實務上常設 0.7~0.9，視業務風險而定

部署下一步（Phase 4 - AWS）：
  把 handle_api_request 包進 AWS Lambda
  → API Gateway 接收 HTTP → Lambda 推論 → 回傳 JSON
""")
