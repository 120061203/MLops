# ============================================================
# 02_upload_model.py
# 主題：把 fine-tuned 模型打包上傳到 S3
#
# SageMaker 部署模型的格式要求：
#   模型必須打包成 model.tar.gz
#   裡面包含：模型權重、tokenizer、推論程式碼（inference.py）
#
# 目錄結構：
#   model.tar.gz
#   ├── pytorch_model.bin   ← 模型權重
#   ├── config.json         ← 模型設定
#   ├── tokenizer files     ← tokenizer
#   └── code/
#       └── inference.py    ← SageMaker 呼叫的推論入口
# ============================================================

import os
import json
import tarfile
import shutil
import boto3
import sagemaker

# ─────────────────────────────────────────
# 0. 載入設定
# ─────────────────────────────────────────
with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region    = config["region"]
bucket    = config["bucket"]
role      = config["role"]
s3_prefix = config["s3_prefix"]

print("=== 設定 ===")
print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"S3 prefix: {s3_prefix}")

# ─────────────────────────────────────────
# 1. 建立 inference.py（SageMaker 推論入口）
# ─────────────────────────────────────────
print("\n=== 1. 建立 inference.py ===")

# SageMaker 會呼叫這個檔案的 model_fn 和 predict_fn
# model_fn：載入模型
# predict_fn：接收輸入、回傳預測結果

inference_code = '''import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "負面", 1: "正面"}

def model_fn(model_dir):
    """SageMaker 啟動時呼叫，載入模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(input_data, model_dict):
    """SageMaker 每次收到請求時呼叫"""
    model     = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    texts = input_data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs    = F.softmax(logits, dim=-1)
    label_ids = probs.argmax(dim=-1).tolist()

    results = []
    for i, label_id in enumerate(label_ids):
        results.append({
            "text":       texts[i],
            "label":      LABELS[label_id],
            "confidence": round(probs[i][label_id].item(), 4),
        })
    return results

def input_fn(request_body, content_type="application/json"):
    """解析 HTTP 請求 body"""
    return json.loads(request_body)

def output_fn(prediction, accept="application/json"):
    """把結果序列化成 HTTP 回應"""
    return json.dumps(prediction, ensure_ascii=False), accept
'''

# 建立 code/ 目錄
os.makedirs("./phase4/model_package/code", exist_ok=True)
with open("./phase4/model_package/code/inference.py", "w") as f:
    f.write(inference_code)

print("inference.py 建立完成")
print("  model_fn    → 載入模型（SageMaker 啟動時執行一次）")
print("  predict_fn  → 推論（每次 API 呼叫時執行）")
print("  input_fn    → 解析 HTTP 請求")
print("  output_fn   → 序列化回應")

# ─────────────────────────────────────────
# 2. 複製模型檔案
# ─────────────────────────────────────────
print("\n=== 2. 複製模型檔案 ===")

MODEL_SRC = "./phase3/finetuned_model"
MODEL_DST = "./phase4/model_package"

if not os.path.exists(MODEL_SRC):
    raise FileNotFoundError(f"找不到 fine-tuned 模型：{MODEL_SRC}\n請先執行 phase3/01_finetune_text_classification.py")

# 複製模型檔案（保留 code/ 目錄）
for fname in os.listdir(MODEL_SRC):
    src = os.path.join(MODEL_SRC, fname)
    dst = os.path.join(MODEL_DST, fname)
    shutil.copy2(src, dst)
    print(f"  複製: {fname}")

# ─────────────────────────────────────────
# 3. 打包成 model.tar.gz
# ─────────────────────────────────────────
print("\n=== 3. 打包成 model.tar.gz ===")

TAR_PATH = "./phase4/model.tar.gz"

with tarfile.open(TAR_PATH, "w:gz") as tar:
    for fname in os.listdir(MODEL_DST):
        fpath = os.path.join(MODEL_DST, fname)
        if os.path.isfile(fpath):
            tar.add(fpath, arcname=fname)
            print(f"  加入: {fname}")
        elif os.path.isdir(fpath):
            tar.add(fpath, arcname=fname)
            print(f"  加入: {fname}/（目錄）")

size_mb = os.path.getsize(TAR_PATH) / 1024 / 1024
print(f"\nmodel.tar.gz 大小: {size_mb:.1f} MB")

# ─────────────────────────────────────────
# 4. 上傳到 S3
# ─────────────────────────────────────────
print("\n=== 4. 上傳到 S3 ===")

s3_key = f"{s3_prefix}/model/model.tar.gz"
s3_uri = f"s3://{bucket}/{s3_key}"

s3 = boto3.client("s3", region_name=region)

print(f"上傳中：{TAR_PATH} → {s3_uri}")

s3.upload_file(
    TAR_PATH,
    bucket,
    s3_key,
    Callback=lambda bytes_transferred: print(
        f"  已上傳 {bytes_transferred / 1024 / 1024:.1f} MB", end="\r"
    ),
)

print(f"\n上傳完成！")
print(f"S3 URI: {s3_uri}")

# ─────────────────────────────────────────
# 5. 更新設定檔
# ─────────────────────────────────────────
config["model_s3_uri"] = s3_uri

with open("./phase4/aws_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n設定已更新：aws_config.json")
print(f"  model_s3_uri: {s3_uri}")

print("""
=== 重點整理 ===

SageMaker 模型包結構：
  model.tar.gz
  ├── 模型權重 + config（來自 phase3 fine-tuned 模型）
  └── code/inference.py（SageMaker 的推論入口）

inference.py 四個函式：
  model_fn   → 啟動時載入模型（只跑一次）
  input_fn   → 解析每次請求的 body
  predict_fn → 實際做推論
  output_fn  → 把結果轉成 JSON 回傳

下一步（03）：
  用這個 S3 URI 建立 SageMaker Model → Endpoint Config → Endpoint
""")
