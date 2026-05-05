# ============================================================
# 04_invoke_endpoint.py
# 主題：呼叫 SageMaker Endpoint 做推論
#
# Endpoint 啟動後，任何人只要有 AWS 憑證就能呼叫。
# 這個檔案示範：
#   - 單筆 / 批次推論
#   - 信心分數閾值
#   - 計算延遲（latency）
#   - 用完後刪除 Endpoint（避免持續收費）
# ============================================================

import json
import time
import boto3

# ─────────────────────────────────────────
# 0. 載入設定
# ─────────────────────────────────────────
with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region        = config["region"]
endpoint_name = config["endpoint_name"]

print("=== 設定 ===")
print(f"Region:        {region}")
print(f"Endpoint name: {endpoint_name}")

# ─────────────────────────────────────────
# 1. 建立 SageMaker Runtime client
# ─────────────────────────────────────────
# invoke_endpoint 用 sagemaker-runtime（不是 sagemaker）

runtime = boto3.client("sagemaker-runtime", region_name=region)

def invoke(texts: list[str]) -> list[dict]:
    """呼叫 SageMaker Endpoint"""
    payload = json.dumps({"texts": texts}, ensure_ascii=False)

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    # endpoint output_fn 回傳 [json_string, content_type]，需要再解析一次
    if isinstance(result, list) and isinstance(result[0], str):
        result = json.loads(result[0])
    return result

# ─────────────────────────────────────────
# 2. 單筆推論
# ─────────────────────────────────────────
print("\n=== 1. 單筆推論 ===")

samples = [
    "This movie was absolutely amazing, I loved every minute!",
    "Terrible film, complete waste of time and money.",
    "It was okay, some good parts but also some bad.",
]

for text in samples:
    start = time.time()
    result = invoke([text])
    latency = (time.time() - start) * 1000

    r = result[0]
    print(f"  文字: {text[:55]}")
    print(f"  結果: {r['label']}  信心度: {r['confidence']:.2%}  延遲: {latency:.0f}ms")
    print()

# ─────────────────────────────────────────
# 3. 批次推論
# ─────────────────────────────────────────
print("=== 2. 批次推論 ===")

batch_texts = [
    "Incredible performances by all actors.",
    "The plot made no sense whatsoever.",
    "A decent film with a few memorable scenes.",
    "One of the worst movies I have ever seen.",
    "Visually stunning and emotionally powerful.",
]

start   = time.time()
results = invoke(batch_texts)
latency = (time.time() - start) * 1000

print(f"5 筆批次推論，總延遲: {latency:.0f}ms（平均 {latency/len(batch_texts):.0f}ms/筆）")
for r in results:
    print(f"  [{r['label']}] ({r['confidence']:.2%}) {r['text'][:50]}")

# ─────────────────────────────────────────
# 4. 信心分數閾值
# ─────────────────────────────────────────
print("\n=== 3. 信心分數閾值過濾 ===")

THRESHOLD = 0.80

ambiguous = [
    "It had its moments, but overall just average.",
    "Not great, not terrible, just... there.",
    "Best film of the decade!",
]

results = invoke(ambiguous)
print(f"閾值: {THRESHOLD}")
for r in results:
    label = r["label"] if r["confidence"] >= THRESHOLD else "不確定"
    print(f"  [{label}] ({r['confidence']:.2%}) {r['text'][:55]}")

# ─────────────────────────────────────────
# 5. 確認 Endpoint 狀態
# ─────────────────────────────────────────
print("\n=== 4. Endpoint 狀態 ===")

sm = boto3.client("sagemaker", region_name=region)
ep = sm.describe_endpoint(EndpointName=endpoint_name)

print(f"Endpoint 名稱:  {ep['EndpointName']}")
print(f"狀態:           {ep['EndpointStatus']}")
print(f"建立時間:       {ep['CreationTime']}")
variant = ep['ProductionVariants'][0]
print(f"Instance type:  {variant.get('CurrentInstanceType', variant.get('DesiredInstanceType', 'N/A'))}")
print(f"Instance 數量:  {variant.get('CurrentInstanceCount', variant.get('DesiredInstanceCount', 'N/A'))}")

# ─────────────────────────────────────────
# 6. 刪除 Endpoint（避免持續收費）
# ─────────────────────────────────────────
print("\n=== 5. 刪除 Endpoint ===")
print("⚠️  Endpoint 會持續收費，測試完請務必刪除！")

confirm = input("確定要刪除 Endpoint 嗎？(y/n): ").strip().lower()

if confirm == "y":
    sm.delete_endpoint(EndpointName=endpoint_name)
    print(f"Endpoint '{endpoint_name}' 已刪除")
    print("（也可以到 AWS Console → SageMaker → Endpoints 確認）")
else:
    print("跳過刪除")
    print(f"請記得之後手動刪除：{endpoint_name}")
    print("AWS Console → SageMaker → Inference → Endpoints → 選取 → Delete")

print("""
=== Phase 4 總結 ===

完整部署流程：
  phase3 fine-tuned 模型
    → 打包成 model.tar.gz（含 inference.py）
    → 上傳到 S3
    → SageMaker HuggingFaceModel
    → .deploy() → Endpoint（HTTPS API）
    → invoke_endpoint() 呼叫推論

費用提醒：
  ml.m5.large = $0.115/hr
  用完記得刪 Endpoint，否則 24hr = $2.76
  可到 Console → SageMaker → Endpoints 查看

下一步（Phase 5）：
  - 自動化部署（CI/CD）
  - 監控 Endpoint（CloudWatch metrics）
  - 模型版本管理
""")
