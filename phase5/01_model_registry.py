# ============================================================
# 01_model_registry.py
# 主題：SageMaker Model Registry — 模型版本管理
#
# 問題：每次 fine-tune 都會產生新模型，沒有管理的話：
#   - 不知道哪個版本在線上
#   - 不知道每個版本的 accuracy 是多少
#   - 出問題無法快速 rollback
#
# Model Registry 解決這些問題：
#   - 每個模型都有版本號
#   - 記錄每個版本的指標（accuracy、F1）
#   - 部署前需要審核（Approved / Rejected）
#   - 可以一鍵 rollback 到舊版本
# ============================================================

import json
import boto3
import sagemaker
from sagemaker.model import Model

# ─────────────────────────────────────────
# 0. 載入設定
# ─────────────────────────────────────────
with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region       = config["region"]
role         = config["role"]
bucket       = config["bucket"]
model_s3_uri = config["model_s3_uri"]

session = sagemaker.Session()
sm      = boto3.client("sagemaker", region_name=region)

print("=== SageMaker Model Registry ===\n")

# ─────────────────────────────────────────
# 1. 建立 Model Package Group（模型的「專案」）
# ─────────────────────────────────────────
print("=== 1. 建立 Model Package Group ===")

GROUP_NAME = "mlops-sentiment-model-group"

try:
    sm.create_model_package_group(
        ModelPackageGroupName=GROUP_NAME,
        ModelPackageGroupDescription="IMDb 情感分類模型版本管理",
    )
    print(f"Model Package Group 建立：{GROUP_NAME}")
except sm.exceptions.ClientError as e:
    if "already exists" in str(e) or "ConflictException" in str(e):
        print(f"Model Package Group 已存在：{GROUP_NAME}")
    else:
        raise

# ─────────────────────────────────────────
# 2. 取得 HuggingFace inference container URI
# ─────────────────────────────────────────
from sagemaker.image_uris import retrieve

container_uri = retrieve(
    framework="huggingface",
    region=region,
    version="4.37",
    py_version="py310",
    base_framework_version="pytorch2.1.0",
    image_scope="inference",
    instance_type="ml.m5.large",
)
print(f"\nContainer URI: {container_uri}")

# ─────────────────────────────────────────
# 3. 登記模型版本（Model Package）
# ─────────────────────────────────────────
print("\n=== 2. 登記模型版本 ===")

# 模擬這次訓練的指標（實際應從訓練結果讀取）
metrics = {
    "accuracy": 0.84,
    "f1_score": 0.83,
    "train_samples": 500,
    "epochs": 2,
}

model_package = sm.create_model_package(
    ModelPackageGroupName=GROUP_NAME,
    ModelPackageDescription=f"DistilBERT fine-tuned, accuracy={metrics['accuracy']}",
    InferenceSpecification={
        "Containers": [
            {
                "Image": container_uri,
                "ModelDataUrl": model_s3_uri,
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
    },
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": model_s3_uri,  # 實際應指向指標 JSON 檔案
            }
        }
    },
    CustomerMetadataProperties={
        "accuracy":       str(metrics["accuracy"]),
        "f1_score":       str(metrics["f1_score"]),
        "train_samples":  str(metrics["train_samples"]),
        "epochs":         str(metrics["epochs"]),
        "base_model":     "distilbert-base-uncased",
    },
)

model_package_arn = model_package["ModelPackageArn"]
print(f"模型版本 ARN: {model_package_arn}")
print(f"指標: accuracy={metrics['accuracy']}, f1={metrics['f1_score']}")

# ─────────────────────────────────────────
# 4. 審核模型（Approve）
# ─────────────────────────────────────────
print("\n=== 3. 審核模型 ===")
print("實際流程：由 ML 工程師審核指標後決定 Approve 或 Reject")
print("這裡自動 Approve（CI/CD pipeline 中可設定閾值自動審核）")

sm.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",  # 或 "Rejected"
)
print(f"模型狀態：Approved")

# ─────────────────────────────────────────
# 5. 列出所有版本
# ─────────────────────────────────────────
print("\n=== 4. 所有模型版本 ===")

versions = sm.list_model_packages(
    ModelPackageGroupName=GROUP_NAME,
    SortBy="CreationTime",
    SortOrder="Descending",
)

for v in versions["ModelPackageSummaryList"]:
    status = v["ModelApprovalStatus"]
    ver    = v["ModelPackageVersion"]
    arn    = v["ModelPackageArn"]
    print(f"  版本 {ver}: [{status}] {arn.split('/')[-1]}")

# ─────────────────────────────────────────
# 6. 更新設定供後續使用
# ─────────────────────────────────────────
config["model_package_arn"]   = model_package_arn
config["model_package_group"] = GROUP_NAME

with open("./phase4/aws_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n設定已更新：model_package_arn")

print("""
=== 重點整理 ===

Model Registry 核心概念：
  Model Package Group → 一個模型專案（例如：情感分類）
  Model Package       → 一個版本（每次訓練產生一個）
  ApprovalStatus      → Approved / Rejected / PendingManualApproval

為什麼需要審核：
  自動化 pipeline 跑完訓練後，不應該直接部署
  需要確認 accuracy 有達標才 Approve，才能進入部署流程

Rollback：
  若新版本有問題，把舊版本的 ApprovalStatus 改回 Approved
  重新部署即可

下一步（02）：
  用 SageMaker Pipelines 把訓練→評估→登記→部署串成自動流程
""")
