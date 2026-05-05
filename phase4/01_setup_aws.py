# ============================================================
# 01_setup_aws.py
# 主題：確認 AWS 環境、建立 S3 bucket、確認 SageMaker IAM role
#
# 部署到 SageMaker 需要三樣東西：
#   1. S3 bucket：放模型檔案
#   2. IAM role：讓 SageMaker 有權限讀 S3、寫 log
#   3. SageMaker SDK：用 Python 操控 SageMaker
#
# 架構：
#   本機模型 → S3 → SageMaker Endpoint → API 呼叫推論
# ============================================================

import boto3
import sagemaker
import json

# ─────────────────────────────────────────
# 1. 確認 AWS 身份
# ─────────────────────────────────────────
print("=== 1. 確認 AWS 身份 ===")

sts = boto3.client("sts")
identity = sts.get_caller_identity()

print(f"Account ID: {identity['Account']}")
print(f"User ARN:   {identity['Arn']}")

# ─────────────────────────────────────────
# 2. 取得 SageMaker session 與預設資訊
# ─────────────────────────────────────────
print("\n=== 2. SageMaker Session ===")

session = sagemaker.Session()
region  = session.boto_region_name
bucket  = session.default_bucket()   # SageMaker 自動建立的預設 bucket

print(f"Region:         {region}")
print(f"Default bucket: {bucket}")
print("（SageMaker 會自動建立這個 bucket，名稱含 account id）")

# ─────────────────────────────────────────
# 3. 確認 SageMaker Execution Role
# ─────────────────────────────────────────
print("\n=== 3. SageMaker Execution Role ===")

# SageMaker 需要一個 IAM role 來代表它執行動作（讀 S3、寫 CloudWatch log）
# 如果你是從 SageMaker Notebook 執行，get_execution_role() 會自動取得
# 如果是從本機執行，需要手動指定已建立的 role ARN

try:
    role = sagemaker.get_execution_role()
    print(f"Execution Role: {role}")
    print("（從 SageMaker 環境自動取得）")
except Exception:
    # 本機執行時 get_execution_role() 會失敗，使用 Domain 上的 role
    role = "arn:aws:iam::081743246838:role/service-role/SageMakerExecutionRole-20260324T120260"
    print(f"本機執行，使用 SageMaker Domain 的 role: {role}")

# ─────────────────────────────────────────
# 4. 確認 S3 bucket 存在
# ─────────────────────────────────────────
print("\n=== 4. 確認 S3 Bucket ===")

s3 = boto3.client("s3", region_name=region)

try:
    s3.head_bucket(Bucket=bucket)
    print(f"Bucket '{bucket}' 已存在")
except Exception:
    print(f"建立 bucket: {bucket}")
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
    print(f"Bucket '{bucket}' 建立完成")

# 列出 bucket 中的物件（看有沒有東西）
response = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
count = response.get("KeyCount", 0)
print(f"Bucket 中目前有 {count} 個檔案")

# ─────────────────────────────────────────
# 5. 安裝確認
# ─────────────────────────────────────────
print("\n=== 5. 套件版本確認 ===")

import boto3 as b3
import sagemaker as sm
import transformers as tf
import torch

print(f"boto3:        {b3.__version__}")
print(f"sagemaker:    {sm.__version__}")
print(f"transformers: {tf.__version__}")
print(f"torch:        {torch.__version__}")

# ─────────────────────────────────────────
# 6. 儲存設定給後續 py 使用
# ─────────────────────────────────────────
print("\n=== 6. 儲存設定 ===")

config = {
    "region":     region,
    "bucket":     bucket,
    "role":       role,
    "account_id": identity["Account"],
    "s3_prefix":  "mlops-phase4",   # 模型在 S3 的資料夾前綴
}

with open("./phase4/aws_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("設定已儲存至 ./phase4/aws_config.json")
print(json.dumps(config, indent=2))

print("""
=== 重點整理 ===

SageMaker 部署需要的元件：
  S3 bucket        → 放模型權重檔（model.tar.gz）
  IAM role         → SageMaker 的身份證，決定它能做什麼
  SageMaker SDK    → Python 操控 SageMaker 的工具

下一步（02）：
  把 phase3 fine-tuned 的模型打包成 model.tar.gz 上傳到 S3
""")
