# ============================================================
# 03_deploy_endpoint.py
# 主題：用 SageMaker 把模型部署成 Endpoint
#
# SageMaker 部署三步驟：
#   1. 建立 Model：告訴 SageMaker 模型在哪（S3）、用什麼 container
#   2. 建立 Endpoint Config：決定用什麼機器、幾台
#   3. 建立 Endpoint：實際啟動服務（會花幾分鐘）
#
# Endpoint 啟動後，你就有一個 HTTPS API，
# 任何程式都可以呼叫它做推論。
# ============================================================

import json
import time
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# ─────────────────────────────────────────
# 0. 載入設定
# ─────────────────────────────────────────
with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region       = config["region"]
bucket       = config["bucket"]
role         = config["role"]
model_s3_uri = config["model_s3_uri"]

print("=== 設定 ===")
print(f"Region:        {region}")
print(f"Model S3 URI:  {model_s3_uri}")
print(f"Role:          {role}")

# ─────────────────────────────────────────
# 1. 建立 HuggingFace Model
# ─────────────────────────────────────────
print("\n=== 1. 建立 SageMaker Model ===")

# HuggingFaceModel：SageMaker 內建的 HuggingFace container
# 省去你自己打包 Docker image 的麻煩
# transformers_version / pytorch_version：指定 container 版本

huggingface_model = HuggingFaceModel(
    model_data=model_s3_uri,          # S3 上的 model.tar.gz
    role=role,                         # IAM role
    transformers_version="4.37",       # transformers 版本
    pytorch_version="2.1",             # PyTorch 版本
    py_version="py310",                # Python 版本
    sagemaker_session=sagemaker.Session(),
)

print("HuggingFaceModel 建立完成")
print(f"  model_data: {model_s3_uri}")
print(f"  container:  transformers=4.37, pytorch=2.1")

# ─────────────────────────────────────────
# 2. 部署 Endpoint
# ─────────────────────────────────────────
print("\n=== 2. 部署 Endpoint（需要幾分鐘）===")
print("SageMaker 正在：")
print("  1. 啟動 EC2 instance")
print("  2. 拉取 Docker container")
print("  3. 從 S3 下載模型")
print("  4. 執行 model_fn 載入模型")
print("  5. 等待健康檢查通過")
print("...")

ENDPOINT_NAME = "mlops-sentiment-endpoint"

# ml.m5.large：2 vCPU、8GB RAM，適合小模型
# 費用約 $0.115/hr（us-west-2），記得用完要刪
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=ENDPOINT_NAME,
)

print(f"\nEndpoint 部署完成！")
print(f"Endpoint 名稱: {ENDPOINT_NAME}")

# ─────────────────────────────────────────
# 3. 更新設定
# ─────────────────────────────────────────
config["endpoint_name"] = ENDPOINT_NAME

with open("./phase4/aws_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"設定已更新：endpoint_name = {ENDPOINT_NAME}")

# ─────────────────────────────────────────
# 4. 快速測試
# ─────────────────────────────────────────
print("\n=== 3. 快速測試 Endpoint ===")

import json as _json

test_payload = {"texts": ["This movie was absolutely amazing!"]}

print(f"送出請求: {test_payload}")

response = predictor.predict(test_payload)
print(f"回應: {response}")

print("""
=== 重點整理 ===

SageMaker Endpoint 部署流程：
  S3 model.tar.gz
    → HuggingFaceModel（指定 container 版本）
    → .deploy()（指定機器規格）
    → Endpoint（HTTPS API）

instance_type 選擇：
  ml.m5.large   → CPU，便宜，適合測試（$0.115/hr）
  ml.g4dn.xlarge → GPU，適合大模型或高流量（$0.736/hr）

⚠️  記得用完刪掉 Endpoint，否則持續收費！
  執行 04_invoke_endpoint.py 測試完後，
  呼叫 predictor.delete_endpoint() 或到 Console 刪除

下一步（04）：
  呼叫這個 Endpoint 做推論，模擬真實 API 使用
""")
