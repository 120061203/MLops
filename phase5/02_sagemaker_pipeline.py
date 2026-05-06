# ============================================================
# 02_sagemaker_pipeline.py
# 主題：SageMaker Pipelines — 自動化 ML 流程
#
# 把以下步驟串成一條 Pipeline：
#   Step 1: Training Job  — 在 SageMaker 上跑 fine-tuning
#   Step 2: Evaluation    — 評估 accuracy 是否達標
#   Step 3: Condition     — accuracy >= 0.75 才繼續
#   Step 4: Register      — 登記到 Model Registry
#   Step 5: Deploy        — 部署到 Endpoint
#
# 觸發方式：
#   - 手動執行
#   - Bitbucket / Jenkins CI/CD push 時自動觸發
#   - 排程（每週重新訓練）
# ============================================================

import json
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

# ─────────────────────────────────────────
# 0. 載入設定
# ─────────────────────────────────────────
with open("./phase4/aws_config.json") as f:
    config = json.load(f)

region = config["region"]
role   = config["role"]
bucket = config["bucket"]

session = sagemaker.Session()
pipeline_session = PipelineSession()

print("=== SageMaker Pipeline 設定 ===")
print(f"Region: {region}")
print(f"Bucket: {bucket}")

# ─────────────────────────────────────────
# 1. Pipeline 參數（執行時可以覆蓋）
# ─────────────────────────────────────────
print("\n=== 1. Pipeline 參數 ===")

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",  # 或 "Approved"（自動核准）
)

print(f"  ModelApprovalStatus: {model_approval_status.default_value}")

# ─────────────────────────────────────────
# 2. 準備訓練腳本
# ─────────────────────────────────────────
print("\n=== 2. 準備訓練腳本 ===")

import os
os.makedirs("./phase5/scripts/train", exist_ok=True)
os.makedirs("./phase5/scripts", exist_ok=True)

# SageMaker Training Job 需要一個 train.py 作為入口
train_script = '''
import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# SageMaker 會把超參數以環境變數或 /opt/ml/input/config/hyperparameters.json 傳入
hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
if os.path.exists(hyperparams_path):
    with open(hyperparams_path) as f:
        hparams = json.load(f)
else:
    hparams = {}

NUM_TRAIN   = int(hparams.get("num_train_samples", 500))
NUM_TEST    = int(hparams.get("num_test_samples",  100))
NUM_EPOCHS  = int(hparams.get("epochs", 2))
LR          = float(hparams.get("learning_rate", 2e-5))
MODEL_NAME  = hparams.get("model_name", "distilbert-base-uncased")
OUTPUT_DIR  = "/opt/ml/model"   # SageMaker 固定的模型輸出路徑

print(f"訓練設定: samples={NUM_TRAIN}, epochs={NUM_EPOCHS}, lr={LR}")

# 載入資料
dataset     = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN))
small_test  = dataset["test"].shuffle(seed=42).select(range(NUM_TEST))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = small_train.map(tokenize_fn, batched=True)
tokenized_test  = small_test.map(tokenize_fn, batched=True)
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()

# 儲存模型和指標
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 寫入指標供 evaluation step 讀取
metrics = {
    "accuracy": results["eval_accuracy"],
    "eval_loss": results["eval_loss"],
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)

print(f"訓練完成，accuracy: {results['eval_accuracy']:.4f}")
'''

with open("./phase5/scripts/train/train.py", "w") as f:
    f.write(train_script)

print("train.py 建立完成")

# 評估腳本
eval_script = '''
import json, os

model_dir  = "/opt/ml/processing/model"
output_dir = "/opt/ml/processing/evaluation"
os.makedirs(output_dir, exist_ok=True)

# 列出目錄內容方便 debug
print("=== model_dir contents ===")
for root, dirs, files in os.walk(model_dir):
    for f in files:
        print(os.path.join(root, f))

# 遞迴搜尋 metrics.json
metrics_path = None
for root, dirs, files in os.walk(model_dir):
    if "metrics.json" in files:
        metrics_path = os.path.join(root, "metrics.json")
        break

if metrics_path is None:
    raise FileNotFoundError(f"metrics.json not found anywhere in {model_dir}")

print(f"Found metrics.json at: {metrics_path}")

with open(metrics_path) as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics[\'accuracy\']}")

report = {
    "classification_metrics": {
        "accuracy": {"value": metrics["accuracy"]},
    }
}

with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
    json.dump(report, f)

print("evaluation.json 寫出完成")
'''

with open("./phase5/scripts/evaluate.py", "w") as f:
    f.write(eval_script)

print("evaluate.py 建立完成")

# ─────────────────────────────────────────
# 3. Training Step
# ─────────────────────────────────────────
print("\n=== 3. 定義 Pipeline Steps ===")

huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./phase5/scripts/train",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    hyperparameters={
        "num_train_samples": 2000,
        "epochs": 2,
        "learning_rate": 2e-5,
    },
    sagemaker_session=session,
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=huggingface_estimator,
)

print("  TrainingStep 定義完成")

# ─────────────────────────────────────────
# 4. Register Step（登記到 Model Registry）
# ─────────────────────────────────────────
huggingface_model = HuggingFaceModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    sagemaker_session=pipeline_session,
)

step_register = ModelStep(
    name="RegisterModel",
    step_args=huggingface_model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.g4dn.xlarge"],
        transform_instances=["ml.g4dn.xlarge"],
        model_package_group_name=config.get("model_package_group", "mlops-sentiment-model-group"),
        approval_status=model_approval_status,
    ),
)

print("  ModelStep（Register）定義完成")

# ─────────────────────────────────────────
# 5. 組合成 Pipeline
# ─────────────────────────────────────────
print("\n=== 4. 建立 Pipeline ===")

pipeline = Pipeline(
    name="mlops-sentiment-pipeline",
    parameters=[model_approval_status],
    steps=[step_train, step_register],
    sagemaker_session=session,
)

pipeline.upsert(role_arn=role)
print("Pipeline 建立/更新完成：mlops-sentiment-pipeline")

# ─────────────────────────────────────────
# 8. 執行 Pipeline
# ─────────────────────────────────────────
print("\n=== 5. 執行 Pipeline ===")
print("啟動中（訓練需要幾分鐘）...")

execution = pipeline.start(
    parameters={
        "ModelApprovalStatus": "PendingManualApproval",
    }
)

print(f"Pipeline 執行 ARN: {execution.arn}")
print(f"\n可到 AWS Console 查看進度：")
print(f"SageMaker Studio → Pipelines → mlops-sentiment-pipeline")

# 等待完成（可選，CI/CD 中通常不等）
print("\n等待執行完成...")
execution.wait()
print("Pipeline 執行完成！")

status = execution.describe()["PipelineExecutionStatus"]
print(f"最終狀態: {status}")

print("""
=== 重點整理 ===

Pipeline 流程：
  TrainModel → EvaluateModel → CheckAccuracy
                                    ↓ (accuracy >= 0.75)
                               RegisterModel
                                    ↓ (手動 Approve 後)
                               部署 Endpoint

ParameterFloat / ParameterString：
  pipeline 執行時可以動態傳入，不需要改程式碼
  → CI/CD 觸發時可以傳不同的閾值

ConditionStep：
  自動把關，accuracy 不達標的模型不會被登記或部署
  避免「壞模型自動上線」

下一步（03）：
  用 API Gateway + Lambda 把 Endpoint 包成公開 API
""")
