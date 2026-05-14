# MLOps 學習筆記

從 ML 核心工具到 AWS 全流程自動化部署的完整學習路徑。

---

## 學習路徑總覽

| Phase | 主題 | 核心技術 | 練習目標 |
|-------|------|----------|----------|
| [Phase 1](#phase-1--ml-核心工具) | ML 核心工具 | NumPy、PyTorch、Autograd | 從零訓練線性迴歸 |
| [Phase 2](#phase-2--模型架構) | 模型架構 | Transformer、Self-Attention、HuggingFace | 載入預訓練模型做推論 |
| [Phase 3](#phase-3--fine-tuning) | Fine-Tuning | DistilBERT、Trainer API | Fine-tune 情感分類模型 |
| [Phase 4](#phase-4--aws-部署) | AWS 部署 | SageMaker、S3、inference.py | 部署模型為 HTTPS API |
| [Phase 5](#phase-5--全流程自動化) | 全流程自動化 | Model Registry、Pipeline、CloudWatch | 訓練→評估→部署閉環 |

---

## Phase 1 — ML 核心工具

**目錄：** `phase1/`

| 檔案 | 主題 |
|------|------|
| `01_numpy_basics.py` | 矩陣運算、Broadcasting、向量化思維 |
| `02_tensor_basics.py` | PyTorch Tensor 建立與操作 |
| `03_autograd.py` | 自動微分原理、計算圖 |
| `04_training_loop.py` | 完整訓練迴圈（Forward → Loss → Backward → Step） |
| `05_dataset_dataloader.py` | 自定義 Dataset、DataLoader、Train/Val 分割 |

**Blog：** [MLOps 學習筆記（一）](https://xsong.us/blog/2026/05/mlops_phase1)

---

## Phase 2 — 模型架構

**目錄：** `phase2/`

| 檔案 | 主題 |
|------|------|
| `01_neural_network.py` | 前向傳播、Activation Function |
| `02_backprop.py` | 反向傳播、chain rule |
| `03_transformer_attention.py` | Self-Attention、Scaled Dot-Product Attention |
| `04_transformer_model.py` | Multi-Head Attention、完整 Transformer Block |
| `05_huggingface_intro.py` | AutoTokenizer、AutoModel、Pipeline API |

**Blog：** [MLOps 學習筆記（二）](https://xsong.us/blog/2026/05/mlops_phase2)

---

## Phase 3 — Fine-Tuning

**目錄：** `phase3/`

| 檔案 | 主題 |
|------|------|
| `01_finetune_text_classification.py` | HuggingFace Trainer API fine-tune DistilBERT |
| `02_finetune_custom_loop.py` | 手動訓練迴圈、LR Scheduler、Gradient Clipping |
| `03_evaluate_and_save.py` | Accuracy / F1 / Confusion Matrix、模型儲存與載入 |
| `04_inference_pipeline.py` | Pipeline API 推論 |

**Blog：** [MLOps 學習筆記（三）](https://xsong.us/blog/2026/05/mlops_phase3)

---

## Phase 4 — AWS 部署

**目錄：** `phase4/`

| 檔案 | 主題 |
|------|------|
| `01_setup_aws.py` | AWS Session、IAM Role、S3 Bucket 初始化 |
| `02_upload_model.py` | 打包 model.tar.gz、撰寫 inference.py、上傳 S3 |
| `03_deploy_endpoint.py` | HuggingFaceModel、SageMaker Endpoint 部署 |
| `04_invoke_endpoint.py` | 呼叫 Endpoint 做推論 |
| `compare_models.py` | Baseline vs Fine-tuned 模型效果對比 |

**Blog：** [MLOps 學習筆記（四）](https://xsong.us/blog/2026/05/mlops_phase4)

---

## Phase 5 — 全流程自動化

**目錄：** `phase5/`

| 檔案 | 主題 |
|------|------|
| `01_model_registry.py` | SageMaker Model Registry 版本管理 |
| `02_sagemaker_pipeline.py` | Training → Register 自動化 Pipeline |
| `03_api_gateway.py` | API Gateway + Lambda 公開 Endpoint |
| `04a_bitbucket_pipeline.yml` | Bitbucket CI/CD 觸發 Pipeline |
| `04b_jenkins_pipeline.groovy` | Jenkins CI/CD 觸發 Pipeline |
| `05_monitoring.py` | CloudWatch Dashboard、警報設定、SNS 通知 |

**Blog：** [MLOps 學習筆記（五）](https://xsong.us/blog/2026/05/mlops_phase5)

---

## 其他檔案

| 檔案 | 說明 |
|------|------|
| `guide.md` | 完整學習指南與工具清單 |
| `mlops_exercise.md` | 延伸練習：Data Engineering 與 ETL Pipeline |
| `heroimage_prompts.md` | Blog 封面圖 AI 生成 prompt（Midjourney / DALL·E） |

---

## 環境安裝

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install transformers datasets peft evaluate accelerate
pip install boto3 sagemaker scikit-learn
```

Python 3.10+ 建議。

---

## 工具清單

| 工具 | 用途 |
|------|------|
| `numpy` | 數值運算 |
| `torch` | 模型訓練核心 |
| `transformers` | Pre-trained model 載入與推論 |
| `datasets` | 資料集管理 |
| `peft` | LoRA / 參數高效微調 |
| `evaluate` | 模型評估指標 |
| `boto3` | AWS SDK |
| `sagemaker` | AWS SageMaker Python SDK |
