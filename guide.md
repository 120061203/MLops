# MLOps 學習指南

從 ML 核心工具到 AWS 部署的完整學習路徑。

---

## Phase 1 — ML 核心工具

### 目標
建立 ML 的數學與程式基礎，理解資料在模型中如何流動。

### 主題

#### NumPy
- 矩陣與向量運算
- Broadcasting 機制
- 向量化思維（避免 for loop）

#### PyTorch 基礎
- Tensor 建立與操作
- GPU vs CPU（`.to(device)`）
- Autograd：自動微分原理
- 基本訓練迴圈（forward → loss → backward → step）

#### 資料處理
- `torch.utils.data.Dataset` 自定義資料集
- `DataLoader`：批次、shuffle、多執行緒載入

### 練習目標
用 PyTorch 從零訓練一個簡單的線性迴歸或分類模型。

---

## Phase 2 — 模型架構

### 目標
理解現代 NLP 模型的底層結構，知道 pre-trained model 是什麼。

### 主題

#### 神經網路原理
- 前向傳播（Forward Pass）
- Loss Function（CrossEntropy、MSE）
- 反向傳播（Backpropagation）與梯度下降

#### Transformer 架構
- Self-Attention 機制
- Multi-Head Attention
- Encoder（BERT 系列）vs Decoder（GPT 系列）架構差異
- Positional Encoding

#### HuggingFace 入門
- `transformers` 套件安裝與基本使用
- `AutoTokenizer`、`AutoModel`
- Pipeline API 快速推論

### 練習目標
載入 HuggingFace 上的 pre-trained model，對文字做推論。

---

## Phase 3 — Fine-Tuning（核心重點）

### 目標
學會針對特定任務微調 pre-trained model，這是實務上最常見的工作。

### 主題

#### 全量 Fine-tune（Full Fine-tuning）
- 凍結（freeze）與解凍層的概念
- 學習率設定策略（Learning Rate Scheduler）
- 訓練穩定性注意事項

#### PEFT / LoRA（參數高效微調）
- 為什麼需要 PEFT：節省記憶體與運算資源
- LoRA 原理：低秩矩陣分解
- `peft` 套件使用（HuggingFace）
- QLoRA：量化 + LoRA 的組合

#### 資料準備
- 訓練資料格式（JSON、CSV、Arrow）
- Tokenization 與 padding/truncation
- `datasets` 套件使用

#### 評估
- Train / Validation / Test 切分
- 評估指標（Accuracy、F1、BLEU、Perplexity）
- `evaluate` 套件使用

### 練習目標
用 LoRA fine-tune 一個小型語言模型（如 `Llama` 或 `Mistral`）完成文字分類或問答任務。

---

## Phase 4 — MLOps & AWS 部署

### 目標
將訓練好的模型包裝成可靠的服務，部署到雲端並持續監控。

### 主題

#### 模型打包
- 模型序列化（`torch.save`、`safetensors`）
- Docker 容器化：撰寫 Dockerfile
- 環境依賴管理（`requirements.txt`、`conda`）

#### AWS SageMaker
- SageMaker Training Job：雲端訓練
- SageMaker Endpoint：部署推論服務
- S3：模型 artifact 儲存
- IAM Role 設定

#### 推論優化
- 模型量化（INT8、FP16）
- Batch Inference vs Real-time Inference
- TorchScript / ONNX 匯出

#### 監控
- 推論延遲（Latency）與吞吐量（Throughput）
- 模型漂移（Data Drift）偵測
- CloudWatch Logs 整合

### 練習目標
將 Phase 3 fine-tuned 的模型部署為 SageMaker Endpoint，透過 API 呼叫推論。

---

## 建議工具清單

| 工具 | 用途 |
|------|------|
| `numpy` | 數值運算 |
| `torch` | 模型訓練核心 |
| `transformers` | Pre-trained model 載入與推論 |
| `datasets` | 資料集管理 |
| `peft` | LoRA / 參數高效微調 |
| `evaluate` | 模型評估指標 |
| `accelerate` | 多 GPU / 混合精度訓練 |
| `boto3` | AWS SDK for Python |
| `sagemaker` | AWS SageMaker Python SDK |
| Docker | 容器化部署 |

---

## 開始之前

安裝基本環境：

```bash
pip install torch torchvision torchaudio
pip install transformers datasets peft evaluate accelerate
```

建議使用 Python 3.10+，並以虛擬環境隔離依賴：

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```
