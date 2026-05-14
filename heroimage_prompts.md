# Hero Image Prompts — MLOps 系列

以下為 5 篇 blog 的 hero image 生成 prompt，適用於 Midjourney、DALL·E 3、Stable Diffusion。

---

## Phase 1 — ML 核心工具（NumPy / PyTorch / 訓練迴圈）

**風格**：等距插圖（isometric），科技感，藍橘配色

```
Isometric illustration of a 3D grid of glowing blue tensors and matrices floating in space, connected by orange arrows showing data flow through a neural network training loop, minimalist tech style, clean white background, soft shadows, high detail
```

---

## Phase 2 — Transformer 架構與 Self-Attention

**風格**：抽象視覺化，強調詞與詞之間的連線關係

```
Abstract visualization of self-attention mechanism, multiple glowing tokens connected by curved lines of varying thickness representing attention weights, transformer architecture cross-section in the background, deep navy blue and electric teal color palette, futuristic data science aesthetic, high resolution
```

---

## Phase 3 — Fine-Tuning 預訓練模型

**風格**：「微調」的視覺隱喻，鎖與解鎖的概念

```
Isometric illustration of a large pre-trained language model represented as a glowing crystal structure, with fine-tuning arrows gently reshaping its surface, small labeled dataset cards feeding into the model, purple and gold color palette, minimalist flat design, clean background
```

---

## Phase 4 — AWS SageMaker 部署

**風格**：雲端部署架構圖，強調本機到雲端的流程

```
Isometric cloud deployment diagram showing a laptop packaging a machine learning model into a container, uploading to AWS S3 bucket, deploying to SageMaker endpoint represented as a glowing server rack, connected by animated pipeline arrows, AWS orange and dark navy blue palette, clean tech illustration style
```

---

## Phase 5 — MLOps 全流程自動化

**風格**：閉環流程圖，強調自動化與監控

```
Isometric illustration of a complete MLOps lifecycle loop, showing CI/CD pipeline triggering model training, model registry version cards stacking up, deployment to cloud endpoint, CloudWatch monitoring dashboard with real-time graphs, all connected in a circular automated flow, green and dark blue color palette, futuristic operations center aesthetic
```

---

## 使用建議

| 工具 | 建議設定 |
|------|----------|
| **Midjourney** | 在 prompt 後加 `--ar 16:9 --style raw --v 6` |
| **DALL·E 3** | 直接貼入，指定「寬幅橫向構圖」 |
| **Stable Diffusion** | 搭配 `negative prompt: text, watermark, blurry, low quality` |

尺寸建議：**1200 × 630 px**（Open Graph 標準，社群分享最佳比例）
