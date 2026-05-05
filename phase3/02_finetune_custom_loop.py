# ============================================================
# 02_finetune_custom_loop.py
# 主題：手動訓練迴圈 fine-tune（不用 Trainer）
#
# 上一個檔案用 Trainer API，幾行就搞定。
# 這個檔案把訓練迴圈全部攤開，讓你看清楚每一步在做什麼：
#
#   for batch in dataloader:
#       output = model(**batch)
#       loss = output.loss
#       loss.backward()
#       optimizer.step()
#
# 這樣的手動迴圈讓你更容易：
#   - 加自訂 loss
#   - 做梯度累積
#   - 插入 debug / logging 邏輯
# ============================================================

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ─────────────────────────────────────────
# 1. 資料準備
# ─────────────────────────────────────────
print("=== 1. 資料準備 ===")

dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(500))
small_test  = dataset["test"].shuffle(seed=42).select(range(100))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = small_train.map(tokenize_fn, batched=True)
tokenized_test  = small_test.map(tokenize_fn, batched=True)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

# DataLoader：負責把 dataset 分成 batch 餵進模型
train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True)
test_loader  = DataLoader(tokenized_test,  batch_size=32)

print(f"訓練 batch 數: {len(train_loader)}")
print(f"測試 batch 數: {len(test_loader)}")

# ─────────────────────────────────────────
# 2. 模型與 Optimizer
# ─────────────────────────────────────────
print("\n=== 2. 模型與 Optimizer ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

# AdamW：Adam 加上 weight decay（L2 正則化），fine-tuning 標準選擇
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Learning rate scheduler：warmup 後線性衰減
num_epochs = 2
total_steps = len(train_loader) * num_epochs
warmup_steps = total_steps // 10   # 前 10% 步做 warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"總訓練步數: {total_steps}")
print(f"Warmup 步數: {warmup_steps}")

# ─────────────────────────────────────────
# 3. 訓練函式
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()   # 開啟 dropout 等訓練模式
    total_loss = 0

    for step, batch in enumerate(loader):
        # 把資料搬到 GPU/CPU
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        # Forward pass
        # AutoModelForSequenceClassification 傳入 labels 時會自動計算 cross-entropy loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss   # scalar

        # Backward pass
        optimizer.zero_grad()   # 清除上一步的梯度
        loss.backward()         # 計算梯度

        # Gradient clipping：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()    # 更新參數
        scheduler.step()    # 更新 learning rate

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            avg = total_loss / (step + 1)
            lr  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} Step {step+1}/{len(loader)} | loss: {avg:.4f} | lr: {lr:.2e}")

    return total_loss / len(loader)

# ─────────────────────────────────────────
# 4. 評估函式
# ─────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()   # 關閉 dropout
    correct = 0
    total   = 0

    with torch.no_grad():   # 不計算梯度，省記憶體
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1)   # 取最大 logit 的 index

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total

# ─────────────────────────────────────────
# 5. 執行訓練
# ─────────────────────────────────────────
print("\n=== 3. 開始訓練 ===")

for epoch in range(1, num_epochs + 1):
    print(f"\n--- Epoch {epoch}/{num_epochs} ---")
    avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    accuracy = evaluate(model, test_loader, device)
    print(f"Epoch {epoch} 結束 | avg_loss: {avg_loss:.4f} | test_accuracy: {accuracy:.4f}")

# ─────────────────────────────────────────
# 6. 儲存模型
# ─────────────────────────────────────────
print("\n=== 4. 儲存模型 ===")
model.save_pretrained("./phase3/finetuned_custom")
tokenizer.save_pretrained("./phase3/finetuned_custom")
print("儲存至 ./phase3/finetuned_custom/")

print("""
=== 重點整理 ===

手動訓練迴圈 vs Trainer API：
  手動迴圈：彈性高，能自訂每一步，適合研究
  Trainer：省事，適合快速實驗

訓練迴圈的五個核心步驟：
  1. forward：model(**batch) 得到 loss
  2. zero_grad：清除舊梯度
  3. backward：loss.backward() 計算新梯度
  4. clip_grad：防止梯度爆炸
  5. optimizer.step() + scheduler.step()：更新參數與 lr

model.train() vs model.eval()：
  train()：開啟 Dropout（訓練時隨機關閉神經元，防止過擬合）
  eval() ：關閉 Dropout，確保推論結果穩定
""")
