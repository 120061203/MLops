# ============================================================
# 01_finetune_text_classification.py
# 主題：用 HuggingFace Trainer API fine-tune 文字分類
#
# Fine-tuning 白話解釋：
#   DistilBERT 預訓練時看過幾億句英文，學會了「語言的感覺」。
#   但它不知道「正面/負面」是什麼。
#
#   Fine-tuning = 拿這個已經懂語言的模型，
#   餵給它你的標記資料（正/負面評論），
#   讓它的參數慢慢調整，學會這個新任務。
#
#   就像一個英文很好的人，你教他分辨「客訴信」和「讚美信」，
#   他很快就學會了——因為他已經懂語言了。
# ============================================================

import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# ─────────────────────────────────────────
# 自訂 Callback：把訓練過程印出來
# ─────────────────────────────────────────
# Trainer 預設只印簡略 log，這裡自訂 callback 讓每一步都說話

class VerboseCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n{'='*55}")
        print(f"  開始 Fine-tuning！")
        print(f"  總共 {state.max_steps} 步，{int(args.num_train_epochs)} 個 epoch")
        print(f"  每 epoch = {state.max_steps // int(args.num_train_epochs)} 步")
        print(f"{'='*55}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + 1
        print(f"\n【Epoch {epoch} 開始】")
        print(f"  模型開始看第 {epoch} 輪訓練資料...")
        print(f"  目前步數: {state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        # 每 10 步印一次，避免刷屏
        if state.global_step % 10 == 0 and state.global_step > 0:
            print(f"  Step {state.global_step:3d}/{state.max_steps} "
                  f"| 模型正在更新參數中...")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            print(f"\n  >> 訓練 loss: {logs['loss']:.4f}  "
                  f"(loss 越低代表模型預測越準確)")
            print(f"     learning rate: {logs.get('learning_rate', 0):.2e}  "
                  f"(lr 會隨訓練慢慢變小)")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = round(state.epoch)
        loss  = metrics.get("eval_loss", 0)
        acc   = metrics.get("eval_accuracy", 0)
        print(f"\n  ── Epoch {epoch} 評估結果 ──")
        print(f"     eval_loss:     {loss:.4f}  ← 測試集上的 loss")
        print(f"     eval_accuracy: {acc:.4f}  ← 答對比例 ({acc*100:.1f}%)")
        print(f"  (Epoch 1→2 若 loss 下降、accuracy 上升，代表模型在進步)")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        print(f"\n【Epoch {epoch} 結束】模型已儲存 checkpoint")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*55}")
        print(f"  Fine-tuning 完成！共訓練 {state.global_step} 步")
        print(f"{'='*55}")

# ─────────────────────────────────────────
# 1. 載入資料集
# ─────────────────────────────────────────
print("=== 步驟 1：載入資料集 ===")
print("IMDb 電影評論資料集，label: 0=負面, 1=正面")

dataset     = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(500))
small_test  = dataset["test"].shuffle(seed=42).select(range(100))

print(f"訓練集: {len(small_train)} 筆")
print(f"測試集: {len(small_test)} 筆")

# 印幾筆讓你看看資料長什麼樣
print("\n範例資料（訓練集前 3 筆）：")
for i in range(3):
    label_str = "正面" if small_train[i]["label"] == 1 else "負面"
    print(f"  [{label_str}] {small_train[i]['text'][:70]}...")

# ─────────────────────────────────────────
# 2. Tokenize
# ─────────────────────────────────────────
print("\n=== 步驟 2：Tokenize（文字 → 數字） ===")
print("模型看不懂文字，要先把文字轉成 token id（數字序列）")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 先示範一筆讓你看轉換結果
sample_text = small_train[0]["text"][:60]
sample_enc  = tokenizer(sample_text)
print(f"\n原始文字: {sample_text}")
print(f"轉成 token ids: {sample_enc['input_ids']}")
print(f"token 數量: {len(sample_enc['input_ids'])}")
print(f"解碼回來: {tokenizer.decode(sample_enc['input_ids'])}")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

print("\n對全部資料 tokenize...")
tokenized_train = small_train.map(tokenize_fn, batched=True)
tokenized_test  = small_test.map(tokenize_fn, batched=True)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

print(f"每筆資料統一 padding 到長度 128")
print(f"input_ids shape: {tokenized_train[0]['input_ids'].shape}")

# ─────────────────────────────────────────
# 3. 載入預訓練模型
# ─────────────────────────────────────────
print("\n=== 步驟 3：載入預訓練模型 ===")
print("載入 distilbert-base-uncased（DistilBERT）")
print("這個模型已經在大量英文文字上預訓練，懂得語言的語義")
print("我們在它上面加一個分類頭（Linear layer），讓它能輸出正/負面")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n總參數量: {total_params:,}")
print(f"可訓練參數: {trainable_params:,}  ← fine-tuning 會調整這些")

# ─────────────────────────────────────────
# 4. 評估指標
# ─────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# ─────────────────────────────────────────
# 5. 訓練前先測一次（baseline）
# ─────────────────────────────────────────
print("\n=== 步驟 4：訓練前 baseline（還沒 fine-tune 的準確率）===")
print("先看看未訓練的模型表現多差，之後對比才有感覺")

model.eval()
device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
model.to(device)

from torch.utils.data import DataLoader
loader = DataLoader(tokenized_test, batch_size=32)
correct = 0
with torch.no_grad():
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["label"].to(device)
        out   = model(input_ids=ids, attention_mask=mask)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == lbls).sum().item()

baseline_acc = correct / len(tokenized_test)
print(f"Baseline accuracy（未訓練）: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
print("（接近 50% 表示跟亂猜差不多，分類頭還沒學會任何東西）")

# ─────────────────────────────────────────
# 6. 設定訓練參數
# ─────────────────────────────────────────
print("\n=== 步驟 5：設定訓練參數 ===")

# 每次從乾淨的起點開始，避免載入舊 checkpoint 造成起點不同
import shutil, os
if os.path.exists("./phase3/checkpoints"):
    shutil.rmtree("./phase3/checkpoints")

training_args = TrainingArguments(
    output_dir="./phase3/checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=32,
    report_to="none",
)

print(f"  num_train_epochs: {training_args.num_train_epochs}")
print(f"    → 全部 500 筆資料看 2 遍")
print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
print(f"    → 每次餵 16 筆給模型，算 loss，更新一次參數")
print(f"  learning_rate: {training_args.learning_rate}")
print(f"    → 每次更新參數的幅度，2e-5 很小是為了不破壞預訓練的知識")
print(f"  eval_strategy: epoch")
print(f"    → 每輪結束後在測試集上評估一次，看有沒有進步")

# ─────────────────────────────────────────
# 7. 建立 Trainer 並訓練
# ─────────────────────────────────────────
print("\n=== 步驟 6：Fine-tuning 開始 ===")
print("Trainer 幫你跑訓練迴圈：")
print("  for batch in 訓練資料:")
print("      loss = model(batch)")
print("      loss.backward()   # 算梯度")
print("      optimizer.step()  # 更新參數")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[VerboseCallback()],
)

trainer.train()

# ─────────────────────────────────────────
# 8. 訓練後評估
# ─────────────────────────────────────────
print("\n=== 步驟 7：最終評估 ===")
results = trainer.evaluate()
final_acc = results["eval_accuracy"]

print(f"\n  Baseline accuracy（訓練前）: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
print(f"  Final accuracy  （訓練後）: {final_acc:.4f}  ({final_acc*100:.1f}%)")
print(f"  提升: +{(final_acc - baseline_acc)*100:.1f}%")
print(f"\n  （只用 500 筆訓練資料，若用全部 25,000 筆可達 92%+）")

# ─────────────────────────────────────────
# 9. 儲存
# ─────────────────────────────────────────
print("\n=== 步驟 8：儲存模型 ===")
trainer.save_model("./phase3/finetuned_model")
tokenizer.save_pretrained("./phase3/finetuned_model")
print("模型儲存至 ./phase3/finetuned_model/")
print("之後可以用 AutoModel.from_pretrained('./phase3/finetuned_model') 直接載入")

print("""
=== Fine-tuning 總結 ===

整個過程做了什麼：
  1. 載入 DistilBERT（預訓練好的語言模型）
  2. 在上面加一個分類頭（一層 Linear）
  3. 餵入 500 筆標記好的評論（正/負面）
  4. 每次餵 16 筆，算預測有多錯（loss）
  5. 用 loss 算梯度，往「更準確」的方向微調參數
  6. 重複 2 輪（epoch）

為什麼 learning_rate 要很小（2e-5）：
  預訓練模型的參數已經「很有價值」。
  用太大的 lr 一次改太多，會把原本學到的語言知識破壞掉。

下一個檔案（02）：把這個過程拆開，手動寫訓練迴圈
""")
