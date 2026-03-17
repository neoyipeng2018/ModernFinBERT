# Plan: Full Fine-Tuning + DataBoost (Notebook 12)

## Objective

Create `notebooks/12_full_ft_databoost.ipynb` — the missing experiment that combines **full fine-tuning** (all 149M parameters unfrozen, no LoRA) with **DataBoost augmentation** (410 VS-CoT samples). This completes the 2x2 comparison matrix:

|  | Base Data | + DataBoost |
|--|-----------|-------------|
| **LoRA r=16** | NB01 (80.44%) | NB02A (82.56%) |
| **Full FT** | NB09e (archived, no results in paper) | **NB12 (this experiment)** |

The hypothesis: Full FT eliminates the LoRA fused-QKV asymmetry that disadvantages ModernBERT, and DataBoost adds targeted augmentation for difficult boundary cases. Together, this should be the strongest configuration.

---

## Architecture

```
Phase A: Full FT Baseline
  Load data (excl FPB) → Train ModernBERT-base (all params) → Evaluate on FPB

Phase B: Full FT + DataBoost
  Load Phase A's augmentation data (410 VS-CoT samples) → Concat with training data
  → Train fresh ModernBERT-base (all params) → Evaluate on FPB

Phase C: Comparison Table
  Aggregate results from NB01, NB02A, NB09e, NB12 into a 2x2 matrix
```

---

## Key Design Decisions

### 1. Hyperparameters: Full FT vs LoRA

Full fine-tuning requires different hyperparameters than LoRA. With 149M trainable parameters (vs ~1.1M for LoRA), the model can overfit aggressively on ~8,643 training samples. We follow the 09e configuration which was already tuned for this:

```python
# LoRA (from 02A)                    # Full FT (from 09e, adopted here)
learning_rate = 2e-4                  learning_rate = 2e-5          # 10x lower
weight_decay  = 0.001                 weight_decay  = 0.01          # 10x higher
warmup        = 10 steps              warmup        = 0.1 ratio     # proportional
grad_accum    = 4 (eff batch 32)      grad_accum    = 2 (eff batch 16)
seed          = 3407                  seed          = 42
```

**Why lower LR**: With all parameters updating, large gradients can destroy pre-trained representations. 2e-5 is the standard BERT full-FT learning rate from the original paper.

**Why higher weight decay**: Stronger regularization prevents 149M parameters from overfitting on 8.6K samples.

**Why smaller effective batch**: More frequent updates with slightly noisier gradients can act as implicit regularization.

### 2. DataBoost Source: Reuse Existing VS Data

We reuse the 410 VS-CoT augmented samples from NB02A (embedded as `VS_DATA_B64`). Rationale:

- **Fair comparison**: Same augmentation data lets us isolate the effect of LoRA vs full FT
- **The confusion patterns are general**: POSITIVE→NEUTRAL (175), NEUTRAL→POSITIVE (115), NEGATIVE→NEUTRAL (85) — these boundary confusions exist regardless of training method
- **Practical**: No API calls needed; the notebook is self-contained and reproducible

A future experiment could re-mine errors from the full FT baseline (which may produce different misclassifications) and generate fresh augmentation — but that's a separate contribution.

### 3. FPB Loading: Robust Fallback

The HuggingFace `financial_phrasebank` dataset loader has been unreliable across environments. We include a zip-file fallback that downloads and parses the raw text files directly. This pattern is proven across NB09b, 09c, 09e.

```python
def load_fpb_dataset(agree_level="50agree"):
    """Load FPB with fallback to zip file parsing."""
    try:
        config = f"sentences_{agree_level}"
        return load_dataset("financial_phrasebank", config, trust_remote_code=True)["train"]
    except Exception:
        from huggingface_hub import hf_hub_download
        import zipfile, os
        path = hf_hub_download(
            "financial_phrasebank",
            "data/FinancialPhraseBank-v1.0.zip",
            repo_type="dataset",
        )
        extract_dir = "/tmp/fpb_data"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)
        filename = {
            "50agree": "Sentences_50Agree.txt",
            "allagree": "Sentences_AllAgree.txt",
        }[agree_level]
        filepath = os.path.join(extract_dir, "FinancialPhraseBank-v1.0", filename)
        sentences, labels = [], []
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                at_idx = line.rfind("@")
                if at_idx == -1:
                    continue
                label_str = line[at_idx + 1 :].strip().lower()
                if label_str in label_map:
                    sentences.append(line[:at_idx].strip())
                    labels.append(label_map[label_str])
        return {"sentence": sentences, "label": labels}
```

### 4. Memory Management

Full FT of ModernBERT-base in FP16 requires ~12-16GB GPU RAM. We enable gradient checkpointing (trades ~30% compute for ~50% memory savings) and clear GPU memory between phases:

```python
model.gradient_checkpointing_enable()

# Between phases:
del model, trainer
gc.collect()
torch.cuda.empty_cache()
```

---

## Implementation: Cell-by-Cell

### Cell 0: Markdown Header

```markdown
# Experiment 12: Full Fine-Tuning + DataBoost

Combine full fine-tuning (all 149M parameters, no LoRA) with DataBoost
VS-CoT augmentation (410 targeted samples). Completes the 2x2 matrix:
LoRA/FullFT × Base/DataBoosted.

**Hypothesis:** Full FT eliminates the LoRA fused-QKV asymmetry and
DataBoost targets decision-boundary errors → best held-out FPB performance.
```

### Cell 1: Setup

```python
%%capture
!pip install -q "datasets>=3.4.1,<4.0.0" scikit-learn matplotlib seaborn accelerate transformers sentencepiece protobuf
```

### Cell 2: Imports & Constants

```python
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, training_args,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns

NUM_CLASSES = 3
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
FPB_SOURCE = 5
```

### Cell 3: Data Loading

Load the aggregated dataset, exclude FPB, convert labels to one-hot.

```python
label_dict = {"NEUTRAL/MIXED": 1, "NEGATIVE": 0, "POSITIVE": 2}

ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
ds = ds.filter(lambda x: x["task"] == "sentiment")
ds = ds.filter(lambda x: x["source"] != FPB_SOURCE)

remove_cols = [c for c in ds["train"].column_names if c not in ("text", "labels")]
ds = ds.map(
    lambda ex: {
        "text": ex["text"],
        "labels": np.eye(NUM_CLASSES)[label_dict[ex["label"]]],
    },
    remove_columns=remove_cols,
)

print(f"Train: {len(ds['train']):,}  |  Val: {len(ds['validation']):,}  |  Test: {len(ds['test']):,}")
```

Load FPB test sets with fallback:

```python
# [load_fpb_dataset function from above]

fpb_50 = load_fpb_dataset("50agree")
fpb_all = load_fpb_dataset("allagree")
print(f"FPB 50agree: {len(fpb_50['sentence']):,}  |  FPB allAgree: {len(fpb_all['sentence']):,}")
```

### Cell 4: Evaluation Helper

Shared function for FPB evaluation. Returns accuracy, macro F1, confusion matrix.

```python
def evaluate_on_fpb(model, tokenizer, fpb_dataset, batch_size=32):
    fpb_texts = fpb_dataset["sentence"]
    fpb_labels = fpb_dataset["label"]
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(fpb_texts), batch_size), desc="Evaluating"):
            batch = fpb_texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
    y_true = np.array(fpb_labels)
    y_pred = np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({int(acc * len(y_true))}/{len(y_true)})")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)
    return {"accuracy": acc, "macro_f1": macro_f1, "cm": cm}
```

### Cell 5: Full FT Training Function

This is the core. No LoRA — all parameters unfrozen. Hyperparameters tuned for full FT.

```python
def train_full_ft(train_dataset, val_dataset, output_dir="trainer_output", epochs=10):
    """Full fine-tune ModernBERT-base (all 149M parameters)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=NUM_CLASSES,
    )
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,} (ALL)")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    train_tok = train_dataset.map(tokenize_fn, batched=True)
    val_tok = val_dataset.map(tokenize_fn, batched=True)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            fp16=True,
            optim=training_args.OptimizerNames.ADAMW_TORCH,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            num_train_epochs=epochs,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=2,
        ),
        compute_metrics=lambda eval_pred: {
            "accuracy": accuracy_score(
                eval_pred[1].argmax(axis=-1), eval_pred[0].argmax(axis=-1)
            )
        },
    )

    trainer.train()
    model = model.cuda().eval()
    return model, tokenizer
```

### Cell 6: Phase A — Full FT Baseline

```python
print("=" * 60)
print("PHASE A: Full Fine-Tuning BASELINE (no augmentation)")
print("=" * 60)
baseline_model, tokenizer = train_full_ft(
    ds["train"], ds["validation"], output_dir="trainer_output_12_baseline"
)
```

### Cell 7: Phase A — Baseline Evaluation

```python
# Validation
val_texts = ds["validation"]["text"]
val_labels = np.argmax(ds["validation"]["labels"], axis=1)

def run_inference(model, tokenizer, texts, batch_size=32):
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

val_preds = run_inference(baseline_model, tokenizer, val_texts)
baseline_val_acc = accuracy_score(val_labels, val_preds)
baseline_val_f1 = f1_score(val_labels, val_preds, average="macro")
print(f"Baseline validation — Accuracy: {baseline_val_acc:.4f}  Macro F1: {baseline_val_f1:.4f}")
print(classification_report(val_labels, val_preds, target_names=LABEL_NAMES))

# Error count for comparison
n_errors = int((val_preds != val_labels).sum())
print(f"Misclassified: {n_errors} / {len(val_texts)} ({n_errors/len(val_texts):.1%})")
```

### Cell 8: Phase A — FPB + Test Evaluation

```python
print("\n" + "=" * 60)
print("Full FT Baseline — FPB sentences_50agree")
print("=" * 60)
baseline_50 = evaluate_on_fpb(baseline_model, tokenizer, fpb_50)

print("\n" + "=" * 60)
print("Full FT Baseline — FPB sentences_allAgree")
print("=" * 60)
baseline_all = evaluate_on_fpb(baseline_model, tokenizer, fpb_all)

# Aggregated test set
test_texts = ds["test"]["text"]
test_labels = np.argmax(ds["test"]["labels"], axis=1)
baseline_test_preds = run_inference(baseline_model, tokenizer, test_texts)
baseline_test_acc = accuracy_score(test_labels, baseline_test_preds)
baseline_test_f1 = f1_score(test_labels, baseline_test_preds, average="macro")
print(f"\nBaseline Agg Test — Accuracy: {baseline_test_acc:.4f}  Macro F1: {baseline_test_f1:.4f}")

# Free memory
del baseline_model
gc.collect()
torch.cuda.empty_cache()
```

### Cell 9: Load VS-CoT Augmentation Data

The 410 VS-CoT samples embedded as a gzip+base64 blob (self-contained, no API calls).

```python
import base64, gzip, io

VS_DATA_B64 = "..."  # [same blob from NB02A — 25K chars]

vs_raw = gzip.decompress(base64.b64decode(VS_DATA_B64))
vs_df = pd.read_csv(io.BytesIO(vs_raw))
print(f"Loaded {len(vs_df)} VS-augmented samples")
print(f"\nBy label:\n{vs_df['label_name'].value_counts().to_string()}")
print(f"\nBy confusion type:\n{vs_df['confusion_type'].value_counts().to_string()}")
```

### Cell 10: Create Augmented Training Set

```python
aug_texts = vs_df["text"].tolist()
aug_labels_int = vs_df["label"].tolist()
aug_labels = [np.eye(NUM_CLASSES)[lbl].tolist() for lbl in aug_labels_int]

aug_ds = Dataset.from_dict({"text": aug_texts, "labels": aug_labels})
augmented_train = concatenate_datasets([ds["train"], aug_ds]).shuffle(seed=42)

print(f"Original train size:  {len(ds['train']):,}")
print(f"VS augmentation size: {len(aug_ds):,}")
print(f"Augmented train size: {len(augmented_train):,}")
print(f"Augmentation ratio:   {len(aug_ds)/len(ds['train']):.1%}")
```

### Cell 11: Phase B — Full FT + DataBoost

```python
print("=" * 60)
print("PHASE B: Full Fine-Tuning + DataBoost (VS augmentation)")
print("=" * 60)
boosted_model, tokenizer = train_full_ft(
    augmented_train, ds["validation"], output_dir="trainer_output_12_boosted"
)
```

### Cell 12: Phase B — DataBoosted Evaluation

```python
# Validation
boosted_val_preds = run_inference(boosted_model, tokenizer, val_texts)
boosted_val_acc = accuracy_score(val_labels, boosted_val_preds)
boosted_val_f1 = f1_score(val_labels, boosted_val_preds, average="macro")
print(f"DataBoosted validation — Accuracy: {boosted_val_acc:.4f}  Macro F1: {boosted_val_f1:.4f}")

# FPB
print("\n" + "=" * 60)
print("Full FT + DataBoost — FPB sentences_50agree")
print("=" * 60)
boosted_50 = evaluate_on_fpb(boosted_model, tokenizer, fpb_50)

print("\n" + "=" * 60)
print("Full FT + DataBoost — FPB sentences_allAgree")
print("=" * 60)
boosted_all = evaluate_on_fpb(boosted_model, tokenizer, fpb_all)

# Aggregated test set
boosted_test_preds = run_inference(boosted_model, tokenizer, test_texts)
boosted_test_acc = accuracy_score(test_labels, boosted_test_preds)
boosted_test_f1 = f1_score(test_labels, boosted_test_preds, average="macro")
print(f"\nDataBoosted Agg Test — Accuracy: {boosted_test_acc:.4f}  Macro F1: {boosted_test_f1:.4f}")

del boosted_model
gc.collect()
torch.cuda.empty_cache()
```

### Cell 13: 2x2 Comparison Matrix

The payoff cell — brings together all four configurations.

```python
print("=" * 80)
print("2×2 COMPARISON: LoRA vs Full FT × Base vs DataBoosted")
print("=" * 80)
print(f"{'Config':<40} {'FPB 50agree':>12} {'FPB allAgree':>12} {'Agg Test':>10}")
print("-" * 80)
# LoRA results from prior notebooks
print(f"{'LoRA r=16 — Baseline (NB01)':<40} {'80.44%':>12} {'92.98%':>12} {'77.71%':>10}")
print(f"{'LoRA r=16 — DataBoosted (NB02A)':<40} {'82.56%':>12} {'95.14%':>12} {'80.83%':>10}")
print(f"{'Full FT — Baseline (this exp)':<40} {baseline_50['accuracy']:>11.2%} {baseline_all['accuracy']:>12.2%} {baseline_test_acc:>9.2%}")
print(f"{'Full FT — DataBoosted (this exp)':<40} {boosted_50['accuracy']:>11.2%} {boosted_all['accuracy']:>12.2%} {boosted_test_acc:>9.2%}")
print()

# Deltas
ft_delta_50 = (boosted_50["accuracy"] - baseline_50["accuracy"]) * 100
lora_delta_50 = (0.8256 - 0.8044) * 100
ft_vs_lora_base = (baseline_50["accuracy"] - 0.8044) * 100
ft_vs_lora_boost = (boosted_50["accuracy"] - 0.8256) * 100

print(f"DataBoost delta (LoRA):    +{lora_delta_50:.2f}pp on FPB 50agree")
print(f"DataBoost delta (Full FT): {ft_delta_50:+.2f}pp on FPB 50agree")
print(f"Full FT vs LoRA (base):    {ft_vs_lora_base:+.2f}pp on FPB 50agree")
print(f"Full FT vs LoRA (boosted): {ft_vs_lora_boost:+.2f}pp on FPB 50agree")
```

### Cell 14: Save Results

```python
results_12 = {
    "baseline_50agree": baseline_50["accuracy"],
    "baseline_allagree": baseline_all["accuracy"],
    "baseline_50_f1": baseline_50["macro_f1"],
    "baseline_test_acc": baseline_test_acc,
    "baseline_test_f1": baseline_test_f1,
    "boosted_50agree": boosted_50["accuracy"],
    "boosted_allagree": boosted_all["accuracy"],
    "boosted_50_f1": boosted_50["macro_f1"],
    "boosted_test_acc": boosted_test_acc,
    "boosted_test_f1": boosted_test_f1,
    "vs_augmentation_samples": len(aug_ds),
    "training_samples": len(ds["train"]),
}
with open("results_12.json", "w") as f:
    json.dump(results_12, f, indent=2)
print("Saved to results_12.json")
```

### Cell 15: Confusion Matrices

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (res, title) in zip(axes, [
    (baseline_50, "Full FT Baseline"),
    (boosted_50, "Full FT + DataBoost"),
]):
    sns.heatmap(
        res["cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_title(f"{title}\nAcc={res['accuracy']:.2%}  F1={res['macro_f1']:.2%}")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

plt.suptitle("Exp 12: Full Fine-Tuning + DataBoost — FPB sentences_50agree", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("12_full_ft_databoost.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **Overfitting** (149M params on 8.6K samples) | lr=2e-5, wd=0.01, early stopping on val loss, gradient checkpointing |
| **GPU OOM** | gradient checkpointing, FP16, batch_size=8 with grad_accum=2 |
| **VS data mismatch** (mined from LoRA errors, not full FT) | Acceptable for fair comparison; fresh mining is a future experiment |
| **Full FT worse than LoRA** | This is a valid finding — would confirm LoRA's regularization benefit on small datasets |

## Expected Outcomes

Three possible results, all informative:

1. **Full FT + DataBoost > LoRA + DataBoost**: Full FT eliminates LoRA asymmetry AND DataBoost helps → new best configuration, update paper
2. **Full FT + DataBoost ≈ LoRA + DataBoost**: LoRA's regularization offsets its capacity limitation → LoRA is the better practical choice (cheaper, faster)
3. **Full FT + DataBoost < LoRA + DataBoost**: Full FT overfits on small data even with DataBoost → LoRA's implicit regularization is essential
