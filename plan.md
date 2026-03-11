# Controlled Experiments: BERT vs ModernBERT on Financial Sentiment

**Goal**: Rigorously confirm (or refute) that BERT genuinely outperforms ModernBERT on financial sentiment classification, with airtight controls for data leakage, LoRA capacity asymmetry, and dataset versioning.

**Current evidence**: On the same 13K training data, BERT achieves 95.19% vs ModernBERT's 91.19% on FPB `sentences_50agree` — a ~4pp gap. But several confounds remain uncontrolled. These experiments eliminate them one by one.

---

## Experiment A: Deduplication Audit

**Purpose**: Verify there is zero data leakage between training data and FPB test set. If BERT's advantage comes from memorizing near-duplicates, this experiment will expose it.

**What we check**:
1. Exact string matches (after lowercasing + stripping)
2. High-overlap fuzzy matches (>90% character similarity)
3. Semantic near-duplicates (cosine similarity > 0.95 using sentence embeddings)

```python
# ── Cell 1: Setup ──
import pandas as pd
import numpy as np
from datasets import load_dataset
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load training data (aggregated dataset, excluding FPB source_id=5)
ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
train_texts = [
    row['text'] for row in ds['train']
    if row.get('source_id') != 5
]
val_texts = [
    row['text'] for row in ds['validation']
    if row.get('source_id') != 5
]
all_train = train_texts + val_texts

# Load FPB test set
fpb = load_dataset("financial_phrasebank", "sentences_50agree")
fpb_texts = fpb['train']['sentence']  # FPB only has 'train' split

print(f"Training samples: {len(all_train)}")
print(f"FPB samples: {len(fpb_texts)}")
```

```python
# ── Cell 2: Exact match check ──
def normalize(text):
    return text.strip().lower()

train_norm = set(normalize(t) for t in all_train)
fpb_norm = [(i, normalize(t)) for i, t in enumerate(fpb_texts)]

exact_matches = [(i, t) for i, t in fpb_norm if t in train_norm]
print(f"Exact matches: {len(exact_matches)} / {len(fpb_texts)}")
for i, t in exact_matches[:10]:
    print(f"  [{i}] {t[:100]}...")
```

```python
# ── Cell 3: Fuzzy match check ──
def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Check each FPB sample against training data (slow but thorough)
# For speed, only check FPB samples against training samples of similar length
fuzzy_matches = []
for i, fpb_t in enumerate(fpb_norm):
    fpb_len = len(fpb_t[1])
    for train_t in train_norm:
        if abs(len(train_t) - fpb_len) > 20:
            continue  # skip length-mismatched pairs
        ratio = fuzzy_ratio(fpb_t[1], train_t)
        if ratio > 0.90:
            fuzzy_matches.append((i, fpb_t[1][:80], train_t[:80], ratio))
            break  # one match is enough

print(f"Fuzzy matches (>90%): {len(fuzzy_matches)}")
for idx, fpb_s, train_s, r in fuzzy_matches[:10]:
    print(f"  [{idx}] ratio={r:.3f}")
    print(f"    FPB:   {fpb_s}")
    print(f"    Train: {train_s}")
```

```python
# ── Cell 4: Semantic near-duplicate check ──
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode in batches
fpb_embs = model.encode(fpb_texts, batch_size=256, show_progress_bar=True)
train_embs = model.encode(all_train, batch_size=256, show_progress_bar=True)

# Compute pairwise cosine similarity (FPB x Train)
# For memory: process in chunks
THRESHOLD = 0.95
semantic_matches = []
chunk_size = 500
for start in range(0, len(fpb_embs), chunk_size):
    end = min(start + chunk_size, len(fpb_embs))
    sims = cosine_similarity(fpb_embs[start:end], train_embs)
    for i in range(end - start):
        max_sim = sims[i].max()
        if max_sim > THRESHOLD:
            best_j = sims[i].argmax()
            semantic_matches.append((
                start + i, best_j, max_sim,
                fpb_texts[start + i][:80],
                all_train[best_j][:80]
            ))

print(f"Semantic near-duplicates (cosine > {THRESHOLD}): {len(semantic_matches)}")
for fpb_i, train_j, sim, fpb_s, train_s in semantic_matches[:10]:
    print(f"  sim={sim:.4f}")
    print(f"    FPB:   {fpb_s}")
    print(f"    Train: {train_s}")
```

```python
# ── Cell 5: Summary ──
print("=" * 60)
print("DEDUPLICATION AUDIT SUMMARY")
print("=" * 60)
print(f"Training set size:           {len(all_train)}")
print(f"FPB test set size:           {len(fpb_texts)}")
print(f"Exact matches:               {len(exact_matches)}")
print(f"Fuzzy matches (>90%):        {len(fuzzy_matches)}")
print(f"Semantic near-dupes (>0.95): {len(semantic_matches)}")
print()
if len(exact_matches) + len(fuzzy_matches) + len(semantic_matches) == 0:
    print("CLEAN: No data leakage detected.")
else:
    print("WARNING: Potential leakage found. Remove these before continuing.")
```

**Expected outcome**: Zero or near-zero leakage (source_id=5 filtering should catch it). If leakage IS found, all previous results are suspect and we re-run after cleaning.

---

## Experiment B: FPB-Only 10-Fold Cross-Validation

**Purpose**: Eliminate the training data variable entirely. Train both models on *only FPB data* using identical 10-fold CV splits. If BERT still wins, the gap is real and comes from pre-training/architecture, not training data composition.

**Why this matters**: Our current comparison uses the aggregated dataset (many sources). Maybe ModernBERT handles FPB-style text fine but struggles with the mixed-source training signal. This test isolates FPB-only performance.

```python
# ── Cell 1: Setup ──
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

SEED = 42
NUM_FOLDS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-4
MAX_LEN = 128

# Load FPB
fpb = load_dataset("financial_phrasebank", "sentences_50agree")
texts = fpb['train']['sentence']
labels = fpb['train']['label']

print(f"FPB total: {len(texts)} samples")
print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
```

```python
# ── Cell 2: Define model configs ──
MODELS = {
    "bert-base": {
        "name": "bert-base-uncased",
        "lora_targets": ["query", "key", "value", "dense"],
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "modernbert-base": {
        "name": "answerdotai/ModernBERT-base",
        "lora_targets": ["Wqkv", "out_proj", "Wi", "Wo"],
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "modernbert-base-r48": {
        "name": "answerdotai/ModernBERT-base",
        "lora_targets": ["Wqkv", "out_proj", "Wi", "Wo"],
        "lora_r": 48,
        "lora_alpha": 96,
    },
}
```

```python
# ── Cell 3: Training function ──
def train_and_evaluate(model_key, train_idx, val_idx, fold_num):
    cfg = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["name"], num_labels=3
    )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_targets"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)

    # Tokenize
    train_texts_fold = [texts[i] for i in train_idx]
    train_labels_fold = [labels[i] for i in train_idx]
    val_texts_fold = [texts[i] for i in val_idx]
    val_labels_fold = [labels[i] for i in val_idx]

    train_enc = tokenizer(train_texts_fold, truncation=True,
                          padding=True, max_length=MAX_LEN)
    val_enc = tokenizer(val_texts_fold, truncation=True,
                        padding=True, max_length=MAX_LEN)

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_ds = SentimentDataset(train_enc, train_labels_fold)
    val_ds = SentimentDataset(val_enc, val_labels_fold)

    args = TrainingArguments(
        output_dir=f"cv_output/{model_key}/fold_{fold_num}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        acc = accuracy_score(eval_pred.label_ids, preds)
        f1 = f1_score(eval_pred.label_ids, preds, average="macro")
        return {"accuracy": acc, "macro_f1": f1}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    preds = trainer.predict(val_ds)
    pred_labels = np.argmax(preds.predictions, axis=-1)

    report = classification_report(
        val_labels_fold, pred_labels,
        target_names=["negative", "neutral", "positive"],
        output_dict=True
    )

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return {
        "fold": fold_num,
        "model": model_key,
        "accuracy": results["eval_accuracy"],
        "macro_f1": results["eval_macro_f1"],
        "eval_loss": results["eval_loss"],
        "per_class": report,
    }
```

```python
# ── Cell 4: Run 10-fold CV ──
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_results = []

for model_key in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_key}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        result = train_and_evaluate(model_key, train_idx, val_idx, fold)
        all_results.append(result)
        print(f"  Acc: {result['accuracy']:.4f}, F1: {result['macro_f1']:.4f}")

# Save all results
with open("cv_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
```

```python
# ── Cell 5: Summary statistics ──
df = pd.DataFrame(all_results)

summary = df.groupby("model").agg(
    mean_acc=("accuracy", "mean"),
    std_acc=("accuracy", "std"),
    mean_f1=("macro_f1", "mean"),
    std_f1=("macro_f1", "std"),
).round(4)

print("\n10-FOLD CV RESULTS (FPB sentences_50agree)")
print("=" * 70)
print(summary.to_string())
print()

# Paired t-test: BERT vs ModernBERT r=16
from scipy import stats
bert_accs = df[df.model == "bert-base"]["accuracy"].values
mb_accs = df[df.model == "modernbert-base"]["accuracy"].values
t_stat, p_val = stats.ttest_rel(bert_accs, mb_accs)
print(f"Paired t-test (BERT vs ModernBERT r=16): t={t_stat:.3f}, p={p_val:.4f}")

mb48_accs = df[df.model == "modernbert-base-r48"]["accuracy"].values
t_stat2, p_val2 = stats.ttest_rel(bert_accs, mb48_accs)
print(f"Paired t-test (BERT vs ModernBERT r=48): t={t_stat2:.3f}, p={p_val2:.4f}")
```

**Expected outcome**: If BERT consistently wins across all 10 folds with p < 0.05, the gap is real and statistically significant.

---

## Experiment C: Clean Held-Out Evaluation (Post-Dedup)

**Purpose**: Re-run the standard aggregated-data experiment after removing any near-duplicates found in Experiment A. Uses the same protocol as NB01/NB05 but with verified-clean training data.

```python
# ── Cell 1: Build clean dataset ──
# (Run after Experiment A)
# Remove any FPB near-duplicates from training data

# Assume `contaminated_indices` is a set of training indices
# identified in Experiment A
contaminated_indices = set()  # populate from Exp A results

ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
clean_train = [
    row for i, row in enumerate(ds['train'])
    if row.get('source_id') != 5 and i not in contaminated_indices
]
clean_val = [
    row for i, row in enumerate(ds['validation'])
    if row.get('source_id') != 5 and i not in contaminated_indices
]

print(f"Original train: {len(ds['train'])}")
print(f"Clean train (no FPB, no dupes): {len(clean_train)}")
print(f"Removed: {len(ds['train']) - len(clean_train)}")
```

```python
# ── Cell 2: Train both models on clean data, evaluate on FPB ──
# Same training protocol as NB01/NB05:
#   - LoRA r=16, alpha=32, dropout=0.05
#   - 10 epochs, cosine LR, warmup 5%
#   - load_best_model_at_end (by eval_loss)
#   - batch_size=16, lr=2e-4

# BERT config
bert_lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.05, bias="none",
    task_type=TaskType.SEQ_CLS,
)

# ModernBERT config
modernbert_lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
    lora_dropout=0.05, bias="none",
    task_type=TaskType.SEQ_CLS,
)

# ... (same training loop as NB05, just with clean_train/clean_val)
# Evaluate on FPB sentences_50agree AND sentences_allAgree
```

**Expected outcome**: Results should be very close to the original NB01/NB05 numbers (since we expect minimal leakage). This confirms the gap is not an artifact of contamination.

---

## Experiment D: Sample Efficiency Curves

**Purpose**: Map how both models scale with training data. If ModernBERT simply needs more data to catch up, we'll see converging curves. If the gap persists at all scales, pre-training alignment is the cause.

```python
# ── Cell 1: Setup ──
SAMPLE_SIZES = [500, 1000, 2000, 4000, 8000, 13000]
MODELS_TO_TEST = ["bert-base-uncased", "answerdotai/ModernBERT-base"]
SEEDS = [42, 123, 456]  # 3 seeds per size for error bars

results = []
```

```python
# ── Cell 2: Training loop ──
for model_name in MODELS_TO_TEST:
    for n_samples in SAMPLE_SIZES:
        for seed in SEEDS:
            # Subsample training data (stratified)
            np.random.seed(seed)
            train_labels = [row['label'] for row in clean_train]

            # Stratified subsample
            from sklearn.model_selection import train_test_split
            if n_samples >= len(clean_train):
                subset = clean_train
            else:
                subset, _ = train_test_split(
                    clean_train,
                    train_size=n_samples,
                    stratify=train_labels,
                    random_state=seed,
                )

            print(f"\n{model_name} | n={len(subset)} | seed={seed}")

            # Configure LoRA based on model
            if "ModernBERT" in model_name:
                targets = ["Wqkv", "out_proj", "Wi", "Wo"]
            else:
                targets = ["query", "key", "value", "dense"]

            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=targets,
                lora_dropout=0.05, bias="none",
                task_type=TaskType.SEQ_CLS,
            )

            # ... (standard training, evaluate on FPB)
            # Record: model, n_samples, seed, fpb_50agree_acc, fpb_allagree_acc

            result = {
                "model": model_name,
                "n_samples": len(subset),
                "seed": seed,
                "fpb_50agree_acc": None,  # fill after eval
                "fpb_allagree_acc": None,
            }
            results.append(result)
```

```python
# ── Cell 3: Plot scaling curves ──
import matplotlib.pyplot as plt

df = pd.DataFrame(results)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for model_name in MODELS_TO_TEST:
    subset = df[df.model == model_name]
    grouped = subset.groupby("n_samples")["fpb_50agree_acc"]
    means = grouped.mean()
    stds = grouped.std()

    label = "BERT-base" if "bert" in model_name.lower() else "ModernBERT-base"
    ax.errorbar(means.index, means.values, yerr=stds.values,
                label=label, marker='o', capsize=4)

ax.set_xlabel("Training Samples")
ax.set_ylabel("FPB 50agree Accuracy (%)")
ax.set_title("Sample Efficiency: BERT vs ModernBERT")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig("sample_efficiency_curves.png", dpi=150)
plt.show()

# Print the gap at each scale
print("\nGap (BERT - ModernBERT) at each scale:")
for n in SAMPLE_SIZES:
    bert_mean = df[(df.model.str.contains("bert-base")) &
                   (df.n_samples == n)]["fpb_50agree_acc"].mean()
    mb_mean = df[(df.model.str.contains("ModernBERT")) &
                 (df.n_samples == n)]["fpb_50agree_acc"].mean()
    print(f"  n={n:>6d}: BERT={bert_mean:.2%}, MB={mb_mean:.2%}, gap={bert_mean-mb_mean:+.2%}")
```

**Key question this answers**: Does the gap shrink as data increases? If so, ModernBERT just needs more data. If the gap is constant (or widens), pre-training alignment is the fundamental issue.

---

## Experiment E: Full Fine-Tuning (No LoRA)

**Purpose**: Eliminate the LoRA capacity / fused-QKV asymmetry entirely. If ModernBERT catches up under full fine-tuning, the gap was a LoRA artifact. If BERT still wins, the gap is genuine.

**Risk**: Overfitting is much more likely with full fine-tuning on small data. We mitigate with early stopping, weight decay, and lower learning rate.

**Kaggle feasibility**: This is comfortably within Kaggle's limits. BERT-base (110M params) and ModernBERT-base (149M params) both fit in a T4's 16GB VRAM with fp16 at batch_size=16. Training on 13K samples for 10 epochs takes ~30-45 min per model (~1.5 hrs total). Kaggle allows 30 hrs/week of T4 time. If memory is tight, reduce batch_size to 8 with `gradient_accumulation_steps=2` for identical training dynamics.

```python
# ── Cell 1: Full fine-tuning setup ──
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def full_finetune(model_name, train_dataset, val_dataset, output_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )

    # NO LoRA — update all parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,              # Lower LR for full FT
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,               # More warmup
        weight_decay=0.01,              # Regularization
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=42,
        report_to="none",
        fp16=True,
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        acc = accuracy_score(eval_pred.label_ids, preds)
        f1 = f1_score(eval_pred.label_ids, preds, average="macro")
        return {"accuracy": acc, "macro_f1": f1}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer
```

```python
# ── Cell 2: Run for both models ──
models_to_test = {
    "bert-base": "bert-base-uncased",
    "modernbert-base": "answerdotai/ModernBERT-base",
}

full_ft_results = {}
for key, name in models_to_test.items():
    print(f"\n{'='*60}")
    print(f"FULL FINE-TUNING: {key}")
    print(f"{'='*60}")

    trainer = full_finetune(
        model_name=name,
        train_dataset=train_ds,  # same clean dataset for both
        val_dataset=val_ds,
        output_dir=f"full_ft_output/{key}",
    )

    # Evaluate on FPB
    # ... (same FPB evaluation as other experiments)
    full_ft_results[key] = {
        "val_acc": None,
        "fpb_50agree_acc": None,
        "fpb_allagree_acc": None,
        "trainable_params": None,
    }
```

```python
# ── Cell 3: Compare LoRA vs Full FT ──
print("\nLoRA vs Full Fine-Tuning Comparison")
print("=" * 70)
print(f"{'Config':<30} {'Params':>10} {'FPB 50agree':>12} {'FPB allAgree':>12}")
print("-" * 70)
# Fill in from results:
# BERT + LoRA r=16
# BERT + Full FT
# ModernBERT + LoRA r=16
# ModernBERT + LoRA r=48
# ModernBERT + Full FT
```

**Key question**: Does full fine-tuning close the gap? If ModernBERT + Full FT matches or beats BERT + Full FT, then LoRA is the bottleneck. If BERT still wins, it's the pre-training.

---

## Experiment Execution Plan

### Priority Order

| # | Experiment | Runtime | Kaggle GPUs | Blocking? |
|---|-----------|---------|-------------|-----------|
| 1 | **A: Dedup Audit** | ~30 min | None (CPU) | Yes — must run first |
| 2 | **B: FPB 10-Fold CV** | ~8-10 hrs | 2x T4/P100 | No |
| 3 | **C: Clean Held-Out** | ~2-3 hrs | 1x T4/P100 | Depends on A |
| 4 | **D: Sample Efficiency** | ~12-15 hrs | 2x T4/P100 | No |
| 5 | **E: Full Fine-Tuning** | ~1.5 hrs | 1x T4 (16GB, fp16) | No |

### Parallelization

- **Run A first** (local, fast). If clean → proceed with B/C/D/E in parallel.
- **B and D** are the most informative. Run these on Kaggle simultaneously.
- **E** can run after B finishes (reuse the same notebook with modifications).
- **C** only matters if A finds contamination.

### Expected Decision Matrix

| If A finds leakage | If B shows BERT wins | If D shows gap constant | If E closes the gap | Conclusion |
|----|----|----|----|----|
| Yes | — | — | — | Previous results invalid. Re-run everything after cleaning. |
| No | Yes (p<0.05) | Yes | No | **Pre-training alignment is the cause.** BERT's Wikipedia training genuinely better for FPB. |
| No | Yes (p<0.05) | Yes | Yes | **LoRA asymmetry is the main cause.** Full FT eliminates it. |
| No | Yes (p<0.05) | No (gap shrinks) | — | **Data scale is the cause.** ModernBERT needs more data to compensate for misaligned pre-training. |
| No | No (p>0.05) | — | — | **Gap is not real.** Previous results were noise from small test set / data versioning. |

### Implementation Notes

1. **All experiments use the same evaluation**:
   - FPB `sentences_50agree` (4,846 samples, harder)
   - FPB `sentences_allAgree` (2,264 samples, easier)
   - Report accuracy AND macro F1

2. **Control variables**:
   - Same tokenizer max_length (128)
   - Same batch size (16)
   - Same optimizer (AdamW)
   - Same scheduler (cosine with warmup)
   - Same number of epochs (10)
   - Same early stopping criterion (best eval_loss)
   - Same random seed (42, with multiple seeds where noted)

3. **LoRA fairness for Experiment B**:
   - BERT r=16 on [query, key, value, dense]: ~0.89M trainable params
   - ModernBERT r=16 on [Wqkv, out_proj, Wi, Wo]: ~1.1M trainable params
   - ModernBERT r=48 on same: ~3.3M trainable params
   - Running all three controls for parameter count

4. **Notebook structure**: One notebook per experiment (NB09-A through NB09-E), each self-contained and runnable on Kaggle with `kaggle_push_09x/` configs.

5. **Kaggle resource budget**: All experiments combined need ~25 hrs of GPU time. Kaggle gives 30 hrs/week of T4. Plan to run B+D across two weeks if needed, or split D into two notebooks (3 sample sizes each). Experiment E is the lightest at ~1.5 hrs — both models are small enough (110M/149M params) for full fine-tuning on a single T4 with fp16. If VRAM is tight, use `gradient_accumulation_steps=2` with halved batch size.

---

## TODO

### Phase 1: Setup & Dedup Audit (Experiment A)

- [ ] **1.1** Create notebook `notebooks/09a_dedup_audit.ipynb`
- [ ] **1.2** Load aggregated dataset and FPB, filter out source_id=5
- [ ] **1.3** Implement exact-match check (normalize + set lookup)
- [ ] **1.4** Implement fuzzy-match check (SequenceMatcher, >90% threshold)
- [ ] **1.5** Implement semantic near-duplicate check (sentence-transformers, cosine >0.95)
- [ ] **1.6** Generate dedup audit summary (counts for each category)
- [ ] **1.7** If contamination found: build `contaminated_indices` set and save to JSON
- [ ] **1.8** Run notebook locally (CPU-only, ~30 min)
- [ ] **1.9** Record results in `research.md` — proceed to Phase 2 if clean, or to Phase 1b if not

**Gate**: Phase 2+ cannot start until 1.8 completes. If leakage is found, all subsequent experiments use the cleaned dataset.

### Phase 1b: Clean Dataset (only if Experiment A finds leakage)

- [ ] **1b.1** Remove contaminated samples from training/validation sets
- [ ] **1b.2** Verify removal didn't break label distribution balance
- [ ] **1b.3** Save cleaned dataset to local disk or re-upload to HuggingFace
- [ ] **1b.4** Update all downstream notebooks to load cleaned version

### Phase 2: FPB Head-to-Head (Experiment B)

- [ ] **2.1** Create notebook `notebooks/09b_fpb_crossval.ipynb`
- [ ] **2.2** Load FPB `sentences_50agree`, set up 10-fold stratified CV splits (seed=42)
- [ ] **2.3** Implement `train_and_evaluate()` function with GPU memory cleanup
- [ ] **2.4** Define 3 model configs: BERT r=16, ModernBERT r=16, ModernBERT r=48
- [ ] **2.5** Run all 30 training runs (3 models x 10 folds)
- [ ] **2.6** Compute summary statistics (mean/std accuracy and macro F1 per model)
- [ ] **2.7** Run paired t-tests: BERT vs ModernBERT r=16, BERT vs ModernBERT r=48
- [ ] **2.8** Save full results to `cv_results.json`
- [ ] **2.9** Create Kaggle push config `kaggle_push_09b/kernel-metadata.json`
- [ ] **2.10** Push to Kaggle, run, download outputs
- [ ] **2.11** Record results and p-values in `research.md`

### Phase 3: Clean Held-Out Evaluation (Experiment C)

- [ ] **3.1** Create notebook `notebooks/09c_clean_holdout.ipynb`
- [ ] **3.2** Load clean training data (post-dedup from Phase 1 or original if clean)
- [ ] **3.3** Train BERT-base + LoRA r=16 on clean data, evaluate on FPB
- [ ] **3.4** Train ModernBERT-base + LoRA r=16 on clean data, evaluate on FPB
- [ ] **3.5** Compare results to original NB01/NB05 numbers — confirm gap is not from contamination
- [ ] **3.6** Create Kaggle push config `kaggle_push_09c/kernel-metadata.json`
- [ ] **3.7** Push to Kaggle, run, download outputs
- [ ] **3.8** Record results in `research.md`

### Phase 4: Sample Efficiency Curves (Experiment D)

- [ ] **4.1** Create notebook `notebooks/09d_sample_efficiency.ipynb`
- [ ] **4.2** Implement stratified subsampling at 6 sizes: 500, 1K, 2K, 4K, 8K, 13K
- [ ] **4.3** Train BERT-base + LoRA r=16 at each size x 3 seeds (18 runs)
- [ ] **4.4** Train ModernBERT-base + LoRA r=16 at each size x 3 seeds (18 runs)
- [ ] **4.5** Evaluate all 36 runs on FPB `sentences_50agree` and `sentences_allAgree`
- [ ] **4.6** Plot scaling curves with error bars (log-x, accuracy-y)
- [ ] **4.7** Print gap at each scale to determine if it converges, stays constant, or diverges
- [ ] **4.8** Create Kaggle push config `kaggle_push_09d/kernel-metadata.json`
- [ ] **4.9** If runtime exceeds Kaggle limits: split into two notebooks (09d1: 500-2K, 09d2: 4K-13K)
- [ ] **4.10** Push to Kaggle, run, download outputs
- [ ] **4.11** Record scaling analysis in `research.md`

### Phase 5: Full Fine-Tuning (Experiment E)

- [ ] **5.1** Create notebook `notebooks/09e_full_finetune.ipynb`
- [ ] **5.2** Implement `full_finetune()` — no LoRA, all params trainable, lr=2e-5, weight_decay=0.01, fp16
- [ ] **5.3** Add gradient_accumulation_steps=2 fallback if OOM at batch_size=16
- [ ] **5.4** Train BERT-base (110M params) on clean data, evaluate on FPB
- [ ] **5.5** Train ModernBERT-base (149M params) on clean data, evaluate on FPB
- [ ] **5.6** Build comparison table: LoRA r=16, LoRA r=48, Full FT for both models
- [ ] **5.7** Create Kaggle push config `kaggle_push_09e/kernel-metadata.json`
- [ ] **5.8** Push to Kaggle, run, download outputs
- [ ] **5.9** Record results in `research.md`

### Phase 6: Analysis & Write-Up

- [ ] **6.1** Collect all results into a single summary table
- [ ] **6.2** Apply the decision matrix to determine root cause
- [ ] **6.3** Update `research.md` with final conclusions
- [ ] **6.4** Update `paper/main.tex` with new experiment results (if findings change the narrative)
- [ ] **6.5** Commit all notebooks, outputs, and updated documents