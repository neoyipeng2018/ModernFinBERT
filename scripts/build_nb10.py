"""Generate notebook 10_finbert_tone_deep_dive.ipynb from code cells."""
import json

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().split("\n") if isinstance(source, str) else source,
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n") if isinstance(source, str) else source,
    }

# Fix source lines to have newlines between them
def fix_source(cell):
    src = cell["source"]
    cell["source"] = [line + "\n" if i < len(src) - 1 else line for i, line in enumerate(src)]
    return cell

cells = []

# ── Cell 1: Imports ──
cells.append(md_cell("# Notebook 10: finbert-tone vs ModernFinBERT — Deep Dive & Gap Widening"))
cells.append(code_cell("""import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from huggingface_hub import hf_hub_download
import os
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_CLASSES = 3
FPB_LABEL_MAP_STR = {"negative": 0, "neutral": 1, "positive": 2}

def load_fpb_from_zip(filename):
    zip_path = hf_hub_download("financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    texts, labels = [], []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"FinancialPhraseBank-v1.0/{filename}") as f:
            for line in f:
                line = line.decode("latin-1").strip()
                if not line:
                    continue
                sep_idx = line.rfind("@")
                if sep_idx == -1:
                    continue
                text = line[:sep_idx].strip()
                label_str = line[sep_idx + 1:].strip().lower()
                if label_str in FPB_LABEL_MAP_STR:
                    texts.append(text)
                    labels.append(FPB_LABEL_MAP_STR[label_str])
    return texts, np.array(labels)

os.makedirs("results", exist_ok=True)"""))

# ── Cell 2: Load models ──
cells.append(md_cell("## Part A: Load Models & Run Inference"))
cells.append(code_cell("""ft_name = "yiyanghkust/finbert-tone"
ft_tokenizer = BertTokenizer.from_pretrained(ft_name)
ft_model = BertForSequenceClassification.from_pretrained(ft_name).to(device).eval()
print(f"finbert-tone id2label: {ft_model.config.id2label}")

mfb_name = "neoyipeng/ModernFinBERT-base"
mfb_tokenizer = AutoTokenizer.from_pretrained(mfb_name)
mfb_model = AutoModelForSequenceClassification.from_pretrained(mfb_name).to(device).eval()
print(f"ModernFinBERT id2label: {mfb_model.config.id2label}")"""))

# ── Cell 3: Load FPB data ──
cells.append(code_cell("""fpb_texts, fpb_labels = load_fpb_from_zip("Sentences_50Agree.txt")
fpb_all_texts, fpb_all_labels = load_fpb_from_zip("Sentences_AllAgree.txt")

print(f"FPB 50agree: {len(fpb_texts)} samples — {Counter(fpb_labels.tolist())}")
print(f"FPB allAgree: {len(fpb_all_texts)} samples — {Counter(fpb_all_labels.tolist())}")"""))

# ── Cell 4: predict_batch ──
cells.append(code_cell("""def predict_batch(texts, model, tokenizer, dev, label_map, batch_size=64, max_length=512):
    all_preds = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            mapped_preds = [label_map[int(p)] for p in pred_ids]
            all_preds.extend(mapped_preds)
            all_probs.append(probs)
    return np.array(all_preds), np.vstack(all_probs)"""))

# ── Cell 5: Build label maps + run inference ──
cells.append(code_cell("""# Build label maps from model configs
_ft_name_to_fpb = {"positive": 2, "negative": 0, "neutral": 1}
ft_label_map = {int(k): _ft_name_to_fpb[v.lower()] for k, v in ft_model.config.id2label.items()}
print(f"finbert-tone label_map: {ft_label_map}")

mfb_label_map = {0: 0, 1: 1, 2: 2}

print("\\nRunning inference on FPB 50agree...")
ft_preds_50, ft_probs_50 = predict_batch(fpb_texts, ft_model, ft_tokenizer, device, ft_label_map)
mfb_preds_50, mfb_probs_50 = predict_batch(fpb_texts, mfb_model, mfb_tokenizer, device, mfb_label_map)

print("Running inference on FPB allAgree...")
ft_preds_all, ft_probs_all = predict_batch(fpb_all_texts, ft_model, ft_tokenizer, device, ft_label_map)
mfb_preds_all, mfb_probs_all = predict_batch(fpb_all_texts, mfb_model, mfb_tokenizer, device, mfb_label_map)
print("Done.")"""))

# ── Cell 6: Metrics ──
cells.append(code_cell("""def full_report(y_true, y_pred, model_name, label_names=LABEL_NAMES):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\\n{'='*60}")
    print(f"{model_name}: Accuracy={acc:.4f}, Macro F1={f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    return acc, f1

print("=" * 70)
print("FPB sentences_50agree (4,846 samples)")
print("=" * 70)
ft_acc_50, ft_f1_50 = full_report(fpb_labels, ft_preds_50, "finbert-tone (zero-shot)")
mfb_acc_50, mfb_f1_50 = full_report(fpb_labels, mfb_preds_50, "ModernFinBERT (held-out)")
print(f"\\n>>> Gap (MFB - FT): Acc={mfb_acc_50 - ft_acc_50:+.4f}, F1={mfb_f1_50 - ft_f1_50:+.4f}")

print("\\n" + "=" * 70)
print("FPB sentences_allAgree (2,264 samples)")
print("=" * 70)
ft_acc_all, ft_f1_all = full_report(fpb_all_labels, ft_preds_all, "finbert-tone (zero-shot)")
mfb_acc_all, mfb_f1_all = full_report(fpb_all_labels, mfb_preds_all, "ModernFinBERT (held-out)")
print(f"\\n>>> Gap (MFB - FT): Acc={mfb_acc_all - ft_acc_all:+.4f}, F1={mfb_f1_all - ft_f1_all:+.4f}")"""))

# ── Cell 7: Confusion matrices ──
cells.append(md_cell("## Part B: Deep Error Analysis"))
cells.append(code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, name in [(axes[0], ft_preds_50, "finbert-tone"), (axes[1], mfb_preds_50, "ModernFinBERT")]:
    cm = confusion_matrix(fpb_labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{name}")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("results/confusion_ft_vs_mfb.png", dpi=150, bbox_inches="tight")
plt.show()"""))

# ── Cell 8: Sentence-level agreement ──
cells.append(code_cell("""df = pd.DataFrame({
    "text": fpb_texts,
    "label": fpb_labels,
    "ft_pred": ft_preds_50,
    "mfb_pred": mfb_preds_50,
    "ft_correct": ft_preds_50 == fpb_labels,
    "mfb_correct": mfb_preds_50 == fpb_labels,
})

df["category"] = "both_wrong"
df.loc[df["ft_correct"] & df["mfb_correct"], "category"] = "both_right"
df.loc[df["ft_correct"] & ~df["mfb_correct"], "category"] = "ft_only"
df.loc[~df["ft_correct"] & df["mfb_correct"], "category"] = "mfb_only"

counts = df["category"].value_counts()
print("Sentence-level agreement:")
print(f"  both_right:  {counts.get('both_right', 0)} — both models correct")
print(f"  both_wrong:  {counts.get('both_wrong', 0)} — neither model correct")
print(f"  ft_only:     {counts.get('ft_only', 0)} — finbert-tone right, MFB wrong")
print(f"  mfb_only:    {counts.get('mfb_only', 0)} — MFB right, finbert-tone wrong")
print(f"\\nMFB unique advantage: {counts.get('mfb_only', 0) - counts.get('ft_only', 0):+d} sentences")"""))

# ── Cell 9: Per-class + text length ──
cells.append(code_cell("""print("Per-class accuracy:")
for cls_idx, cls_name in enumerate(LABEL_NAMES):
    mask = fpb_labels == cls_idx
    ft_cls_acc = (ft_preds_50[mask] == fpb_labels[mask]).mean()
    mfb_cls_acc = (mfb_preds_50[mask] == fpb_labels[mask]).mean()
    n = mask.sum()
    print(f"  {cls_name:>10} (n={n:4d}): FT={ft_cls_acc:.4f}  MFB={mfb_cls_acc:.4f}  delta={mfb_cls_acc - ft_cls_acc:+.4f}")

df["word_count"] = df["text"].str.split().str.len()
df["length_bin"] = pd.cut(df["word_count"], bins=[0, 15, 25, 40, 999], labels=["short", "medium", "long", "very_long"])

print("\\nAccuracy by text length:")
for bin_name in ["short", "medium", "long", "very_long"]:
    mask = df["length_bin"] == bin_name
    if mask.sum() == 0:
        continue
    ft_acc = df.loc[mask, "ft_correct"].mean()
    mfb_acc = df.loc[mask, "mfb_correct"].mean()
    n = mask.sum()
    print(f"  {bin_name:>10} (n={n:4d}): FT={ft_acc:.4f}  MFB={mfb_acc:.4f}  delta={mfb_acc - ft_acc:+.4f}")"""))

# ── Cell 10: Confidence analysis ──
cells.append(code_cell("""ft_label_map_reverse = {v: k for k, v in ft_label_map.items()}
ft_probs_fpb_order = np.column_stack([
    ft_probs_50[:, ft_label_map_reverse[0]],
    ft_probs_50[:, ft_label_map_reverse[1]],
    ft_probs_50[:, ft_label_map_reverse[2]],
])
mfb_probs_fpb_order = mfb_probs_50

df["ft_confidence"] = ft_probs_fpb_order.max(axis=1).astype(np.float64)
df["mfb_confidence"] = mfb_probs_fpb_order.max(axis=1).astype(np.float64)

for model_name, conf_col, correct_col in [
    ("finbert-tone", "ft_confidence", "ft_correct"),
    ("ModernFinBERT", "mfb_confidence", "mfb_correct"),
]:
    print(f"\\n{model_name} — accuracy by confidence quartile:")
    try:
        df["_q"] = pd.qcut(df[conf_col], 4, labels=False, duplicates="drop")
        n_bins = df["_q"].nunique()
        bin_labels = [f"Q{i+1}" for i in range(n_bins)]
        df["_q"] = df["_q"].map(dict(enumerate(bin_labels)))
        for q in bin_labels:
            mask = df["_q"] == q
            if mask.sum() == 0:
                continue
            acc = df.loc[mask, correct_col].mean()
            mean_conf = df.loc[mask, conf_col].mean()
            print(f"  {q}: acc={acc:.4f}, mean_conf={mean_conf:.4f}, n={mask.sum()}")
    except ValueError:
        acc = df[correct_col].mean()
        mean_conf = df[conf_col].mean()
        print(f"  (all): acc={acc:.4f}, mean_conf={mean_conf:.4f}, n={len(df)} — confidence too concentrated")"""))

# ── Cell 11: Qualitative examples ──
cells.append(code_cell("""print("=" * 80)
print("finbert-tone CORRECT, ModernFinBERT WRONG:")
print("=" * 80)
ft_only_df = df[df["category"] == "ft_only"]
for _, row in ft_only_df.sample(min(10, len(ft_only_df)), random_state=42).iterrows():
    print(f"\\n  Text: {row['text'][:120]}...")
    print(f"  True: {LABEL_NAMES[row['label']]} | FT: {LABEL_NAMES[row['ft_pred']]} ok | MFB: {LABEL_NAMES[row['mfb_pred']]} X")

print("\\n" + "=" * 80)
print("ModernFinBERT CORRECT, finbert-tone WRONG:")
print("=" * 80)
mfb_only_df = df[df["category"] == "mfb_only"]
for _, row in mfb_only_df.sample(min(10, len(mfb_only_df)), random_state=42).iterrows():
    print(f"\\n  Text: {row['text'][:120]}...")
    print(f"  True: {LABEL_NAMES[row['label']]} | FT: {LABEL_NAMES[row['ft_pred']]} X | MFB: {LABEL_NAMES[row['mfb_pred']]} ok")"""))

# ── Cell 12: Shared training infrastructure ──
cells.append(md_cell("## Part C: Techniques to Widen the Gap\n\nAll techniques train from scratch on the aggregated dataset (FPB excluded) and evaluate on FPB `sentences_50agree`."))
cells.append(code_cell("""from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch.nn.functional as F

SEED = 3407
MAX_LENGTH = 512
LABEL_DICT = {"NEGATIVE": 0, "NEUTRAL/MIXED": 1, "POSITIVE": 2}

def load_aggregated_data(exclude_fpb=True):
    ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
    def process_split(split):
        texts, labels = [], []
        for item in split:
            if item["task"] != "sentiment":
                continue
            if exclude_fpb and item.get("source") == 5:
                continue
            texts.append(item["text"])
            labels.append(LABEL_DICT[item["label"]])
        return texts, labels
    train_texts, train_labels = process_split(ds["train"])
    val_split = "dev" if "dev" in ds else "validation"
    val_texts, val_labels = process_split(ds[val_split])
    return train_texts, train_labels, val_texts, val_labels

def make_hf_dataset(texts, labels, tokenizer, max_length=MAX_LENGTH):
    from datasets import Dataset as HFDataset
    ds = HFDataset.from_dict({"text": texts, "label": labels})
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_length), batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def get_base_training_args(output_dir, **overrides):
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        fp16=True,
        learning_rate=2e-4,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=SEED,
        num_train_epochs=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=2,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def eval_on_fpb(model, tokenizer, label_map=None):
    if label_map is None:
        label_map = {0: 0, 1: 1, 2: 2}
    preds, probs = predict_batch(fpb_texts, model, tokenizer, device, label_map)
    acc = accuracy_score(fpb_labels, preds)
    f1 = f1_score(fpb_labels, preds, average="macro")
    report = classification_report(fpb_labels, preds, target_names=LABEL_NAMES, digits=4, output_dict=True)
    print(f"  FPB 50agree — Acc: {acc:.4f}, Macro F1: {f1:.4f}")
    return {"accuracy": acc, "macro_f1": f1, "report": report}

train_texts, train_labels, val_texts, val_labels = load_aggregated_data()
print(f"Training data: {len(train_texts)} samples")
print(f"Validation data: {len(val_texts)} samples")
print(f"Label distribution: {Counter(train_labels)}")"""))

# ── Cell 13: Technique 1 — LoRA r=64 ──
cells.append(md_cell("### Technique 1: Higher LoRA rank (r=64)"))
cells.append(code_cell("""def train_lora_variant(lora_r, lora_alpha=None, output_dir="output_r64"):
    if lora_alpha is None:
        lora_alpha = lora_r * 2
    base_model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=NUM_CLASSES)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train_ds = make_hf_dataset(train_texts, train_labels, tokenizer)
    val_ds = make_hf_dataset(val_texts, val_labels, tokenizer)

    args = get_base_training_args(output_dir)
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=args, compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    return model, tokenizer

print("Training LoRA r=64...")
model_r64, tok_r64 = train_lora_variant(lora_r=64, output_dir="output_r64")
print("\\nEvaluating on FPB:")
results_r64 = eval_on_fpb(model_r64, tok_r64)"""))

# ── Cell 14: Technique 2 — Label smoothing ──
cells.append(md_cell("### Technique 2: Label Smoothing (alpha=0.1)"))
cells.append(code_cell("""def train_with_label_smoothing(alpha=0.1, output_dir="output_label_smooth"):
    base_model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=NUM_CLASSES)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train_ds = make_hf_dataset(train_texts, train_labels, tokenizer)
    val_ds = make_hf_dataset(val_texts, val_labels, tokenizer)

    args = get_base_training_args(output_dir, label_smoothing_factor=alpha)
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=args, compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    return model, tokenizer

print("Training with label smoothing alpha=0.1...")
model_ls, tok_ls = train_with_label_smoothing(alpha=0.1)
print("\\nEvaluating on FPB:")
results_ls = eval_on_fpb(model_ls, tok_ls)"""))

# ── Cell 15: Technique 3 — Focal loss ──
cells.append(md_cell("### Technique 3: Focal Loss (gamma=2.0) + Class Weights"))
cells.append(code_cell("""class FocalLossTrainer(Trainer):
    def __init__(self, gamma=2.0, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, labels, reduction="none", weight=weight)
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        return (focal_loss, outputs) if return_outputs else focal_loss

def train_with_focal_loss(gamma=2.0, output_dir="output_focal"):
    base_model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=NUM_CLASSES)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train_ds = make_hf_dataset(train_texts, train_labels, tokenizer)
    val_ds = make_hf_dataset(val_texts, val_labels, tokenizer)

    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [total / (NUM_CLASSES * label_counts[i]) for i in range(NUM_CLASSES)]
    print(f"Class weights: {class_weights}")

    args = get_base_training_args(output_dir)
    trainer = FocalLossTrainer(
        gamma=gamma, class_weights=class_weights,
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=args, compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    return model, tokenizer

print("Training with focal loss gamma=2.0...")
model_focal, tok_focal = train_with_focal_loss(gamma=2.0)
print("\\nEvaluating on FPB:")
results_focal = eval_on_fpb(model_focal, tok_focal)"""))

# ── Cell 16: Technique 4 — Curriculum learning ──
cells.append(md_cell("### Technique 4: Curriculum Learning (easy → hard)"))
cells.append(code_cell("""def train_curriculum(output_dir="output_curriculum"):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Score difficulty using finbert-tone confidence
    print("Scoring training samples with finbert-tone...")
    ft_train_preds, ft_train_probs = predict_batch(train_texts, ft_model, ft_tokenizer, device, ft_label_map, max_length=512)
    ft_confidence = ft_train_probs.max(axis=1)

    threshold = np.percentile(ft_confidence, 40)
    easy_mask = ft_confidence >= threshold
    easy_texts = [t for t, m in zip(train_texts, easy_mask) if m]
    easy_labels = [l for l, m in zip(train_labels, easy_mask) if m]
    print(f"Curriculum: {len(easy_texts)} easy / {len(train_texts)} total (threshold={threshold:.4f})")

    base_model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=NUM_CLASSES)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()

    easy_ds = make_hf_dataset(easy_texts, easy_labels, tokenizer)
    all_ds = make_hf_dataset(train_texts, train_labels, tokenizer)
    val_ds = make_hf_dataset(val_texts, val_labels, tokenizer)

    # Stage 1: 5 epochs on easy samples
    print("\\nStage 1: Training on easy samples (5 epochs)...")
    args_s1 = get_base_training_args(f"{output_dir}_s1", num_train_epochs=5, save_strategy="no", load_best_model_at_end=False)
    Trainer(model=model, processing_class=tokenizer, train_dataset=easy_ds, eval_dataset=val_ds, args=args_s1, compute_metrics=compute_metrics_fn).train()

    # Stage 2: 5 epochs on all samples with lower LR
    print("\\nStage 2: Training on all samples (5 epochs, lower LR)...")
    args_s2 = get_base_training_args(f"{output_dir}_s2", num_train_epochs=5, learning_rate=1e-4)
    Trainer(model=model, processing_class=tokenizer, train_dataset=all_ds, eval_dataset=val_ds, args=args_s2, compute_metrics=compute_metrics_fn).train()

    return model, tokenizer

print("Training with curriculum learning...")
model_curr, tok_curr = train_curriculum()
print("\\nEvaluating on FPB:")
results_curr = eval_on_fpb(model_curr, tok_curr)"""))

# ── Cell 17: Results summary ──
cells.append(md_cell("## Part D: Results Summary"))
cells.append(code_cell("""results_table = pd.DataFrame([
    {"Model": "finbert-tone (zero-shot)", "Acc": ft_acc_50, "F1": ft_f1_50, "Training": "None"},
    {"Model": "ModernFinBERT baseline (r=16)", "Acc": mfb_acc_50, "F1": mfb_f1_50, "Training": "8.6K, LoRA r=16"},
    {"Model": "ModernFinBERT (r=64)", "Acc": results_r64["accuracy"], "F1": results_r64["macro_f1"], "Training": "8.6K, LoRA r=64"},
    {"Model": "ModernFinBERT + label smooth", "Acc": results_ls["accuracy"], "F1": results_ls["macro_f1"], "Training": "8.6K, LoRA r=16, smooth=0.1"},
    {"Model": "ModernFinBERT + focal loss", "Acc": results_focal["accuracy"], "F1": results_focal["macro_f1"], "Training": "8.6K, LoRA r=16, focal g=2"},
    {"Model": "ModernFinBERT curriculum", "Acc": results_curr["accuracy"], "F1": results_curr["macro_f1"], "Training": "8.6K, 2-stage curriculum"},
])

results_table = results_table.sort_values("Acc", ascending=False).reset_index(drop=True)
results_table["gap_vs_FT"] = results_table["Acc"] - ft_acc_50

print("\\n" + "=" * 90)
print("FULL RESULTS — FPB sentences_50agree (4,846 samples)")
print("=" * 90)
print(results_table.to_string(index=False, float_format="%.4f"))

results_table.to_csv("results/finbert_tone_gap_analysis.csv", index=False)
print(f"\\nResults saved to results/finbert_tone_gap_analysis.csv")"""))

# ── Build notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": [fix_source(c) for c in cells],
}

with open("notebooks/10_finbert_tone_deep_dive.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Created notebooks/10_finbert_tone_deep_dive.ipynb")
print(f"Total cells: {len(cells)}")
