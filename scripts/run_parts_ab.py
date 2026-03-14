"""Run Parts A+B of notebook 10 locally (inference + error analysis). No peft needed."""
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import zipfile
from huggingface_hub import hf_hub_download
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_CLASSES = 3
FPB_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


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
                if label_str in FPB_LABEL_MAP:
                    texts.append(text)
                    labels.append(FPB_LABEL_MAP[label_str])
    return texts, np.array(labels)


# ── A1: Load models ──
print("\n=== Loading models ===")
ft_name = "yiyanghkust/finbert-tone"
ft_tokenizer = BertTokenizer.from_pretrained(ft_name)
ft_model = BertForSequenceClassification.from_pretrained(ft_name).to(device).eval()
print(f"finbert-tone id2label: {ft_model.config.id2label}")

mfb_name = "neoyipeng/ModernFinBERT-base"
mfb_tokenizer = AutoTokenizer.from_pretrained(mfb_name)
mfb_model = AutoModelForSequenceClassification.from_pretrained(mfb_name).to(device).eval()
print(f"ModernFinBERT id2label: {mfb_model.config.id2label}")

# ── A2: Load FPB ──
print("\n=== Loading FPB data ===")
fpb_texts, fpb_labels = load_fpb_from_zip("Sentences_50Agree.txt")
fpb_all_texts, fpb_all_labels = load_fpb_from_zip("Sentences_AllAgree.txt")

print(f"FPB 50agree: {len(fpb_texts)} samples — {Counter(fpb_labels.tolist())}")
print(f"FPB allAgree: {len(fpb_all_texts)} samples — {Counter(fpb_all_labels.tolist())}")

# ── A3: predict_batch ──
def predict_batch(texts, model, tokenizer, dev, label_map, batch_size=64, max_length=512):
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
    return np.array(all_preds), np.vstack(all_probs)

# ── A4: Build label maps + run inference ──
_ft_name_to_fpb = {"positive": 2, "negative": 0, "neutral": 1}
ft_label_map = {int(k): _ft_name_to_fpb[v.lower()] for k, v in ft_model.config.id2label.items()}
print(f"\nfinbert-tone label_map: {ft_label_map}")
mfb_label_map = {0: 0, 1: 1, 2: 2}

print("\nRunning inference on FPB 50agree...")
ft_preds_50, ft_probs_50 = predict_batch(fpb_texts, ft_model, ft_tokenizer, device, ft_label_map)
mfb_preds_50, mfb_probs_50 = predict_batch(fpb_texts, mfb_model, mfb_tokenizer, device, mfb_label_map)

print("Running inference on FPB allAgree...")
ft_preds_all, ft_probs_all = predict_batch(fpb_all_texts, ft_model, ft_tokenizer, device, ft_label_map)
mfb_preds_all, mfb_probs_all = predict_batch(fpb_all_texts, mfb_model, mfb_tokenizer, device, mfb_label_map)

# ── A5: Metrics ──
def full_report(y_true, y_pred, model_name, label_names=LABEL_NAMES):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'='*60}")
    print(f"{model_name}: Accuracy={acc:.4f}, Macro F1={f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    return acc, f1

print("\n" + "=" * 70)
print("FPB sentences_50agree (4,846 samples)")
print("=" * 70)
ft_acc_50, ft_f1_50 = full_report(fpb_labels, ft_preds_50, "finbert-tone (zero-shot)")
mfb_acc_50, mfb_f1_50 = full_report(fpb_labels, mfb_preds_50, "ModernFinBERT (held-out)")
print(f"\n>>> Gap (MFB - FT): Acc={mfb_acc_50 - ft_acc_50:+.4f}, F1={mfb_f1_50 - ft_f1_50:+.4f}")

print("\n" + "=" * 70)
print("FPB sentences_allAgree (2,264 samples)")
print("=" * 70)
ft_acc_all, ft_f1_all = full_report(fpb_all_labels, ft_preds_all, "finbert-tone (zero-shot)")
mfb_acc_all, mfb_f1_all = full_report(fpb_all_labels, mfb_preds_all, "ModernFinBERT (held-out)")
print(f"\n>>> Gap (MFB - FT): Acc={mfb_acc_all - ft_acc_all:+.4f}, F1={mfb_f1_all - ft_f1_all:+.4f}")

# ── B1: Confusion matrices ──
print("\n=== Confusion Matrices ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, name in [(axes[0], ft_preds_50, "finbert-tone"), (axes[1], mfb_preds_50, "ModernFinBERT")]:
    cm = confusion_matrix(fpb_labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{name}")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_ft_vs_mfb.png"), dpi=150, bbox_inches="tight")
print(f"Saved confusion matrices to results/confusion_ft_vs_mfb.png")

# ── B2: Sentence-level agreement ──
print("\n=== Sentence-Level Agreement ===")
df = pd.DataFrame({
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
print(f"  both_right:  {counts.get('both_right', 0)} — both models correct")
print(f"  both_wrong:  {counts.get('both_wrong', 0)} — neither model correct")
print(f"  ft_only:     {counts.get('ft_only', 0)} — finbert-tone right, MFB wrong")
print(f"  mfb_only:    {counts.get('mfb_only', 0)} — MFB right, finbert-tone wrong")
print(f"\nMFB unique advantage: {counts.get('mfb_only', 0) - counts.get('ft_only', 0):+d} sentences")

# ── B3: Per-class + text length ──
print("\n=== Per-Class Accuracy ===")
for cls_idx, cls_name in enumerate(LABEL_NAMES):
    mask = fpb_labels == cls_idx
    ft_cls_acc = (ft_preds_50[mask] == fpb_labels[mask]).mean()
    mfb_cls_acc = (mfb_preds_50[mask] == fpb_labels[mask]).mean()
    n = mask.sum()
    print(f"  {cls_name:>10} (n={n:4d}): FT={ft_cls_acc:.4f}  MFB={mfb_cls_acc:.4f}  delta={mfb_cls_acc - ft_cls_acc:+.4f}")

df["word_count"] = df["text"].str.split().str.len()
df["length_bin"] = pd.cut(df["word_count"], bins=[0, 15, 25, 40, 999], labels=["short", "medium", "long", "very_long"])

print("\nAccuracy by text length:")
for bin_name in ["short", "medium", "long", "very_long"]:
    mask = df["length_bin"] == bin_name
    if mask.sum() == 0:
        continue
    ft_acc = df.loc[mask, "ft_correct"].mean()
    mfb_acc = df.loc[mask, "mfb_correct"].mean()
    n = mask.sum()
    print(f"  {bin_name:>10} (n={n:4d}): FT={ft_acc:.4f}  MFB={mfb_acc:.4f}  delta={mfb_acc - ft_acc:+.4f}")

# ── B4: Confidence analysis ──
print("\n=== Confidence Analysis ===")
ft_label_map_reverse = {v: k for k, v in ft_label_map.items()}
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
    print(f"\n{model_name} — accuracy by confidence quartile:")
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
        print(f"  (all): acc={acc:.4f}, mean_conf={mean_conf:.4f}, n={len(df)} — confidence too concentrated for quartiles")

# ── B5: Qualitative examples ──
print("\n" + "=" * 80)
print("finbert-tone CORRECT, ModernFinBERT WRONG (sample 10):")
print("=" * 80)
ft_only_df = df[df["category"] == "ft_only"]
for _, row in ft_only_df.sample(min(10, len(ft_only_df)), random_state=42).iterrows():
    print(f"\n  Text: {row['text'][:120]}...")
    print(f"  True: {LABEL_NAMES[row['label']]} | FT: {LABEL_NAMES[row['ft_pred']]} ok | MFB: {LABEL_NAMES[row['mfb_pred']]} X")

print("\n" + "=" * 80)
print("ModernFinBERT CORRECT, finbert-tone WRONG (sample 10):")
print("=" * 80)
mfb_only_df = df[df["category"] == "mfb_only"]
for _, row in mfb_only_df.sample(min(10, len(mfb_only_df)), random_state=42).iterrows():
    print(f"\n  Text: {row['text'][:120]}...")
    print(f"  True: {LABEL_NAMES[row['label']]} | FT: {LABEL_NAMES[row['ft_pred']]} X | MFB: {LABEL_NAMES[row['mfb_pred']]} ok")

# ── Save results to JSON ──
import json
results = {
    "fpb_50agree": {
        "finbert_tone": {"accuracy": float(ft_acc_50), "macro_f1": float(ft_f1_50)},
        "modernfinbert": {"accuracy": float(mfb_acc_50), "macro_f1": float(mfb_f1_50)},
        "gap_acc": float(mfb_acc_50 - ft_acc_50),
        "gap_f1": float(mfb_f1_50 - ft_f1_50),
        "n_samples": len(fpb_texts),
    },
    "fpb_allagree": {
        "finbert_tone": {"accuracy": float(ft_acc_all), "macro_f1": float(ft_f1_all)},
        "modernfinbert": {"accuracy": float(mfb_acc_all), "macro_f1": float(mfb_f1_all)},
        "gap_acc": float(mfb_acc_all - ft_acc_all),
        "gap_f1": float(mfb_f1_all - ft_f1_all),
        "n_samples": len(fpb_all_texts),
    },
    "agreement": {k: int(v) for k, v in counts.items()},
}
results_path = os.path.join(RESULTS_DIR, "nb10_parts_ab.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved numeric results to {results_path}")
print("\n=== DONE ===")
