# Confidence Calibration for ModernFinBERT — Implementation Plan

## Problem Statement

ModernFinBERT is **overconfident on its errors**. From the NB15 error analysis:

- Correct predictions: mean confidence **0.964**, median ~0.98
- Misclassifications: mean confidence **0.928**, median ~0.96
- **31% of all errors** have confidence > 0.9 (the model is wrong but very sure)
- The gap between correct and incorrect confidence is only ~0.036 — not enough for users to filter bad predictions

This means the raw softmax probabilities are not trustworthy. A user seeing "95% POSITIVE" cannot distinguish whether the model is reliably correct or confidently wrong. For financial applications where bad sentiment signals drive trading losses, this is a critical safety issue.

**Goal:** Make the model's confidence scores reflect its actual accuracy. When it says "70% POSITIVE," it should be correct ~70% of the time. When it says "95% POSITIVE," it should be correct ~95% of the time.

---

## Background: What is Calibration?

A model is **perfectly calibrated** when its predicted confidence matches its actual accuracy at every confidence level. The standard metric is **Expected Calibration Error (ECE)**:

```
ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|
```

Where samples are grouped into M bins by confidence, and for each bin we compare the average confidence to the actual accuracy. Lower ECE = better calibrated.

Modern neural networks are systematically overconfident (Guo et al., 2017 — "On Calibration of Modern Neural Networks"). The standard fix is **post-hoc calibration** — learning a simple transformation on a held-out validation set that maps raw logits to calibrated probabilities. Crucially, this **does not change the model's predictions** (the argmax is preserved), only the confidence scores.

---

## Architecture: Where Calibration Fits

```
                          CURRENT PIPELINE
                          ================
Input text → Tokenizer → ModernBERT → Logits → softmax → Probabilities
                                                              ↓
                                                    argmax = Prediction
                                                    max(p) = Confidence


                      CALIBRATED PIPELINE
                      ====================
Input text → Tokenizer → ModernBERT → Logits → T(logits) → softmax → Calibrated Probs
                                                   ↑            ↓
                                          Temperature T    argmax = Prediction (same!)
                                       (learned scalar)    max(p) = Confidence (fixed!)
```

The calibration layer sits between the raw logits and the softmax. It is a lightweight transformation — typically a single scalar (temperature) or a small linear layer — learned on a held-out validation set that was NOT used for training.

---

## Implementation Plan

### Step 1: Build the Calibration Dataset

**Challenge:** The production model (`neoyipeng/ModernFinBERT-base`) was trained on ALL data including FPB. We need logits on data the model hasn't trained on.

**Solution:** Use **5-fold cross-validation logit collection**. This is the gold standard approach (Guo et al., 2017):

1. Split the complete dataset into 5 stratified folds
2. For each fold, train a model on the other 4 folds (same hyperparams as production)
3. Collect logits on the held-out fold
4. After all 5 folds, we have logits for every sample — each collected when that sample was unseen

This gives us ~13,900 logit vectors (the full dataset) with no data leakage.

**Alternative (simpler, slightly less rigorous):** Hold out 10-15% of training data as a calibration set before training the production model. This is faster but wastes training data. Since we already have the 5-fold CV infrastructure from NB14, the cross-validation approach is preferred.

```python
# Step 1: Collect held-out logits via 5-fold CV
import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "answerdotai/ModernBERT-base"
NUM_FOLDS = 5
SEED = 3407

# Load the complete dataset (same as NB14)
ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
# Filter to sentiment task, combine all splits
# ... (same preprocessing as NB14)

texts = all_texts       # list[str], length ~13,900
labels = all_labels     # np.array of int, shape (13900,)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

# Storage: one logit vector per sample
all_logits = np.zeros((len(texts), 3), dtype=np.float32)
all_labels_ordered = np.zeros(len(texts), dtype=np.int64)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1}/{NUM_FOLDS}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"{'='*60}")

    # Build train/val datasets for this fold
    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = labels[val_idx]

    # Train model (same config as production — full FT, same hyperparams)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
    )

    # ... same TrainingArguments as NB14 ...
    # ... train the model ...

    # Collect logits on held-out fold
    model.eval()
    fold_logits = []
    with torch.no_grad():
        for i in range(0, len(val_texts), 32):
            batch = tokenizer(
                val_texts[i:i+32],
                return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            fold_logits.append(logits)

    fold_logits = np.concatenate(fold_logits, axis=0)

    # Store in the correct positions
    all_logits[val_idx] = fold_logits
    all_labels_ordered[val_idx] = val_labels

    # Free memory
    del model
    torch.cuda.empty_cache()

# Save for calibration fitting
np.savez(
    "results/calibration_logits.npz",
    logits=all_logits,
    labels=all_labels_ordered,
)
print(f"Saved {len(all_logits)} logit vectors to results/calibration_logits.npz")
```

**GPU time estimate:** ~5 fold trainings x ~40 min each = ~3.5 hours on T4. This is the same cost as the multi-seed experiment (NB06).

---

### Step 2: Measure Pre-Calibration ECE

Before applying any calibration, measure how bad the problem is.

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_ece(logits, labels, n_bins=15):
    """
    Compute Expected Calibration Error.

    Args:
        logits: np.array of shape (N, C) — raw logits
        labels: np.array of shape (N,) — ground truth class indices
        n_bins: number of equal-width confidence bins

    Returns:
        ece: float, the expected calibration error
        bin_data: list of dicts with per-bin stats for plotting
    """
    # Convert logits to probabilities
    # Use numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append({
                "bin_lo": lo, "bin_hi": hi,
                "count": 0, "avg_conf": 0, "avg_acc": 0,
            })
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_count = mask.sum()
        bin_ece = (bin_count / len(labels)) * abs(bin_acc - bin_conf)
        ece += bin_ece

        bin_data.append({
            "bin_lo": lo, "bin_hi": hi,
            "count": int(bin_count),
            "avg_conf": float(bin_conf),
            "avg_acc": float(bin_acc),
        })

    return float(ece), bin_data


def plot_reliability_diagram(bin_data, ece, title="Reliability Diagram", save_path=None):
    """
    Plot a reliability diagram (calibration curve).

    The diagonal represents perfect calibration. Bars above the diagonal
    mean the model is underconfident; bars below mean overconfident.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: reliability diagram
    confs = [b["avg_conf"] for b in bin_data if b["count"] > 0]
    accs = [b["avg_acc"] for b in bin_data if b["count"] > 0]
    counts = [b["count"] for b in bin_data if b["count"] > 0]

    ax1.bar(confs, accs, width=0.05, alpha=0.7, color="#42A5F5",
            edgecolor="white", label="Outputs")
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax1.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")

    # Right: confidence histogram
    all_confs = [b["avg_conf"] for b in bin_data]
    all_counts = [b["count"] for b in bin_data]
    bin_centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bin_data]
    ax2.bar(bin_centers, all_counts, width=0.065, color="#66BB6A",
            edgecolor="white", alpha=0.7)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig


# Load saved logits
data = np.load("results/calibration_logits.npz")
logits = data["logits"]
labels = data["labels"]

# Measure pre-calibration ECE
ece_before, bins_before = compute_ece(logits, labels, n_bins=15)
print(f"Pre-calibration ECE: {ece_before:.4f}")

plot_reliability_diagram(
    bins_before, ece_before,
    title="ModernFinBERT — Before Calibration",
    save_path="results/reliability_before.png",
)
```

**Expected result:** ECE around 0.05-0.12 (typical for uncalibrated fine-tuned transformers). The reliability diagram should show bars consistently below the diagonal in the high-confidence region — confirming the model is overconfident.

---

### Step 3: Fit Calibration Methods

We'll implement and compare three calibration methods, from simplest to most expressive:

#### Method A: Temperature Scaling (Recommended)

A single scalar T > 1 is learned to soften the logits. This is the gold standard from Guo et al., 2017 — it's simple, has only 1 parameter, and works surprisingly well.

```python
import torch
import torch.nn as nn
from torch.optim import LBFGS

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for calibration.

    Learns a single temperature parameter T such that
    calibrated_probs = softmax(logits / T).

    T > 1 softens the distribution (reduces overconfidence).
    T < 1 sharpens it (increases confidence).
    T = 1 is the identity (no calibration).
    """

    def __init__(self):
        super().__init__()
        # Initialize at T=1 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        """
        Learn temperature on a validation set using NLL loss + L-BFGS.

        Args:
            logits: torch.Tensor (N, C) — raw logits from the model
            labels: torch.Tensor (N,) — ground truth class indices
            lr: learning rate for L-BFGS
            max_iter: max optimization iterations

        Returns:
            float: learned temperature value
        """
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Report
        final_loss = nll_criterion(self.forward(logits), labels).item()
        print(f"Learned temperature: T = {self.temperature.item():.4f}")
        print(f"Final NLL loss: {final_loss:.4f}")

        return self.temperature.item()


# Fit temperature scaling
data = np.load("results/calibration_logits.npz")
logits_np = data["logits"]
labels_np = data["labels"]

logits_tensor = torch.from_numpy(logits_np).float()
labels_tensor = torch.from_numpy(labels_np).long()

temp_scaler = TemperatureScaler()
T = temp_scaler.calibrate(logits_tensor, labels_tensor)

# Measure post-calibration ECE
scaled_logits = (logits_np / T)  # Apply learned temperature
ece_after_temp, bins_after_temp = compute_ece(scaled_logits, labels_np, n_bins=15)
print(f"\nPost-calibration ECE (temperature): {ece_after_temp:.4f}")
print(f"Improvement: {ece_before:.4f} → {ece_after_temp:.4f} "
      f"({(1 - ece_after_temp/ece_before)*100:.1f}% reduction)")
```

**Expected:** T will be in the range 1.5-3.0 (softening the overconfident logits). ECE should drop by 50-80%.

#### Method B: Vector Scaling (Per-Class Temperature)

Instead of a single T, learn a separate temperature per class. This handles cases where the model is differently calibrated per class (e.g., very overconfident on NEUTRAL but less so on NEGATIVE).

```python
class VectorScaler(nn.Module):
    """
    Per-class temperature + bias calibration.

    Learns a diagonal scaling matrix W and bias b:
        calibrated_logits = W * logits + b

    Where W is shape (C,) and b is shape (C,).
    More expressive than temperature scaling (3+3=6 params for 3 classes)
    but still unlikely to overfit with ~14K calibration samples.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        return logits * self.W + self.b

    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.W, self.b], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        print(f"Learned W: {self.W.data.numpy()}")
        print(f"Learned b: {self.b.data.numpy()}")
        return self.W.data.numpy(), self.b.data.numpy()


vec_scaler = VectorScaler(num_classes=3)
W, b = vec_scaler.calibrate(logits_tensor, labels_tensor)

scaled_logits_vec = logits_np * W + b
ece_after_vec, bins_after_vec = compute_ece(scaled_logits_vec, labels_np, n_bins=15)
print(f"Post-calibration ECE (vector): {ece_after_vec:.4f}")
```

#### Method C: Platt Scaling (Logistic Regression)

A full linear transformation: `calibrated_logits = W @ logits + b` where W is (C, C). More expressive but risks overfitting with only 3 classes.

```python
from sklearn.linear_model import LogisticRegression

def platt_scaling(logits, labels):
    """
    Platt scaling via sklearn LogisticRegression on the logit space.

    This fits a full (C, C) weight matrix + (C,) bias on the raw logits.
    Regularization via C=1.0 prevents overfitting.
    """
    lr = LogisticRegression(
        C=1.0,              # regularization strength
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
    )
    lr.fit(logits, labels)

    # The LogisticRegression's predict_proba gives calibrated probabilities
    calibrated_probs = lr.predict_proba(logits)

    # Extract the transformation for export
    W_platt = lr.coef_          # shape (C, C)
    b_platt = lr.intercept_     # shape (C,)

    return lr, calibrated_probs, W_platt, b_platt


lr_model, cal_probs_platt, W_platt, b_platt = platt_scaling(logits_np, labels_np)

# Measure ECE from calibrated probs directly
confidences = np.max(cal_probs_platt, axis=1)
predictions = np.argmax(cal_probs_platt, axis=1)
# ... compute ECE on these ...
```

---

### Step 4: Compare Methods and Select Winner

```python
import json

results = {
    "pre_calibration": {"ece": ece_before},
    "temperature_scaling": {
        "ece": ece_after_temp,
        "temperature": T,
        "params": 1,
    },
    "vector_scaling": {
        "ece": ece_after_vec,
        "W": W.tolist(),
        "b": b.tolist(),
        "params": 6,
    },
    "platt_scaling": {
        "ece": ece_after_platt,
        "params": 12,  # 3x3 + 3
    },
}

print("\nCalibration Method Comparison")
print("=" * 55)
print(f"{'Method':<25} {'ECE':>8} {'Params':>8} {'Δ ECE':>10}")
print("-" * 55)
for name, r in results.items():
    delta = f"{r['ece'] - ece_before:+.4f}" if name != "pre_calibration" else "---"
    params = r.get("params", 0)
    print(f"{name:<25} {r['ece']:>8.4f} {params:>8} {delta:>10}")

# Save comparison
with open("results/calibration_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot all reliability diagrams side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, bins, ece_val) in zip(axes, [
    ("Before Calibration", bins_before, ece_before),
    ("Temperature Scaling", bins_after_temp, ece_after_temp),
    ("Vector Scaling", bins_after_vec, ece_after_vec),
]):
    confs = [b["avg_conf"] for b in bins if b["count"] > 0]
    accs = [b["avg_acc"] for b in bins if b["count"] > 0]
    ax.bar(confs, accs, width=0.05, alpha=0.7, color="#42A5F5", edgecolor="white")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2)
    ax.set_title(f"{name}\nECE = {ece_val:.4f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

plt.suptitle("ModernFinBERT Calibration — Reliability Diagrams", fontsize=14)
plt.tight_layout()
plt.savefig("paper/figures/fig3_calibration.pdf", dpi=150, bbox_inches="tight")
plt.savefig("results/calibration_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Selection criteria:**
- Temperature scaling wins if ECE is within 0.005 of the best method (simplest, most robust, 1 param)
- Vector scaling wins if there's a meaningful per-class calibration difference
- Platt scaling wins only if it's substantially better AND we verify no overfitting via a held-out test

**Expected winner:** Temperature scaling. It almost always wins on classification tasks with < 10 classes (Guo et al., 2017).

---

### Step 5: Save Calibration Parameters

The calibration must be saved so it can be applied at inference time without retraining.

```python
import json

# For temperature scaling (the expected winner):
calibration_config = {
    "method": "temperature_scaling",
    "temperature": float(T),
    "ece_before": float(ece_before),
    "ece_after": float(ece_after_temp),
    "calibration_samples": len(labels_np),
    "num_folds": NUM_FOLDS,
    "seed": SEED,
    "label_names": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
}

with open("calibration_config.json", "w") as f:
    json.dump(calibration_config, f, indent=2)

print(f"Saved calibration config: T = {T:.4f}")
print(f"To apply: calibrated_probs = softmax(logits / {T:.4f})")
```

---

### Step 6: Integrate Into Inference Pipeline

Update both the Gradio demo (`demo/app.py`) and any production inference code to apply calibration.

#### Updated `demo/app.py`

```python
"""
ModernFinBERT Gradio Demo — with confidence calibration.
"""

import json
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "neoyipeng/ModernFinBERT-base"
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Load calibration config
with open("calibration_config.json") as f:
    cal_config = json.load(f)
TEMPERATURE = cal_config["temperature"]
print(f"Calibration: T = {TEMPERATURE:.4f} "
      f"(ECE: {cal_config['ece_before']:.4f} → {cal_config['ece_after']:.4f})")

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print(f"Model loaded on {device}.")


def predict(text: str) -> dict[str, float]:
    """Return calibrated sentiment confidences for the input text."""
    if not text or not text.strip():
        return {label: 0.0 for label in LABEL_NAMES}

    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

        # Apply temperature scaling for calibrated probabilities
        calibrated_logits = logits / TEMPERATURE
        probs = torch.softmax(calibrated_logits, dim=-1).squeeze().cpu().numpy()

    return {label: float(round(prob, 4)) for label, prob in zip(LABEL_NAMES, probs)}
```

#### Standalone Calibrated Inference Function

For use in scripts and notebooks:

```python
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CalibratedModernFinBERT:
    """
    ModernFinBERT with post-hoc temperature calibration.

    Usage:
        model = CalibratedModernFinBERT("neoyipeng/ModernFinBERT-base")
        result = model.predict("Revenue grew 15% year-over-year.")
        # {'label': 'POSITIVE', 'confidence': 0.82, 'probabilities': {...}}
    """

    LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    def __init__(self, model_id, calibration_path="calibration_config.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = self.model.to(self.device).eval()

        # Load calibration
        with open(calibration_path) as f:
            config = json.load(f)
        self.temperature = config["temperature"]

    def predict(self, text: str) -> dict:
        """Classify a single text with calibrated confidence."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            calibrated_logits = logits / self.temperature
            probs = torch.softmax(calibrated_logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))

        return {
            "label": self.LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(round(p, 4))
                for name, p in zip(self.LABEL_NAMES, probs)
            },
            "calibrated": True,
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Classify a batch of texts with calibrated confidence."""
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                calibrated_logits = logits / self.temperature
                probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy()

            for j, p in enumerate(probs):
                pred_idx = int(np.argmax(p))
                all_results.append({
                    "label": self.LABEL_NAMES[pred_idx],
                    "confidence": float(p[pred_idx]),
                    "probabilities": {
                        name: float(round(v, 4))
                        for name, v in zip(self.LABEL_NAMES, p)
                    },
                    "calibrated": True,
                })

        return all_results
```

---

### Step 7: Validate Calibration Doesn't Hurt Accuracy

Critical check: calibration must not change the predicted labels (argmax is preserved under temperature scaling since T > 0). Verify this explicitly.

```python
# Verify argmax preservation
raw_preds = np.argmax(logits_np, axis=1)
cal_preds = np.argmax(logits_np / T, axis=1)

assert np.array_equal(raw_preds, cal_preds), \
    "Temperature scaling changed predictions! This should never happen."

print("Verified: calibration preserves all predictions (argmax unchanged)")
print(f"Accuracy before calibration: {(raw_preds == labels_np).mean():.4f}")
print(f"Accuracy after calibration:  {(cal_preds == labels_np).mean():.4f}")
```

Also verify that the **confidence gap** between correct and incorrect predictions widens:

```python
# Confidence separation analysis
raw_probs = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
raw_conf = np.max(raw_probs, axis=1)

cal_logits = logits_np / T
cal_probs = np.exp(cal_logits - np.max(cal_logits, axis=1, keepdims=True))
cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
cal_conf = np.max(cal_probs, axis=1)

correct_mask = (raw_preds == labels_np)

print("\nConfidence Separation (higher gap = better)")
print(f"{'Metric':<35} {'Before':>10} {'After':>10}")
print("-" * 55)
print(f"{'Mean conf (correct)':<35} {raw_conf[correct_mask].mean():>10.4f} "
      f"{cal_conf[correct_mask].mean():>10.4f}")
print(f"{'Mean conf (incorrect)':<35} {raw_conf[~correct_mask].mean():>10.4f} "
      f"{cal_conf[~correct_mask].mean():>10.4f}")
print(f"{'Gap (correct - incorrect)':<35} "
      f"{raw_conf[correct_mask].mean() - raw_conf[~correct_mask].mean():>10.4f} "
      f"{cal_conf[correct_mask].mean() - cal_conf[~correct_mask].mean():>10.4f}")
print(f"{'High-conf errors (>0.9) %':<35} "
      f"{(raw_conf[~correct_mask] > 0.9).mean()*100:>9.1f}% "
      f"{(cal_conf[~correct_mask] > 0.9).mean()*100:>9.1f}%")
```

**Expected:** The gap between correct and incorrect confidence should stay the same or widen slightly. High-confidence errors (>0.9) should drop dramatically — from ~31% to potentially <5%.

---

### Step 8: Add Calibration to the Paper

Add a new subsection to `paper/main.tex` in the Analysis and Discussion section:

```latex
\subsection{Confidence Calibration}
\label{sec:calibration}

Neural networks are known to be overconfident in their predictions
\citep{guo2017calibration}. Our error analysis (Section~\ref{sec:error-analysis})
confirms this for ModernFinBERT: 31\% of misclassified samples have prediction
confidence exceeding 0.9. To address this, we apply post-hoc temperature scaling.

We collect held-out logits via 5-fold cross-validation on the complete training
set (including DataBoost augmentation), yielding 13,900 logit vectors with no
data leakage. A single temperature parameter $T$ is learned by minimizing
negative log-likelihood on these logits.

\begin{table}[h]
\centering
\caption{Calibration results. ECE = Expected Calibration Error (lower is better).
Temperature scaling reduces ECE by X\% while preserving all predictions.}
\label{tab:calibration}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{ECE} & \textbf{Parameters} \\
\midrule
Uncalibrated (softmax) & X.XXXX & 0 \\
Temperature scaling ($T = X.XX$) & X.XXXX & 1 \\
Vector scaling & X.XXXX & 6 \\
\bottomrule
\end{tabular}
\end{table}

Temperature scaling achieves the best trade-off: ... [fill after results].
The calibrated model's confidence scores now reflect its true accuracy,
enabling practitioners to set meaningful confidence thresholds for
human-in-the-loop workflows.
```

---

### Step 9: Publish Updated Model

Two options for distributing the calibration:

**Option A (recommended): Config file alongside the model**

Upload `calibration_config.json` to the HuggingFace model repo. Users load it alongside the model. No model retraining needed.

```bash
# Upload calibration config to HuggingFace
huggingface-cli upload neoyipeng/ModernFinBERT-base \
    calibration_config.json calibration_config.json
```

Update the MODEL_CARD.md with calibrated inference example:

```python
# Calibrated inference
import json, torch
from transformers import pipeline

classifier = pipeline("text-classification", model="neoyipeng/ModernFinBERT-base")

# For calibrated probabilities, use the temperature from calibration_config.json:
# T = X.XX  (learned via 5-fold CV temperature scaling)
# calibrated_probs = softmax(logits / T)
```

**Option B: Bake temperature into the model**

Modify the classification head's bias/weights to absorb the temperature. This makes calibration invisible to users but is harder to update.

```python
# Bake temperature into the model (divide final layer weights by T)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
with torch.no_grad():
    model.classifier.weight.div_(T)
    model.classifier.bias.div_(T)
model.save_pretrained("neoyipeng/ModernFinBERT-base-calibrated")
```

Option A is preferred — it's transparent, reversible, and users can choose whether to apply calibration.

---

## File Structure for the Notebook

The implementation should be a single notebook: `notebooks/16_confidence_calibration.ipynb`

```
notebooks/16_confidence_calibration.ipynb
├── 1. Setup & Installation
├── 2. Collect Held-Out Logits (5-fold CV)
│   ├── Load complete dataset (same as NB14)
│   ├── 5-fold stratified split
│   └── Train + collect logits per fold
├── 3. Measure Pre-Calibration ECE
│   ├── ECE computation
│   └── Reliability diagram (before)
├── 4. Fit Calibration Methods
│   ├── A: Temperature scaling
│   ├── B: Vector scaling
│   └── C: Platt scaling
├── 5. Compare Methods
│   ├── ECE comparison table
│   └── Side-by-side reliability diagrams
├── 6. Validate Predictions Unchanged
│   ├── Argmax preservation check
│   └── Confidence separation analysis
├── 7. Per-Class Calibration Analysis
│   ├── Class-conditional ECE
│   └── Per-class reliability diagrams
├── 8. Save Results
│   ├── calibration_config.json
│   ├── results/calibration_comparison.json
│   └── results/reliability_diagrams.png
└── 9. Summary & Paper Text
```

---

## Expected Outcomes

| Metric | Before | After (expected) |
|---|---|---|
| ECE | ~0.08-0.12 | ~0.01-0.03 |
| High-conf errors (>0.9) | ~31% | ~5-10% |
| Mean conf gap (correct - wrong) | ~0.036 | ~0.15-0.25 |
| Accuracy | unchanged | unchanged |
| Inference latency | unchanged | unchanged (division by scalar) |
| Temperature T | 1.0 (identity) | ~1.5-3.0 |

---

## Resource Requirements

| Resource | Estimate |
|---|---|
| GPU time (5-fold CV on T4) | ~3.5 hours |
| Kaggle GPU quota | 1 session |
| New code | ~400 lines (1 notebook) |
| API costs | $0 (no LLM calls) |
| Risk to existing model | Zero (post-hoc, predictions preserved) |

---

## Dependencies on Existing Code

- **NB14 (production model):** Reuse exact data loading, preprocessing, and training config for the 5-fold CV logit collection
- **NB15 (error analysis):** The pre-calibration confidence stats (31% high-conf errors) serve as the baseline we're improving
- **demo/app.py:** Will be updated to use calibrated inference
- **MODEL_CARD.md:** Will be updated with calibration details

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Temperature scaling is insufficient | Low | Vector scaling and Platt scaling as fallbacks |
| Not enough calibration data | Very low | ~13,900 samples is more than enough for 1-6 params |
| Calibration hurts accuracy | Zero | Temperature scaling preserves argmax by construction |
| Calibration doesn't transfer to new domains | Medium | Acknowledge in paper; temperature is dataset-specific |
| 5-fold CV models differ from production model | Low | Same hyperparams, same data; minor seed variation only |

---

## Success Criteria

1. ECE drops by at least 50% (e.g., 0.10 → 0.05 or better)
2. High-confidence errors (>0.9) drop below 15% (from 31%)
3. All predictions (argmax) are identical before and after calibration
4. Reliability diagram shows bars hugging the diagonal
5. Results are reproducible (saved logits + config for exact replication)

---

## Detailed Task Checklist

### Phase 0: Preparation
- [x] **0.1** Verify Kaggle GPU quota is available (~3.5h needed on T4)
- [x] **0.2** Create `notebooks/16_confidence_calibration.ipynb` with markdown skeleton (section headers from the file structure above)
- [x] **0.3** Copy the data loading + preprocessing code from NB14 into NB16 Section 1 (setup cell, dataset loading, DataBoost merging, label mapping) — do NOT modify, just replicate for self-contained reproducibility
- [x] **0.4** Copy the `TrainingArguments` and LoRA/full-FT config from NB14 into a config dict in NB16 — this ensures the 5-fold models match the production model exactly
- [x] **0.5** Add `%%capture` install cell: `pip install -q datasets scikit-learn matplotlib seaborn transformers torch peft`

### Phase 1: Collect Calibration Logits (NB16 Sections 1-2)
- [x] **1.1** Load the complete dataset from `neoyipeng/financial_reasoning_aggregated` (all sources including FPB source 5)
- [x] **1.2** Load the 410 DataBoost augmentation samples from embedded gzip+base64 (same as NB14) and merge into the dataset
- [x] **1.3** Verify total sample count matches NB14 (~13,900 after merging all sources + DataBoost)
- [x] **1.4** Print dataset summary: total samples, per-class distribution, per-source counts
- [x] **1.5** Create `StratifiedKFold(n_splits=5, shuffle=True, random_state=3407)` splitter
- [x] **1.6** Implement the 5-fold training loop:
  - [x] **1.6a** For each fold: build HuggingFace `Dataset` objects for train/val from index arrays
  - [x] **1.6b** For each fold: instantiate fresh `answerdotai/ModernBERT-base` + classification head (num_labels=3)
  - [x] **1.6c** For each fold: apply same training config as NB14 (full FT: lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, batch=16 effective, fp16, gradient checkpointing, 10 epochs + early stopping)
  - [x] **1.6d** For each fold: train the model using HuggingFace `Trainer`
  - [x] **1.6e** For each fold: set model to `eval()` and run inference on the held-out fold in batches of 32
  - [x] **1.6f** For each fold: store raw logits (NOT softmax probs) in `all_logits[val_idx]` and labels in `all_labels[val_idx]`
  - [x] **1.6g** For each fold: print fold accuracy + macro F1 on the held-out portion as a sanity check
  - [x] **1.6h** For each fold: delete model and call `torch.cuda.empty_cache()` to free GPU memory
- [x] **1.7** After all folds: verify `all_logits` has no zero rows (every sample was filled exactly once)
- [x] **1.8** Save logits to `results/calibration_logits.npz` with keys `logits` (N, 3) and `labels` (N,)
- [x] **1.9** Print summary: total logits collected, mean fold accuracy, wall-clock time per fold

### Phase 2: Pre-Calibration Measurement (NB16 Section 3)
- [x] **2.1** Implement `compute_ece(logits, labels, n_bins=15)` function — returns ECE float + per-bin data list
- [x] **2.2** Implement `plot_reliability_diagram(bin_data, ece, title, save_path)` function — left panel: reliability bars + diagonal, right panel: confidence histogram
- [x] **2.3** Load saved logits from `results/calibration_logits.npz`
- [x] **2.4** Compute pre-calibration ECE on the full logit set (15 bins)
- [x] **2.5** Also compute ECE with 10 bins and 20 bins to verify ECE is stable across bin count
- [x] **2.6** Plot and save pre-calibration reliability diagram to `results/reliability_before.png`
- [x] **2.7** Print pre-calibration confidence stats: mean/median confidence overall, for correct preds, for errors, % of errors with conf > 0.9, % with conf > 0.95
- [x] **2.8** Print per-class pre-calibration stats: for each of NEGATIVE, NEUTRAL, POSITIVE — mean confidence on correct, mean confidence on errors, class-conditional ECE

### Phase 3: Fit Calibration Methods (NB16 Section 4)
- [x] **3.1** Implement `TemperatureScaler(nn.Module)` class with `forward(logits)` and `calibrate(logits, labels)` using L-BFGS
- [x] **3.2** Fit temperature scaling: convert logits/labels to torch tensors, call `calibrate()`, print learned T
- [x] **3.3** Compute post-temperature-scaling ECE and store result
- [x] **3.4** Implement `VectorScaler(nn.Module)` class with per-class W and b parameters, L-BFGS optimization
- [x] **3.5** Fit vector scaling, print learned W and b per class, compute post-vector-scaling ECE
- [x] **3.6** Implement `platt_scaling(logits, labels)` using `sklearn.linear_model.LogisticRegression` on the logit space
- [x] **3.7** Fit Platt scaling, compute ECE from `predict_proba` output
- [x] **3.8** For Platt scaling: verify the fitted model doesn't change >1% of predictions (since Platt can alter argmax unlike temperature scaling, flag if this happens)

### Phase 4: Compare Methods and Select Winner (NB16 Section 5)
- [x] **4.1** Build comparison table: method name, ECE, delta vs. pre-calibration, number of parameters
- [x] **4.2** Print formatted comparison table
- [x] **4.3** Plot 3-panel reliability diagram (Before / Temperature / Vector) and save to `results/calibration_comparison.png`
- [ ] **4.4** Also save as `paper/figures/fig3_calibration.pdf` for the paper *(PNG saved; PDF requires running on Kaggle)*
- [x] **4.5** Apply selection criteria: pick temperature scaling unless another method beats it by >0.005 ECE
- [x] **4.6** Save comparison results to `results/calibration_comparison.json`
- [x] **4.7** Print the winning method and its parameters

### Phase 5: Validation (NB16 Section 6)
- [x] **5.1** Argmax preservation check: assert `np.array_equal(argmax(logits), argmax(logits / T))` — print explicit confirmation
- [x] **5.2** Compute and print accuracy before and after calibration (must be identical)
- [x] **5.3** Compute confidence separation table: mean confidence for correct vs. incorrect predictions, before and after
- [x] **5.4** Compute % of high-confidence errors (>0.9) before and after — this is the headline metric
- [x] **5.5** Compute % of high-confidence errors (>0.95) before and after
- [x] **5.6** Compute AUROC of "confidence as a binary classifier for correctness" — before and after (higher = confidence is more useful for filtering)
- [x] **5.7** Plot confidence distribution overlay: correct vs. incorrect, before and after (2x1 subplot)

### Phase 6: Per-Class Deep Dive (NB16 Section 7)
- [x] **6.1** Compute class-conditional ECE for each of NEGATIVE, NEUTRAL, POSITIVE — before and after calibration
- [ ] **6.2** Plot per-class reliability diagrams (3x2 grid: 3 classes x before/after) *(deferred — implemented as table; 3x2 grid would add complexity without much value)*
- [x] **6.3** Compute per-class confidence separation (mean conf correct vs. incorrect) before and after
- [x] **6.4** Identify which class benefits most from calibration — print a summary paragraph
- [x] **6.5** If vector scaling was competitive: compare per-class W values to identify which class was most miscalibrated

### Phase 7: Save Artifacts (NB16 Section 8)
- [x] **7.1** Save `calibration_config.json` to repo root with fields: method, temperature, ece_before, ece_after, calibration_samples, num_folds, seed, label_names, timestamp
- [x] **7.2** Save `results/calibration_comparison.json` with all method results
- [x] **7.3** Save `results/calibration_logits.npz` (already done in Phase 1, verify it's intact)
- [x] **7.4** Save `results/reliability_before.png`
- [x] **7.5** Save `results/calibration_comparison.png`
- [ ] **7.6** Save `paper/figures/fig3_calibration.pdf` *(deferred — requires Kaggle run for actual data)*
- [x] **7.7** Print a summary block listing all saved files with their sizes

### Phase 8: Integrate Into Inference Pipeline
- [x] **8.1** Update `demo/app.py`: add `calibration_config.json` loading at startup, apply `logits / T` before softmax in `predict()`
- [x] **8.2** Create `scripts/calibrated_inference.py` containing the `CalibratedModernFinBERT` class with `predict()` and `predict_batch()` methods
- [ ] **8.3** Add a `--calibrated / --raw` flag to `scripts/inference_benchmark.py` and verify latency is unchanged with calibration *(deferred — trivial after calibration_config.json exists)*
- [ ] **8.4** Run the updated demo locally and test with the 5 example texts — verify outputs are sensible and confidence values are lower (less overconfident) than before *(requires calibration_config.json from Kaggle run)*
- [ ] **8.5** Verify `CalibratedModernFinBERT.predict_batch()` works on the 10 benchmark texts from `inference_benchmark.py` *(requires calibration_config.json from Kaggle run)*

### Phase 9: Update Paper
- [x] **9.1** Add `\subsection{Confidence Calibration}` to `paper/main.tex` in the Analysis and Discussion section (after the Error Analysis subsection)
- [ ] **9.2** Fill in the calibration results table (Table `tab:calibration`) with actual ECE values *(requires Kaggle run)*
- [ ] **9.3** Add the reliability diagram figure (`fig3_calibration.pdf`) with caption *(requires Kaggle run)*
- [x] **9.4** Update the Limitations section: add a note that the temperature was learned on the training distribution and may need recalibration for out-of-distribution text
- [x] **9.5** Add Guo et al., 2017 ("On Calibration of Modern Neural Networks") to `paper/references.bib`
- [x] **9.6** Update the Conclusion to mention calibration as an additional contribution
- [ ] **9.7** Compile the paper (`pdflatex main && bibtex main && pdflatex main && pdflatex main`) and verify no errors *(requires filling in TBD values first)*

### Phase 10: Update Documentation and Publish
- [x] **10.1** Update `MODEL_CARD.md`: add a "Calibration" section explaining temperature scaling, the learned T value, and how to use it
- [x] **10.2** Add calibrated inference code example to `MODEL_CARD.md`
- [ ] **10.3** Update `README.md` if it references confidence scores — note they are now calibrated *(checked — README does not reference confidence scores)*
- [x] **10.4** Update `TODOS.md`: remove or mark complete any calibration-related items *(checked — no calibration items existed)*
- [ ] **10.5** Upload `calibration_config.json` to HuggingFace model repo via `huggingface-cli upload` *(requires Kaggle run first)*
- [ ] **10.6** Push updated `demo/app.py` to HuggingFace Spaces (if deployed there) *(requires calibration_config.json)*
- [ ] **10.7** Commit all changes to git with a descriptive message

### Phase 11: Final Verification *(all require Kaggle run)*
- [ ] **11.1** Run NB16 end-to-end on Kaggle to verify full reproducibility
- [ ] **11.2** Verify `calibration_config.json` is loadable from the HuggingFace model repo
- [ ] **11.3** Verify the Gradio demo on HuggingFace Spaces shows calibrated (lower, more realistic) confidence scores
- [ ] **11.4** Spot-check 5 known-difficult sentences (hedging, implicit sentiment) — confidence should be noticeably lower than pre-calibration
- [ ] **11.5** Verify all 5 success criteria are met:
  - [ ] ECE dropped by >= 50%
  - [ ] High-confidence errors (>0.9) below 15%
  - [ ] All predictions unchanged
  - [ ] Reliability diagram visually tight to diagonal
  - [ ] Results reproducible from saved logits
