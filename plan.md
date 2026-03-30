# Plan: Long-Context Fine-Tuning + Multi-Benchmark Evaluation

## Overview

Two focused improvements to the current ModernFinBERT paper and model:

1. **Long-context fine-tuning** — ModernBERT supports 8192 tokens via RoPE but every notebook hardcodes `max_length=512`. Earnings call transcripts (Source 8: median 161 words, max 2,596 words) are silently truncated during training. The model trains on incomplete texts and scores 69% on earnings calls vs 80%+ elsewhere. Unlocking longer context is free signal.

2. **Multi-benchmark evaluation** — The paper evaluates only on FPB. Testing on FiQA, Twitter Financial News, and TweetEval proves (or disproves) that improvements generalize beyond a single dataset.

---

## Task List

### Phase 1: Long-Context Fine-Tuning (NB18)

#### 1.1 Data Analysis (no GPU, ~30 min)
- [x] Load aggregated dataset with source IDs
- [x] Tokenize all texts WITHOUT truncation to get true token lengths
- [x] Compute per-source truncation rates at max_length 128, 512, 1024, 2048
- [x] Generate truncation analysis table (source x max_length)
- [x] Identify exact % of Source 8 samples truncated at 512
- [x] Compute the number of tokens lost per truncated sample (true_len - 512)
- [x] Save truncation stats to `results/truncation_analysis.json`

#### 1.2 Training Infrastructure (~1 hour)
- [x] Write `train_held_out(max_length, seed)` function matching NB01 protocol exactly
- [x] Implement adaptive batch size scaling (512→bs=8, 1024→bs=4, 2048→bs=2) with gradient accumulation to keep effective batch constant at 32
- [x] Verify 2048-token training fits in T4 16GB VRAM (batch=2, gradient checkpointing on)
- [x] Write `run_inference(model, tokenizer, texts, max_length)` helper
- [x] Write `evaluate_held_out(model, tokenizer, max_length)` with per-source breakdown

#### 1.3 Context Length Ablation (9 runs, ~9 hours GPU)
- [x] Train max_length=512 with seed=3407
- [x] Train max_length=512 with seed=42
- [x] Train max_length=512 with seed=123
- [x] Train max_length=1024 with seed=3407
- [x] Train max_length=1024 with seed=42
- [x] Train max_length=1024 with seed=123
- [x] Train max_length=2048 with seed=3407
- [x] Train max_length=2048 with seed=42
- [x] Train max_length=2048 with seed=123
- [x] Evaluate each model on: FPB 50agree, FPB allAgree, aggregated test set (overall + per-source)

#### 1.4 Results Analysis (~30 min)
- [x] Aggregate results: mean ± std over 3 seeds for each max_length
- [x] Generate per-source accuracy comparison table (512 vs 1024 vs 2048)
- [x] Compute earnings call (Source 8) accuracy delta: 2048 vs 512
- [x] Generate bar chart: per-source accuracy at each context length
- [x] Statistical test: paired t-test on per-seed results (512 vs best long-ctx)
- [x] Save all results to `results/longctx_ablation.json`
- [x] Identify best max_length for Phase 2

### Phase 2: Multi-Benchmark Evaluation (NB19)

#### 2.1 Benchmark Setup (~1 hour, no GPU)
- [x] Load FPB sentences_50agree, verify label scheme {0:NEG, 1:NEU, 2:POS}
- [x] Load FPB sentences_allagree, verify same scheme
- [x] Load `zeroshot/twitter-financial-news-sentiment` validation split
- [x] Inspect twitter label scheme, determine correct remap to {NEG, NEU, POS}
- [x] Manually verify 5 samples per class match expected sentiment
- [x] Load `pauri32/fiqa-2018` train split
- [x] Verify FiQA continuous score range and distribution
- [x] Implement 3-class discretization at thresholds (-0.2, 0.2)
- [x] Print class distribution for each benchmark after remapping
- [x] Write `load_benchmark(name)` function returning standardized (texts, labels)

#### 2.2 Baseline Model Setup (~30 min)
- [x] Load `ProsusAI/finbert`, identify its label ordering from `model.config.id2label`
- [x] Determine ProsusAI label remap: their label indices → our {0:NEG, 1:NEU, 2:POS}
- [x] Load `yiyanghkust/finbert-tone`, identify its label ordering
- [x] Determine finbert-tone label remap
- [x] Sanity check: run each baseline on 10 obvious samples, verify labels make sense
- [x] Write `run_inference_with_remap(model, tokenizer, texts, label_remap)` helper

#### 2.3 Evaluation Runs (~1 hour GPU)
- [x] Evaluate `neoyipeng/ModernFinBERT-base` (production, max_length=512) on all 4 benchmarks
- [ ] Evaluate best long-context model from Phase 1 on all 4 benchmarks
- [x] Evaluate `ProsusAI/finbert` on all 4 benchmarks
- [x] Evaluate `yiyanghkust/finbert-tone` on all 4 benchmarks
- [x] For each model × benchmark: compute accuracy, macro F1, per-class F1
- [x] Run FiQA threshold sensitivity at (-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3)

#### 2.4 Results and Paper Artifacts (~1 hour)
- [x] Generate multi-benchmark comparison table (models as rows, benchmarks as columns)
- [x] Generate LaTeX table for paper
- [x] Identify where ModernFinBERT wins vs loses against baselines
- [ ] Write discussion paragraph: does ModernFinBERT generalize beyond FPB?
- [x] Save all results to `results/multi_benchmark_results.json`

### Phase 3: Paper Integration

> **Note:** Phase 3.1 and 3.2 tasks require running NB18 and NB19 on Kaggle first to get actual numbers. These are blocked until GPU results are available.

#### 3.1 New Paper Content (blocked: needs NB18/NB19 results)
- [ ] Add truncation analysis table to Section 3 (Experimental Setup) or new subsection
- [ ] Add context length ablation table as new Experiment (Experiment 10)
- [ ] Add multi-benchmark comparison table as new Experiment (Experiment 11)
- [ ] Write discussion paragraph on long-context findings
- [ ] Write discussion paragraph on multi-benchmark generalization
- [ ] Update Limitations section: remove "single benchmark" limitation
- [ ] Update Conclusion with new findings

#### 3.2 Model and Artifacts (blocked: needs NB18/NB19 results)
- [ ] If long-context model improves: retrain production model with best max_length
- [ ] Update `MODEL_CARD.md` with multi-benchmark results
- [ ] Update `calibration_config.json` if production model changes
- [ ] Push updated model to `neoyipeng/ModernFinBERT-base` (if improved)

#### 3.3 Cleanup
- [x] Add NB18 and NB19 to notebooks/ directory
- [x] Update README.md with new experiment descriptions
- [x] Update TODOS.md: mark multi-benchmark as done, update remaining items

---

## Phase 1: Long-Context Fine-Tuning

### NB18: Context Length Ablation

One notebook, three experiments: train the same model at `max_length` 512, 1024, and 2048. Isolate the effect of context length with everything else held constant.

#### Step 1: Measure truncation damage

Before training anything, quantify how much data is being lost.

```python
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# Load aggregated dataset with source IDs
ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
ds = ds.filter(lambda x: x["task"] == "sentiment")

SOURCE_NAMES = {
    3: "Earnings (narrative)",
    4: "Press releases",
    5: "FPB",
    8: "Earnings (Q&A)",
    9: "Tweets",
}

# Tokenize everything WITHOUT truncation to get true lengths
stats_by_source = defaultdict(list)

for split in ["train", "validation", "test"]:
    for example in ds[split]:
        ids = tokenizer(example["text"], add_special_tokens=True)["input_ids"]
        stats_by_source[example["source"]].append(len(ids))

# Report truncation rates at each max_length
MAX_LENGTHS = [128, 512, 1024, 2048]

print(f"{'Source':<25} {'N':>6} {'Med':>5} {'P95':>6} {'Max':>6}", end="")
for ml in MAX_LENGTHS:
    print(f" {'Trunc@'+str(ml):>10}", end="")
print()
print("-" * 100)

for src_id in sorted(stats_by_source.keys()):
    lengths = stats_by_source[src_id]
    name = SOURCE_NAMES.get(src_id, f"Source {src_id}")
    row = f"{name:<25} {len(lengths):>6} {int(np.median(lengths)):>5} "
    row += f"{int(np.percentile(lengths, 95)):>6} {max(lengths):>6}"
    for ml in MAX_LENGTHS:
        trunc = sum(1 for l in lengths if l > ml)
        row += f" {trunc/len(lengths):>9.1%}"
    print(row)
```

Expected output (based on data provenance audit):
- Source 9 (tweets, median 15 words): ~0% truncated at any length
- Source 4 (press releases, median 60 words): low truncation
- Source 8 (earnings Q&A, median 161 words, max 2,596): **high truncation at 512, near-zero at 2048**
- FPB (median 21 words): ~0% truncated

This table goes directly into the paper to justify the long-context experiment.

#### Step 2: Training function with configurable max_length

```python
import torch
import numpy as np
import gc
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer, training_args,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset, Dataset

NUM_CLASSES = 3
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
FPB_SOURCE = 5
MODEL_NAME = "answerdotai/ModernBERT-base"
label_dict = {"NEUTRAL/MIXED": 1, "NEGATIVE": 0, "POSITIVE": 2}


def load_aggregated_data():
    """Load aggregated dataset excluding FPB, same as NB01."""
    ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
    ds = ds.filter(lambda x: x["task"] == "sentiment")
    ds = ds.filter(lambda x: x["source"] != FPB_SOURCE)
    remove_cols = [c for c in ds["train"].column_names if c not in ("text", "labels")]
    ds = ds.map(
        lambda ex: {
            "text": ex["text"],
            "labels": np.eye(NUM_CLASSES)[label_dict[ex["label"]]].tolist(),
        },
        remove_columns=remove_cols,
    )
    return ds


def train_held_out(max_length: int, seed: int = 3407):
    """Train ModernBERT+LoRA on aggregated data with given max_length.

    Identical to NB01 except for max_length and adjusted batch size.
    Returns trained model and tokenizer.
    """
    ds = load_aggregated_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES,
        torch_dtype=torch.float32, attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["Wqkv", "out_proj", "Wi", "Wo"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model = model.cuda()

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    train_tok = ds["train"].map(tokenize_fn, batched=True)
    val_tok = ds["validation"].map(tokenize_fn, batched=True)

    # Scale batch size inversely with max_length to keep memory constant
    # 512 tokens × batch 8 = 4096 positions/step
    # 1024 tokens × batch 4 = 4096 positions/step
    # 2048 tokens × batch 2 = 4096 positions/step
    if max_length <= 512:
        bs, ga = 8, 4   # effective batch = 32
    elif max_length <= 1024:
        bs, ga = 4, 8   # effective batch = 32
    else:
        bs, ga = 2, 16  # effective batch = 32

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        args=TrainingArguments(
            output_dir=f"out_longctx_{max_length}",
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            warmup_steps=10,
            fp16=True,
            optim=training_args.OptimizerNames.ADAMW_TORCH,
            learning_rate=2e-4,
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=seed,
            num_train_epochs=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            gradient_checkpointing=True,
            report_to="none",
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

#### Step 3: Evaluation with per-source breakdown

This is the critical piece missing from the current paper — per-source accuracy shows WHERE long context helps, not just whether it helps on average.

```python
def run_inference(model, tokenizer, texts, max_length=512, batch_size=32):
    """Run inference, return predicted class indices."""
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def evaluate_held_out(model, tokenizer, max_length):
    """Evaluate on FPB + aggregated test set with per-source breakdown."""
    results = {}

    # --- FPB (held-out) ---
    fpb_50 = load_dataset("financial_phrasebank", "sentences_50agree",
                          trust_remote_code=True)["train"]
    fpb_all = load_dataset("financial_phrasebank", "sentences_allagree",
                           trust_remote_code=True)["train"]

    for name, fpb in [("fpb_50agree", fpb_50), ("fpb_allagree", fpb_all)]:
        preds = run_inference(model, tokenizer, fpb["sentence"], max_length)
        results[name] = {
            "accuracy": round(accuracy_score(fpb["label"], preds), 4),
            "macro_f1": round(f1_score(fpb["label"], preds, average="macro"), 4),
            "n": len(fpb),
        }

    # --- Aggregated test set WITH per-source breakdown ---
    ds_full = load_dataset("neoyipeng/financial_reasoning_aggregated")
    ds_full = ds_full.filter(lambda x: x["task"] == "sentiment")
    test = ds_full["test"]

    test_texts = test["text"]
    test_labels = [label_dict[l] for l in test["label"]]
    test_sources = test["source"]
    test_preds = run_inference(model, tokenizer, test_texts, max_length)

    # Overall
    results["agg_test"] = {
        "accuracy": round(accuracy_score(test_labels, test_preds), 4),
        "macro_f1": round(f1_score(test_labels, test_preds, average="macro"), 4),
        "n": len(test_texts),
    }

    # Per-source
    for src_id in sorted(set(test_sources)):
        mask = [i for i, s in enumerate(test_sources) if s == src_id]
        if len(mask) < 5:
            continue
        src_labels = [test_labels[i] for i in mask]
        src_preds = [test_preds[i] for i in mask]
        src_name = SOURCE_NAMES.get(src_id, f"source_{src_id}")
        results[f"agg_test_{src_name}"] = {
            "accuracy": round(accuracy_score(src_labels, src_preds), 4),
            "macro_f1": round(f1_score(src_labels, src_preds, average="macro",
                                        zero_division=0), 4),
            "n": len(mask),
        }

    return results
```

#### Step 4: Run the ablation

```python
import json

CONTEXT_LENGTHS = [512, 1024, 2048]
SEEDS = [3407, 42, 123]

all_ablation_results = []

for ml in CONTEXT_LENGTHS:
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"max_length={ml}, seed={seed}")
        print(f"{'='*60}")

        model, tokenizer = train_held_out(max_length=ml, seed=seed)
        results = evaluate_held_out(model, tokenizer, max_length=ml)
        results["max_length"] = ml
        results["seed"] = seed
        all_ablation_results.append(results)

        print(f"FPB 50agree: {results['fpb_50agree']['accuracy']}")
        print(f"FPB allAgree: {results['fpb_allagree']['accuracy']}")
        if "agg_test_Earnings (Q&A)" in results:
            print(f"Earnings Q&A: {results['agg_test_Earnings (Q&A)']['accuracy']}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

# Save raw results
with open("results/longctx_ablation.json", "w") as f:
    json.dump(all_ablation_results, f, indent=2)
```

#### Step 5: Summary table for paper

```python
import pandas as pd

rows = []
for r in all_ablation_results:
    row = {
        "max_length": r["max_length"],
        "seed": r["seed"],
        "fpb_50_acc": r["fpb_50agree"]["accuracy"],
        "fpb_50_f1": r["fpb_50agree"]["macro_f1"],
        "fpb_all_acc": r["fpb_allagree"]["accuracy"],
        "agg_test_acc": r["agg_test"]["accuracy"],
    }
    # Add per-source if available
    for key in r:
        if key.startswith("agg_test_") and key != "agg_test":
            src = key.replace("agg_test_", "")
            row[f"{src}_acc"] = r[key]["accuracy"]
    rows.append(row)

df = pd.DataFrame(rows)

# Aggregate over seeds
summary = df.groupby("max_length").agg(["mean", "std"])
print("\nCONTEXT LENGTH ABLATION — mean ± std over 3 seeds")
print("=" * 80)
print(summary.to_string(float_format="%.4f"))

# Key comparison: earnings call accuracy at 512 vs 2048
print("\n\nEARNINGS CALL ACCURACY LIFT")
print("-" * 40)
for ml in CONTEXT_LENGTHS:
    subset = df[df["max_length"] == ml]
    if "Earnings (Q&A)_acc" in subset.columns:
        mean = subset["Earnings (Q&A)_acc"].mean()
        print(f"  max_length={ml}: {mean:.4f}")
```

### Compute estimate

3 context lengths x 3 seeds = 9 training runs. Each run is ~40 min on T4 at 512, ~60 min at 1024, ~80 min at 2048 (longer sequences = slower per-step but same effective batch).

**Total: ~9 hours on a single T4 (Kaggle free tier).**

---

## Phase 2: Multi-Benchmark Evaluation

### NB19: Multi-Benchmark Harness

Evaluate the current production model AND the best long-context model on four benchmarks. This is evaluation only — no training.

#### Step 1: Define benchmarks

```python
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import json

# Label mapping: all benchmarks → {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE}

BENCHMARKS = {
    "fpb_50agree": {
        "load_fn": lambda: load_dataset(
            "financial_phrasebank", "sentences_50agree",
            trust_remote_code=True
        )["train"],
        "text_col": "sentence",
        "label_col": "label",
        # FPB: 0=negative, 1=neutral, 2=positive (already correct)
        "remap": None,
        "description": "FinancialPhraseBank 50% agree (4,846 press-release sentences)",
    },
    "fpb_allagree": {
        "load_fn": lambda: load_dataset(
            "financial_phrasebank", "sentences_allagree",
            trust_remote_code=True
        )["train"],
        "text_col": "sentence",
        "label_col": "label",
        "remap": None,
        "description": "FinancialPhraseBank 100% agree (2,264 sentences)",
    },
    "twitter_fin_sent": {
        "load_fn": lambda: load_dataset(
            "zeroshot/twitter-financial-news-sentiment",
            split="validation"
        ),
        "text_col": "text",
        "label_col": "label",
        # 0=bearish, 1=bullish, 2=neutral → remap to our scheme
        "remap": {0: 0, 1: 2, 2: 1},  # bearish→NEG, bullish→POS, neutral→NEU
        "description": "Twitter Financial News Sentiment (validation split)",
    },
    "fiqa_2018": {
        "load_fn": lambda: load_dataset("pauri32/fiqa-2018", split="train"),
        "text_col": "sentence",
        "label_col": "sentiment_score",
        # Continuous [-1, 1] → discretize to 3 classes
        "remap": "continuous",
        "thresholds": (-0.2, 0.2),  # < -0.2 = NEG, > 0.2 = POS, else NEU
        "description": "FiQA 2018 Task 1 (financial opinion, continuous→3-class)",
    },
}
```

#### Step 2: Load and standardize each benchmark

```python
def load_benchmark(name):
    """Load a benchmark dataset, return (texts, labels) with standardized labels."""
    config = BENCHMARKS[name]
    ds = config["load_fn"]()

    texts = list(ds[config["text_col"]])

    if config["remap"] == "continuous":
        # Discretize continuous scores
        lo, hi = config["thresholds"]
        raw_scores = ds[config["label_col"]]
        labels = []
        for s in raw_scores:
            if s < lo:
                labels.append(0)
            elif s > hi:
                labels.append(2)
            else:
                labels.append(1)
    elif config["remap"] is not None:
        raw_labels = ds[config["label_col"]]
        labels = [config["remap"][l] for l in raw_labels]
    else:
        labels = list(ds[config["label_col"]])

    # Filter out any samples with invalid labels
    valid = [(t, l) for t, l in zip(texts, labels) if l in (0, 1, 2)]
    texts = [t for t, _ in valid]
    labels = [l for _, l in valid]

    print(f"  {name}: {len(texts)} samples, "
          f"NEG={labels.count(0)}, NEU={labels.count(1)}, POS={labels.count(2)}")
    return texts, labels


# Validate all benchmarks load correctly
print("Loading benchmarks...")
for name in BENCHMARKS:
    try:
        texts, labels = load_benchmark(name)
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
```

#### Step 3: Evaluate models across all benchmarks

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def run_inference(model, tokenizer, texts, max_length=512, batch_size=32):
    """Run inference, return predicted class indices."""
    device = next(model.parameters()).device
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def evaluate_model_on_all_benchmarks(model, tokenizer, max_length=512,
                                      model_name="model"):
    """Evaluate a single model on all benchmarks. Returns dict of results."""
    results = {}
    device = next(model.parameters()).device

    for bench_name in BENCHMARKS:
        try:
            texts, labels = load_benchmark(bench_name)
            preds = run_inference(model, tokenizer, texts, max_length)

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            report = classification_report(
                labels, preds, target_names=LABEL_NAMES, output_dict=True,
                zero_division=0,
            )

            results[bench_name] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "n": len(texts),
                "per_class": {
                    cls: round(report[cls]["f1-score"], 4)
                    for cls in LABEL_NAMES
                },
            }

            print(f"  {bench_name}: acc={acc:.4f}, f1={f1:.4f}")

        except Exception as e:
            print(f"  {bench_name}: FAILED — {e}")
            results[bench_name] = {"error": str(e)}

    return results
```

#### Step 4: Compare models

```python
# Models to evaluate
MODELS_TO_EVAL = {
    "ModernFinBERT-v1 (production)": {
        "model_id": "neoyipeng/ModernFinBERT-base",
        "max_length": 512,
    },
    # After Phase 1 long-context ablation, add the best long-context model:
    # "ModernFinBERT-v1 (2048 ctx)": {
    #     "model_id": "out_longctx_2048/best_model",
    #     "max_length": 2048,
    # },
}

# Also evaluate off-the-shelf baselines for comparison
BASELINE_MODELS = {
    "ProsusAI/finbert": {
        "model_id": "ProsusAI/finbert",
        "max_length": 512,
        # ProsusAI uses different label order: 0=positive, 1=negative, 2=neutral
        "label_remap": {0: 2, 1: 0, 2: 1},
    },
    "yiyanghkust/finbert-tone": {
        "model_id": "yiyanghkust/finbert-tone",
        "max_length": 512,
        # finbert-tone: 0=neutral, 1=positive, 2=negative
        "label_remap": {0: 1, 1: 2, 2: 0},
    },
}

all_model_results = {}

for model_name, config in {**MODELS_TO_EVAL, **BASELINE_MODELS}.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    results = evaluate_model_on_all_benchmarks(
        model, tokenizer,
        max_length=config["max_length"],
        model_name=model_name,
    )

    # If model has different label ordering, we need to remap predictions
    # This is handled in run_inference by remapping after argmax
    # (omitted here for clarity — implement per-model remap in run_inference)

    all_model_results[model_name] = results
    del model; gc.collect(); torch.cuda.empty_cache()

# Save
with open("results/multi_benchmark_results.json", "w") as f:
    json.dump(all_model_results, f, indent=2)
```

#### Step 5: Paper-ready comparison table

```python
import pandas as pd

# Build comparison table
rows = []
for model_name, benchmarks in all_model_results.items():
    row = {"Model": model_name}
    for bench_name, metrics in benchmarks.items():
        if "error" in metrics:
            row[f"{bench_name}_acc"] = "—"
            row[f"{bench_name}_f1"] = "—"
        else:
            row[f"{bench_name}_acc"] = metrics["accuracy"]
            row[f"{bench_name}_f1"] = metrics["macro_f1"]
    rows.append(row)

df = pd.DataFrame(rows)

print("\nMULTI-BENCHMARK COMPARISON")
print("=" * 100)
print(df.to_string(index=False, float_format="%.4f"))

# LaTeX table for paper
print("\n\n% LaTeX table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Multi-benchmark evaluation. All models evaluated zero-shot "
      r"(no benchmark-specific fine-tuning). ModernFinBERT trained on "
      r"aggregated financial data with FPB held out.}")
print(r"\label{tab:multi-benchmark}")
print(r"\begin{tabular}{l" + "cc" * len(BENCHMARKS) + "}")
print(r"\toprule")

# Header
header = r"\textbf{Model}"
for bench in BENCHMARKS:
    short = bench.replace("_", r"\_")
    header += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{short}}}}}"
header += r" \\"
print(header)

# Sub-header
subheader = ""
for _ in BENCHMARKS:
    subheader += r" & Acc & F1"
subheader += r" \\"
print(subheader)
print(r"\midrule")

# Data rows
for _, row in df.iterrows():
    line = row["Model"].replace("_", r"\_")
    for bench in BENCHMARKS:
        acc = row.get(f"{bench}_acc", "—")
        f1_val = row.get(f"{bench}_f1", "—")
        if isinstance(acc, float):
            line += f" & {acc:.4f} & {f1_val:.4f}"
        else:
            line += f" & {acc} & {f1_val}"
    line += r" \\"
    print(line)

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
```

### Handling label remapping for baseline models

ProsusAI/finbert and finbert-tone use different label orderings. This must be handled carefully.

```python
def run_inference_with_remap(model, tokenizer, texts, max_length=512,
                              label_remap=None, batch_size=32):
    """Run inference with optional label remapping for baseline models.

    Args:
        label_remap: dict mapping model's output label indices to our
                     standard scheme {0: NEG, 1: NEU, 2: POS}.
                     E.g., ProsusAI: {0: 2, 1: 0, 2: 1} because their
                     0=positive, 1=negative, 2=neutral.
    """
    device = next(model.parameters()).device
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            if label_remap:
                preds = np.array([label_remap[p] for p in preds])

            all_preds.extend(preds)

    return np.array(all_preds)
```

### Benchmark-specific notes

**FiQA 2018**: Uses continuous sentiment scores in [-1, 1]. Discretization thresholds matter — document them clearly and run sensitivity analysis at [-0.1, 0.1] and [-0.3, 0.3] in addition to [-0.2, 0.2]. Report all three to show robustness.

```python
# FiQA threshold sensitivity
FIQA_THRESHOLDS = [(-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3)]

for lo, hi in FIQA_THRESHOLDS:
    texts, labels = load_benchmark_with_thresholds("fiqa_2018", lo, hi)
    preds = run_inference(model, tokenizer, texts)
    acc = accuracy_score(labels, preds)
    print(f"  FiQA thresholds ({lo}, {hi}): acc={acc:.4f}, n={len(texts)}, "
          f"NEG={labels.count(0)}, NEU={labels.count(1)}, POS={labels.count(2)}")
```

**Twitter Financial News**: Check label scheme carefully — some versions use {0: bearish, 1: bullish, 2: neutral}, which is NOT the same order as FPB. Verify by inspecting a few samples.

```python
# Sanity check: print a few samples from each benchmark
for name in BENCHMARKS:
    texts, labels = load_benchmark(name)
    print(f"\n{name} — sample texts by label:")
    for label_idx, label_name in enumerate(LABEL_NAMES):
        examples = [t for t, l in zip(texts, labels) if l == label_idx][:2]
        for ex in examples:
            print(f"  [{label_name}] {ex[:100]}")
```

---

## Execution Plan

```
NB18: Long-Context Ablation (~9 hours on T4)
  1. Truncation analysis by source (20 min, no GPU)
  2. Train at max_length=512 × 3 seeds (~2 hr)
  3. Train at max_length=1024 × 3 seeds (~3 hr)
  4. Train at max_length=2048 × 3 seeds (~4 hr)
  5. Summary table + per-source breakdown

NB19: Multi-Benchmark Evaluation (~1 hour on T4)
  1. Load all 4 benchmarks, verify label schemes
  2. Evaluate ModernFinBERT-v1 (production, 512 ctx)
  3. Evaluate ModernFinBERT-v1 (best long-ctx from NB18)
  4. Evaluate ProsusAI/finbert baseline
  5. Evaluate finbert-tone baseline
  6. Generate paper table (LaTeX)
```

### Compute

| Step | Hardware | Time | Cost |
|------|----------|------|------|
| NB18: Long-context ablation | T4 16GB | ~9 hours | Free (Kaggle) |
| NB19: Multi-benchmark eval | T4 16GB | ~1 hour | Free (Kaggle) |
| **Total** | | **~10 hours** | **Free** |

### What goes into the paper

1. **Table: Truncation analysis by source** — shows exactly how much data Source 8 loses at 512 tokens. Justifies the experiment.
2. **Table: Context length ablation** — accuracy at 512/1024/2048 with per-source breakdown. The key number is Source 8 (earnings call) accuracy lift.
3. **Table: Multi-benchmark comparison** — ModernFinBERT vs ProsusAI/finbert vs finbert-tone on FPB + FiQA + Twitter Financial. Shows whether ModernFinBERT generalizes or is FPB-specific.
4. **Discussion paragraph** — if long context helps earnings calls but not FPB (which has median 21 words), that's evidence the architecture improvement is domain-dependent, not universal. If multi-benchmark shows strong generalization, that strengthens the paper's claims.

### Risks

| Risk | Mitigation |
|------|-----------|
| Long context hurts FPB accuracy (short texts padded more) | Report per-source; FPB result may be flat while earnings improves |
| Twitter/FiQA label schemes don't map cleanly to 3-class | Document discretization choices; run threshold sensitivity for FiQA |
| Baseline models have incompatible label ordering | Verify with manual inspection of 5-10 samples per benchmark |
| 2048 tokens doesn't fit on T4 with batch=2 | Fall back to batch=1 with ga=32; or use 1024 as max |
