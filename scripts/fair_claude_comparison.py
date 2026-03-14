"""
Fair comparison: ModernFinBERT vs Claude Opus 4.6 on the same 723-sample test set.
Outputs results to stdout and saves JSON to results/fair_comparison_results.json.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BENCH_PATH = ROOT / "skills" / "financial-sentiment-engine" / "blind_benchmark_results.csv"
RESULTS_PATH = ROOT / "results" / "fair_comparison_results.json"

LABEL_ORDER = ["NEGATIVE", "NEUTRAL/MIXED", "POSITIVE"]
MODEL_ID_TO_BENCH_LABEL = {0: "NEGATIVE", 1: "NEUTRAL/MIXED", 2: "POSITIVE"}


def load_benchmark():
    bench = pd.read_csv(BENCH_PATH)
    assert len(bench) == 723, f"Expected 723 rows, got {len(bench)}"
    assert list(bench.columns) == ["text", "labels", "blind_predicted_label"]
    assert bench.isna().sum().sum() == 0, "Found NaN values"

    dist = bench["labels"].value_counts()
    print(f"Loaded benchmark: {len(bench)} samples")
    print(f"  NEUTRAL/MIXED: {dist.get('NEUTRAL/MIXED', 0)}")
    print(f"  POSITIVE:      {dist.get('POSITIVE', 0)}")
    print(f"  NEGATIVE:      {dist.get('NEGATIVE', 0)}")
    return bench


def tag_sources(bench):
    def normalize(t):
        return re.sub(r"\s+", " ", str(t).strip().lower())

    ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
    source_map = {}
    for item in ds["test"]:
        if item.get("task") == "sentiment":
            source_map[normalize(item["text"])] = item.get("source", -1)

    bench["source"] = bench["text"].apply(lambda t: source_map.get(normalize(t), -1))
    bench["split_type"] = bench["source"].apply(
        lambda s: "FPB (held-out)" if s == 5 else ("Non-FPB (in-distribution)" if s >= 0 else "External")
    )

    counts = bench["split_type"].value_counts()
    print(f"\nSource breakdown:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    assert counts.get("Non-FPB (in-distribution)", 0) >= 475, f"Expected ~480 non-FPB, got {counts.get('Non-FPB (in-distribution)', 0)}"
    assert counts.get("FPB (held-out)", 0) == 243
    return bench


def load_model():
    model_name = "neoyipeng/ModernFinBERT-base"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    print(f"  Device: {device}")
    print(f"  id2label: {model.config.id2label}")
    print(f"  label2id: {model.config.label2id}")
    return model, tokenizer, device


def run_inference(texts, model, tokenizer, device, batch_size=32, max_length=512):
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend([MODEL_ID_TO_BENCH_LABEL[int(pid)] for pid in pred_ids])
    return predictions


def compute_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER)
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, output_dict=True)

    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f} ({int(acc * len(y_true))}/{len(y_true)})")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER))

    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "per_class": report}


def per_subset_metrics(bench, mfb_preds, claude_preds):
    gold = bench["labels"].tolist()
    results = {}

    for split_type in ["Non-FPB (in-distribution)", "FPB (held-out)", "External", "Overall"]:
        if split_type == "Overall":
            mask = [True] * len(bench)
        else:
            mask = (bench["split_type"] == split_type).tolist()

        if sum(mask) == 0:
            continue

        sub_gold = [g for g, m in zip(gold, mask) if m]
        sub_mfb = [p for p, m in zip(mfb_preds, mask) if m]
        sub_claude = [p for p, m in zip(claude_preds, mask) if m]

        mfb_acc = accuracy_score(sub_gold, sub_mfb)
        mfb_f1 = f1_score(sub_gold, sub_mfb, average="macro", labels=LABEL_ORDER)
        claude_acc = accuracy_score(sub_gold, sub_claude)
        claude_f1 = f1_score(sub_gold, sub_claude, average="macro", labels=LABEL_ORDER)

        results[split_type] = {
            "n": sum(mask),
            "mfb_acc": round(mfb_acc, 4),
            "mfb_f1": round(mfb_f1, 4),
            "claude_acc": round(claude_acc, 4),
            "claude_f1": round(claude_f1, 4),
            "gap_acc": round(mfb_acc - claude_acc, 4),
            "gap_f1": round(mfb_f1 - claude_f1, 4),
        }

    print(f"\n{'=' * 60}")
    print("Per-Subset Breakdown")
    print(f"{'=' * 60}")
    print(f"{'Subset':<30} {'N':>4}  {'MFB Acc':>8} {'Claude':>8} {'Gap':>8}")
    print("-" * 66)
    for subset, m in results.items():
        print(f"{subset:<30} {m['n']:>4}  {m['mfb_acc']:.4f}   {m['claude_acc']:.4f}   {m['gap_acc']:+.4f}")

    return results


def per_text_type_metrics(bench, mfb_preds, claude_preds):
    def classify_text_type(text):
        if re.search(r"Question:|Answer:", text):
            return "Earnings Calls"
        if re.search(r"@|https?://|\$[A-Z]{2,5}|\bRT\b", text):
            return "Social Media"
        if re.search(r"TSX-V|TSX:|hectare|drill|assay", text, re.IGNORECASE):
            return "Mining/TSX-V"
        return "Press Release / Other"

    non_fpb = bench["split_type"] == "Non-FPB (in-distribution)"
    bench.loc[non_fpb, "text_type"] = bench.loc[non_fpb, "text"].apply(classify_text_type)

    gold = bench["labels"].tolist()
    results = {}

    print(f"\n{'=' * 60}")
    print("Non-FPB Text Type Breakdown")
    print(f"{'=' * 60}")
    print(f"{'Type':<25} {'N':>4}  {'MFB Acc':>8} {'Claude':>8} {'Gap':>8}")
    print("-" * 60)

    for text_type in sorted(bench.loc[non_fpb, "text_type"].dropna().unique()):
        mask = (bench["text_type"] == text_type).tolist()
        sub_gold = [g for g, m in zip(gold, mask) if m]
        sub_mfb = [p for p, m in zip(mfb_preds, mask) if m]
        sub_claude = [p for p, m in zip(claude_preds, mask) if m]

        mfb_acc = accuracy_score(sub_gold, sub_mfb)
        claude_acc = accuracy_score(sub_gold, sub_claude)
        gap = mfb_acc - claude_acc

        results[text_type] = {
            "n": sum(mask),
            "mfb_acc": round(mfb_acc, 4),
            "claude_acc": round(claude_acc, 4),
            "gap_acc": round(gap, 4),
        }
        print(f"{text_type:<25} {sum(mask):>4}  {mfb_acc:.4f}   {claude_acc:.4f}   {gap:+.4f}")

    return results


def main():
    bench = load_benchmark()
    bench = tag_sources(bench)

    model, tokenizer, device = load_model()

    texts = bench["text"].tolist()
    gold = bench["labels"].tolist()
    claude_preds = bench["blind_predicted_label"].tolist()

    print("\nRunning ModernFinBERT inference on 723 samples...")
    mfb_preds = run_inference(texts, model, tokenizer, device)
    print(f"Prediction distribution: {pd.Series(mfb_preds).value_counts().to_dict()}")

    mfb_metrics = compute_metrics(gold, mfb_preds, "ModernFinBERT (149M)")
    claude_metrics = compute_metrics(gold, claude_preds, "Claude Opus 4.6 + skill")

    subset_results = per_subset_metrics(bench, mfb_preds, claude_preds)
    text_type_results = per_text_type_metrics(bench, mfb_preds, claude_preds)

    print(f"\n{'=' * 60}")
    print("SUMMARY — Fair Comparison on Same 723 Samples")
    print(f"{'=' * 60}")
    print(f"ModernFinBERT: {mfb_metrics['accuracy']:.4f} acc, {mfb_metrics['macro_f1']:.4f} F1")
    print(f"Claude Opus:   {claude_metrics['accuracy']:.4f} acc, {claude_metrics['macro_f1']:.4f} F1")
    print(f"Gap:           {mfb_metrics['accuracy'] - claude_metrics['accuracy']:+.4f} acc, "
          f"{mfb_metrics['macro_f1'] - claude_metrics['macro_f1']:+.4f} F1")

    output = {
        "test_set": "blind_benchmark_results.csv (723 samples)",
        "modernfinbert": mfb_metrics,
        "claude": claude_metrics,
        "per_subset": subset_results,
        "per_text_type": text_type_results,
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
