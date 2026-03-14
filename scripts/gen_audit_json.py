import json, re, pandas as pd, numpy as np
from datasets import load_dataset

SOURCE_NAMES = {
    3: "Earnings Calls (Narrative)",
    4: "Press Releases & News",
    5: "FinancialPhraseBank",
    8: "Earnings Calls (Q&A)",
    9: "Financial Tweets",
}

ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
ds_sent = ds.filter(lambda x: x["task"] == "sentiment")

rows = []
for split in ds_sent:
    for r in ds_sent[split]:
        rows.append({
            "text": r["text"],
            "label": r["label"],
            "source": r["source"],
            "split": split,
            "prompt": r.get("prompt", ""),
            "word_count": len(str(r["text"]).split()),
            "char_count": len(str(r["text"])),
        })

df = pd.DataFrame(rows)

train_nf = df[(df["split"] == "train") & (df["source"] != 5)]
print(f"Total sentiment samples: {len(df):,}")
print(f"Training (excl FPB): {len(train_nf):,}")

# Source 4 mining
src4 = df[df["source"] == 4]
mining_kw = r"TSX|hectare|drill|assay|mining|gold|copper|zinc|mineral|ore|exploration|deposit"
is_mining = src4["text"].str.contains(mining_kw, case=False, regex=True)
print(f"Source 4 mining: {is_mining.mean():.1%}")

# Provenance table
for src in sorted(df["source"].unique()):
    sub = df[df["source"] == src]
    train_sub = sub[sub["split"] == "train"]
    neg = (sub["label"] == "NEGATIVE").mean() * 100
    neu = (sub["label"] == "NEUTRAL/MIXED").mean() * 100
    pos = (sub["label"] == "POSITIVE").mean() * 100
    print(f"Src {src}: N={len(sub)}, N_train={len(train_sub)}, NEG={neg:.1f}%, NEU={neu:.1f}%, POS={pos:.1f}%, med_words={int(sub['word_count'].median())}")

# Training set class balance
for lbl in ["NEGATIVE", "NEUTRAL/MIXED", "POSITIVE"]:
    cnt = (train_nf["label"] == lbl).sum()
    print(f"Train {lbl}: {cnt} ({cnt/len(train_nf)*100:.1f}%)")

# Export JSON
audit_results = {
    "dataset": "neoyipeng/financial_reasoning_aggregated",
    "total_sentiment_samples": len(df),
    "training_samples_excl_fpb": len(train_nf),
    "source_4_mining_pct": round(is_mining.mean(), 3),
    "sources": {}
}
for src in sorted(df["source"].unique()):
    sub = df[df["source"] == src]
    audit_results["sources"][str(src)] = {
        "name": SOURCE_NAMES[src],
        "n_total": len(sub),
        "n_train": len(sub[sub["split"] == "train"]),
        "annotation_method": "human" if src == 5 else "llm",
        "label_distribution": {
            "NEGATIVE": round((sub["label"] == "NEGATIVE").mean(), 3),
            "NEUTRAL/MIXED": round((sub["label"] == "NEUTRAL/MIXED").mean(), 3),
            "POSITIVE": round((sub["label"] == "POSITIVE").mean(), 3),
        },
        "text_length": {
            "median_words": int(sub["word_count"].median()),
            "mean_words": round(sub["word_count"].mean(), 1),
            "max_words": int(sub["word_count"].max()),
        }
    }

with open("results/data_provenance_audit.json", "w") as f:
    json.dump(audit_results, f, indent=2)
print("Saved results/data_provenance_audit.json")
