"""
Post-process annotated medium chunks into a final dataset.

Steps:
1. Merge all batch outputs with the annotation pool
2. Entity verification (substring matching)
3. Compute label distribution and class weights
4. Stratified train/val/test split
5. Save final dataset

Usage:
    python scripts/build_medium_dataset.py
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = Path("data/processed/entity_annotations_medium")
POOL_PATH = Path("data/processed/medium_annotation_pool.parquet")
OUTPUT_DIR = Path("data/processed")


def verify_entity(entity: str, text: str) -> bool:
    if entity in ("NONE", "MARKET"):
        return True
    text_lower = text.lower()
    if entity.lower() in text_lower:
        return True
    skip = {"the", "a", "an", "of", "and", "inc.", "inc", "corp.", "corp",
            "corporation", "company", "ltd.", "ltd", "llc", "group", "holdings",
            "co.", "co", "incorporated", "limited", "plc"}
    for word in entity.split():
        if len(word) > 2 and word.lower() not in skip and word.lower() in text_lower:
            return True
    return False


def main():
    print("Loading annotation pool...")
    pool = pd.read_parquet(POOL_PATH)
    print(f"  Pool: {len(pool)} rows")

    print("\nMerging batch outputs...")
    all_outputs = []
    batch_files = sorted(DATA_DIR.glob("batch_*_output.csv"))
    for f in batch_files:
        df = pd.read_csv(f, index_col=0)
        all_outputs.append(df)
    annotations = pd.concat(all_outputs).sort_index()
    print(f"  Annotations: {len(annotations)} rows")

    merged = pool.copy()
    for col in ["label", "entity", "entity_sentiment", "label_confidence"]:
        merged[col] = annotations[col].values

    print(f"\nRaw label distribution:")
    print(merged["entity_sentiment"].value_counts())
    print(f"\nBy source:")
    print(merged.groupby("source")["entity_sentiment"].value_counts())

    # Entity verification
    print("\nRunning entity verification...")
    hallucinated = 0
    for i in merged.index:
        if not verify_entity(merged.at[i, "entity"], merged.at[i, "text"]):
            merged.at[i, "entity"] = "NONE"
            merged.at[i, "entity_sentiment"] = merged.at[i, "label"]
            hallucinated += 1

    entity_coverage = (merged["entity"] != "NONE").mean()
    print(f"  Removed {hallucinated} hallucinated entities ({100*hallucinated/len(merged):.1f}%)")
    print(f"  Final entity coverage: {entity_coverage:.1%}")

    # Entity stats
    print(f"\nTop 20 entities:")
    print(merged["entity"].value_counts().head(20))

    # Final label distribution
    print(f"\nFinal label distribution:")
    label_counts = merged["entity_sentiment"].value_counts()
    print(label_counts)

    # Compute class weights
    total = len(merged)
    n_classes = 3
    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    counts_by_id = {label_map[k]: v for k, v in label_counts.items()}
    weights = {i: total / (n_classes * counts_by_id[i]) for i in range(n_classes)}
    print(f"\nClass weights (inverse frequency):")
    for label_name, label_id in label_map.items():
        print(f"  {label_name} (id={label_id}): weight={weights[label_id]:.4f}")

    # Stratified split
    print("\nSplitting train/val/test (80/10/10)...")
    merged["stratify_key"] = merged["source"] + "_" + merged["entity_sentiment"]

    train_df, temp_df = train_test_split(
        merged, test_size=0.2, random_state=SEED, stratify=merged["stratify_key"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["stratify_key"]
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n  {name} label distribution:")
        print(f"    {split_df['entity_sentiment'].value_counts().to_dict()}")

    # Verify no text leakage
    train_texts = set(train_df["text"].str[:200])
    val_texts = set(val_df["text"].str[:200])
    test_texts = set(test_df["text"].str[:200])
    tv_leak = train_texts & val_texts
    tt_leak = train_texts & test_texts
    vt_leak = val_texts & test_texts
    print(f"\n  Text leakage check: train∩val={len(tv_leak)}, train∩test={len(tt_leak)}, val∩test={len(vt_leak)}")

    # Save
    columns = ["text", "label", "source", "source_domain", "label_confidence",
               "entity", "entity_sentiment"]

    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        out_path = OUTPUT_DIR / f"medium_{name}.parquet"
        split_df[columns].to_parquet(out_path, index=False)
        print(f"  Saved {name} to {out_path}")

    # Save class weights
    weights_path = OUTPUT_DIR / "medium_class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  Saved class weights to {weights_path}")

    # Save combined for upload
    combined_path = OUTPUT_DIR / "medium_dataset_final.parquet"
    merged["split"] = "train"
    merged.loc[val_df.index, "split"] = "validation"
    merged.loc[test_df.index, "split"] = "test"
    merged[columns + ["split"]].to_parquet(combined_path, index=False)
    print(f"  Saved combined dataset to {combined_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
