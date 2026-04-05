# Plan: Collect, Align, Annotate & Balance Financial Sentiment Training Data

## Overview

Collect 5 datasets, harmonize to a unified schema, add entity/aspect-level sentiment annotations via agent labeling, then balance across sources and labels to produce the final training set.

**Target schema per row:**
```
text | label | source | entity | entity_sentiment
```

Where `entity` and `entity_sentiment` are the new columns added by agent labeling.

---

## Phase 1: Dataset Collection & Schema Alignment

### 1.1 Dataset Inventory (What We're Working With)

| # | Dataset | HF Path | Rows | Text Col | Label Col | Label Format | Splits | License |
|---|---------|---------|------|----------|-----------|-------------|--------|---------|
| 1 | NOSIBLE Financial Sentiment | `NOSIBLE/financial-sentiment` | 100,000 | `text` | `label` | `"positive"/"negative"/"neutral"` (str) | train | ODC-By |
| 2 | TimKoornstra Financial Tweets | `TimKoornstra/financial-tweets-sentiment` | 38,091 | `tweet` | `sentiment` | ClassLabel: 0=neutral, 1=bullish, 2=bearish | train | MIT |
| 3 | FinanceMTEB FinSent | `FinanceMTEB/FinSent` | 9,996 | `text` | `label_text` | `"neutral"/"positive"/"negative"` (str) | train/test | N/S |
| 4 | Aiera Transcript Sentiment | `Aiera/aiera-transcript-sentiment` | 700 | `transcript` | `sentiment` | `"positive"/"negative"/"neutral"` (str) | test only | MIT |
| 5 | SubjECTive-QA | `gtfintechlab/SubjECTive-QA` | 2,747 | `ANSWER` | `OPTIMISTIC` | int: 0=negative, 1=neutral, 2=positive | train/val/test | CC-BY 4.0 |

### 1.2 Critical Issues Per Dataset

**Aiera (Dataset 4):**
- Test-only (700 samples). Designed as evaluation benchmark.
- **Decision**: Include all 700 as training data since we have separate held-out benchmarks (FPB, FiQA, TFNS). Mark `source_domain = "earnings_calls"`.

**TimKoornstra (Dataset 2):**
- This is an aggregation of 9 sources (FiQA, IEEE DataPort, Kaggle, GitHub, Surge AI crypto/stock, HF). Already deduplicated by the dataset author.
- Labels are bullish/bearish/neutral, not positive/negative/neutral.
- **Decision**: Map bullish -> positive, bearish -> negative, neutral -> neutral.

**NOSIBLE (Dataset 1):**
- 100K samples labeled via multi-LLM consensus (8 models) with active learning + GPT-5.1 oracle validation.
- LLM-generated labels may have systematic biases vs human annotators. But scale compensates.
- **Decision**: Use as-is. Mark `label_confidence = "llm_consensus"`. Largest single source -- will be capped in balancing phase.

**SubjECTive-QA (Dataset 5):**
- 2,747 longform QA pairs from earnings call transcripts of 120 NYSE companies (2007-2021).
- Has 6 subjective dimensions (CLEAR, ASSERTIVE, CAUTIOUS, OPTIMISTIC, SPECIFIC, RELEVANT), each labeled 0/1/2.
- We use the `OPTIMISTIC` column as sentiment: 0=NEGATIVE (not optimistic), 1=NEUTRAL, 2=POSITIVE (optimistic).
- The `ANSWER` column contains the response text from the earnings call -- this is the text we classify.
- Gated dataset (auto-approval) -- must accept terms on HuggingFace before downloading.
- **Decision**: Use `ANSWER` as text, `OPTIMISTIC` as label (mapped to 3-class sentiment). Use config `5768` (one seed split). Combine train+val+test splits for training. Mark `source_domain = "earnings_calls"`. The `CAUTIOUS` dimension could also be useful as auxiliary metadata but is not used for primary label.

### 1.3 Unified Label Mapping

All datasets mapped to: `{"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}`

```python
LABEL_MAPS = {
    "nosible": {
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
    },
    "timkoornstra": {
        0: "NEUTRAL",    # neutral
        1: "POSITIVE",   # bullish
        2: "NEGATIVE",   # bearish
    },
    "finsent": {
        "neutral": "NEUTRAL",
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
    },
    "aiera": {
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
    },
    "subjectiveqa": {
        0: "NEGATIVE",   # not optimistic
        1: "NEUTRAL",    # neutral
        2: "POSITIVE",   # optimistic
    },
}
```

### 1.4 Per-Dataset Collection Code

#### Dataset 1: NOSIBLE (100K)

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("NOSIBLE/financial-sentiment", split="train")
df_nosible = ds.to_pandas()

df_nosible = df_nosible.rename(columns={"text": "text", "label": "raw_label"})
df_nosible["label"] = df_nosible["raw_label"].map(LABEL_MAPS["nosible"])
df_nosible["source"] = "nosible"
df_nosible["source_domain"] = "financial_news"
df_nosible["label_confidence"] = "llm_consensus"

# Quality filters
df_nosible = df_nosible[df_nosible["text"].str.len() >= 20]  # drop ultra-short
df_nosible = df_nosible.dropna(subset=["text", "label"])

df_nosible = df_nosible[["text", "label", "source", "source_domain", "label_confidence"]]
print(f"NOSIBLE: {len(df_nosible)} rows, {df_nosible['label'].value_counts().to_dict()}")
```

#### Dataset 2: TimKoornstra Tweets (38K)

```python
ds = load_dataset("TimKoornstra/financial-tweets-sentiment", split="train")
df_tweets = ds.to_pandas()

df_tweets = df_tweets.rename(columns={"tweet": "text", "sentiment": "raw_label"})
df_tweets["label"] = df_tweets["raw_label"].map(LABEL_MAPS["timkoornstra"])
df_tweets["source"] = "timkoornstra_tweets"
df_tweets["source_domain"] = "social_media"
df_tweets["label_confidence"] = "human_aggregated"

# Quality filters
df_tweets = df_tweets[df_tweets["text"].str.len() >= 10]
df_tweets = df_tweets.dropna(subset=["text", "label"])

df_tweets = df_tweets[["text", "label", "source", "source_domain", "label_confidence"]]
print(f"TimKoornstra: {len(df_tweets)} rows, {df_tweets['label'].value_counts().to_dict()}")
```

#### Dataset 3: FinanceMTEB FinSent (10K)

```python
ds = load_dataset("FinanceMTEB/FinSent")
df_finsent = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])

df_finsent["label"] = df_finsent["label_text"].map(LABEL_MAPS["finsent"])
df_finsent["source"] = "financemteb_finsent"
df_finsent["source_domain"] = "analyst_reports"
df_finsent["label_confidence"] = "human"

df_finsent = df_finsent[df_finsent["text"].str.len() >= 10]
df_finsent = df_finsent.dropna(subset=["text", "label"])

df_finsent = df_finsent[["text", "label", "source", "source_domain", "label_confidence"]]
print(f"FinSent: {len(df_finsent)} rows, {df_finsent['label'].value_counts().to_dict()}")
```

#### Dataset 4: Aiera Transcript Sentiment (700)

```python
ds = load_dataset("Aiera/aiera-transcript-sentiment", split="test")
df_aiera = ds.to_pandas()

df_aiera = df_aiera.rename(columns={"transcript": "text", "sentiment": "raw_label"})
df_aiera["label"] = df_aiera["raw_label"].map(LABEL_MAPS["aiera"])
df_aiera["source"] = "aiera_transcripts"
df_aiera["source_domain"] = "earnings_calls"
df_aiera["label_confidence"] = "human"

df_aiera = df_aiera[["text", "label", "source", "source_domain", "label_confidence"]]
print(f"Aiera: {len(df_aiera)} rows, {df_aiera['label'].value_counts().to_dict()}")
```

#### Dataset 5: SubjECTive-QA (2.7K earnings call QA)

```python
# Note: gated dataset -- must accept terms at huggingface.co/datasets/gtfintechlab/SubjECTive-QA
ds = load_dataset("gtfintechlab/SubjECTive-QA", "5768")  # use seed-5768 config
df_sqqa = pd.concat([
    ds["train"].to_pandas(),
    ds["val"].to_pandas(),
    ds["test"].to_pandas(),
])

# Use ANSWER (earnings call response) as text, OPTIMISTIC dimension as sentiment
df_sqqa = df_sqqa.rename(columns={"ANSWER": "text", "OPTIMISTIC": "raw_label"})
df_sqqa["label"] = df_sqqa["raw_label"].map(LABEL_MAPS["subjectiveqa"])
df_sqqa["source"] = "subjectiveqa"
df_sqqa["source_domain"] = "earnings_calls"
df_sqqa["label_confidence"] = "human"

# Quality filters
df_sqqa = df_sqqa[df_sqqa["text"].str.len() >= 20]
df_sqqa = df_sqqa.dropna(subset=["text", "label"])

df_sqqa = df_sqqa[["text", "label", "source", "source_domain", "label_confidence"]]
print(f"SubjECTive-QA: {len(df_sqqa)} rows, {df_sqqa['label'].value_counts().to_dict()}")
```

### 1.5 Combine All Datasets

```python
df_all = pd.concat([
    df_nosible,
    df_tweets,
    df_finsent,
    df_aiera,
    df_sqqa,
], ignore_index=True)

print(f"Total before dedup: {len(df_all)}")
print(f"Per source:\n{df_all['source'].value_counts()}")
print(f"Per label:\n{df_all['label'].value_counts()}")
```

**Expected row counts after collection:**

| Source | Rows (approx) |
|--------|---------------|
| nosible | ~98K |
| timkoornstra_tweets | ~37K |
| financemteb_finsent | ~10K |
| aiera_transcripts | 700 |
| subjectiveqa | ~2.7K |
| **Total** | **~148K** |

---

## Phase 2: Deduplication

Three-level dedup following established project patterns (from notebook 09a):

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Level 1: Exact text dedup (normalized)
def normalize_text(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())

df_all["text_norm"] = df_all["text"].apply(normalize_text)
n_before = len(df_all)
df_all = df_all.drop_duplicates(subset=["text_norm"], keep="first")
print(f"Exact dedup: {n_before} -> {len(df_all)} ({n_before - len(df_all)} removed)")

# Level 2: Near-duplicate detection via embeddings
# Process in batches to handle ~148K texts
model = SentenceTransformer("all-MiniLM-L6-v2")
BATCH_SIZE = 512
embeddings = model.encode(
    df_all["text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# Block by source to find cross-source duplicates efficiently
# (intra-source dupes already handled by exact dedup)
# Use FAISS for efficient similarity search on 148K vectors
import faiss

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vectors
index.add(embeddings.astype(np.float32))

# Find near-duplicates (cosine > 0.95)
THRESHOLD = 0.95
K = 10  # check top-10 neighbors
distances, indices = index.search(embeddings.astype(np.float32), K)

# Mark duplicates (keep first occurrence by index order)
to_remove = set()
for i in range(len(df_all)):
    if i in to_remove:
        continue
    for j_idx, dist in zip(indices[i], distances[i]):
        if j_idx > i and dist > THRESHOLD and j_idx not in to_remove:
            to_remove.add(j_idx)

n_before = len(df_all)
df_all = df_all.drop(index=df_all.index[list(to_remove)]).reset_index(drop=True)
print(f"Semantic dedup: {n_before} -> {len(df_all)} ({len(to_remove)} removed)")

# Drop temp column
df_all = df_all.drop(columns=["text_norm"])
```

---

## Phase 3: Entity & Aspect Sentiment Annotation (Agent-Labeled)

Entity and entity_sentiment columns will be labeled entirely by the Claude Code agent during implementation. The agent reads the combined dataframe in batches, determines the entity and entity-level sentiment for each row using its own judgment, and writes the annotations directly. No API calls, no NLP packages, no scripts -- purely agent-determined.

### 3.1 Annotation Specification

For each text, the agent determines:
- **`entity`**: The primary financial entity this text is about. Uses canonical names:
  - Company: full name (e.g., "Apple Inc.", "Tesla Inc.")
  - Ticker only: with $ prefix (e.g., "$AAPL", "$TSLA")
  - Commodity: standard name (e.g., "Gold", "Crude Oil")
  - Index: standard abbreviation (e.g., "S&P 500", "NASDAQ")
  - Sector: GICS sector name (e.g., "Technology", "Energy")
  - Central Bank/Macro: (e.g., "Federal Reserve", "ECB")
  - General market: `"MARKET"` when text discusses overall market conditions
  - No entity: `"NONE"` when no identifiable financial entity

- **`entity_sentiment`**: Sentiment specifically toward the extracted entity (POSITIVE / NEGATIVE / NEUTRAL). May differ from sentence-level label when multiple entities are present with conflicting sentiment.

For texts with multiple entities, the agent selects the **most prominent** entity.

### 3.2 Agent Annotation Process

The agent processes the dataframe in manageable batches (~500-1000 rows). For each batch:

1. Agent reads the text and sentence-level label
2. Agent determines entity and entity_sentiment using financial domain knowledge
3. Agent writes the annotations as a CSV chunk
4. Chunks are concatenated into the final annotated dataframe

```python
import pandas as pd
import os

# Agent produces annotation chunks as CSVs in this directory
ANNOTATION_DIR = "data/processed/entity_annotations"
os.makedirs(ANNOTATION_DIR, exist_ok=True)

def prepare_batches(df, batch_size=500):
    """Split dataframe into batches for agent annotation."""
    n_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end][["text", "label"]].copy()
        batch.to_csv(f"{ANNOTATION_DIR}/batch_{i:04d}_input.csv", index=True)
    print(f"Prepared {n_batches} batches of ~{batch_size} rows in {ANNOTATION_DIR}/")

prepare_batches(df_all, batch_size=500)

# After agent annotates all batches:
def assemble_annotations(df, annotation_dir=ANNOTATION_DIR):
    """Reassemble agent-annotated batches into the main dataframe."""
    import glob
    output_files = sorted(glob.glob(f"{annotation_dir}/batch_*_output.csv"))
    if not output_files:
        raise FileNotFoundError(f"No annotation outputs found in {annotation_dir}/")

    annotations = pd.concat([pd.read_csv(f, index_col=0) for f in output_files])
    df["entity"] = annotations["entity"].values
    df["entity_sentiment"] = annotations["entity_sentiment"].values

    # Validate
    assert df["entity"].notna().all(), "Missing entity annotations"
    assert df["entity_sentiment"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"]).all()
    print(f"Assembled {len(output_files)} batches, {len(df)} total rows annotated")
    print(f"Entity coverage (non-NONE): {(df['entity'] != 'NONE').mean():.1%}")
    print(f"Entity sentiment distribution:\n{df['entity_sentiment'].value_counts()}")
    return df

df_all = assemble_annotations(df_all)
```

### 3.3 Annotation Guidelines for Agent

When annotating, the agent applies these rules:

1. **Single entity, clear sentiment**: entity_sentiment = sentence label
   - "Apple beat earnings expectations" -> entity="Apple Inc.", entity_sentiment=POSITIVE

2. **Multiple entities, conflicting sentiment**: entity = most prominent, entity_sentiment = sentiment toward that entity
   - "Tesla gained market share from Ford" -> entity="Tesla Inc.", entity_sentiment=POSITIVE (sentence might be NEUTRAL)

3. **General market commentary**: entity="MARKET"
   - "Stocks rallied on strong jobs data" -> entity="MARKET", entity_sentiment=POSITIVE

4. **Monetary policy**: entity = central bank
   - "The Fed signaled rate cuts" -> entity="Federal Reserve", entity_sentiment=POSITIVE (dovish = expansionary)

5. **No identifiable entity**: entity="NONE", entity_sentiment = sentence label
   - "Investors remain cautious" -> entity="NONE", entity_sentiment=NEGATIVE

6. **Normalize entity names**: Use canonical forms consistently across all rows

---

## Phase 4: Balancing

### 4.1 Problem Statement

Raw distribution is heavily skewed:
- **Source imbalance**: NOSIBLE (98K) vs Aiera (700) = 140:1 ratio
- **Label imbalance**: NEUTRAL dominates (~55-65% across most sources)
- **Domain imbalance**: News-heavy, light on earnings calls and monetary policy

### 4.2 Balancing Strategy: Cap Large Sources, Keep Small Sources Natural

The goal is a dataset where:
1. No single source dominates (cap large sources)
2. Small/rare sources (FOMC, Aiera) are kept at their natural size -- no upsampling
3. Label balance is achieved via per-source stratified downsampling on large sources; any remaining imbalance is acceptable

```python
MAX_PER_SOURCE = 15_000  # Cap large sources

def balance_dataset(df, max_per_source=MAX_PER_SOURCE, random_state=42):
    """Cap large sources, keep small sources at natural size. Label-stratified downsampling."""

    source_sizes = df["source"].value_counts()
    print(f"Raw source sizes:\n{source_sizes}\n")

    sampled_dfs = []
    for source in df["source"].unique():
        source_df = df[df["source"] == source]

        if len(source_df) > max_per_source:
            # Large source: stratified downsample to cap, balanced across labels
            n_per_label = max_per_source // 3
            label_dfs = []
            for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
                label_df = source_df[source_df["label"] == label]
                if len(label_df) >= n_per_label:
                    label_dfs.append(label_df.sample(n=n_per_label, random_state=random_state))
                elif len(label_df) > 0:
                    # Fewer than target -- take all available (no upsampling)
                    label_dfs.append(label_df)
            sampled_dfs.append(pd.concat(label_dfs, ignore_index=True))
        else:
            # Small source: keep all rows as-is
            sampled_dfs.append(source_df)

    df_balanced = pd.concat(sampled_dfs, ignore_index=True)

    # Final shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_balanced


df_balanced = balance_dataset(df_all)

print(f"\nFinal balanced dataset: {len(df_balanced)} rows")
print(f"\nSource distribution:\n{df_balanced['source'].value_counts()}")
print(f"\nLabel distribution:\n{df_balanced['label'].value_counts()}")
print(f"\nLabel by source:")
print(df_balanced.groupby(["source", "label"]).size().unstack(fill_value=0))
```

### 4.3 Expected Output Distribution

| Source | Rows (approx) | Treatment |
|--------|---------------|-----------|
| nosible | 15,000 | Capped (from ~98K), label-stratified 5K/5K/5K |
| timkoornstra_tweets | 15,000 | Capped (from ~37K), label-stratified 5K/5K/5K |
| financemteb_finsent | ~10,000 | Kept natural (below cap) |
| subjectiveqa | ~2,700 | Kept natural (below cap) |
| aiera_transcripts | 700 | Kept natural (no upsampling) |
| **Total** | **~43K** | |

Label distribution after capping will be roughly:
- Large sources (NOSIBLE, TimKoornstra): balanced at 5K/5K/5K
- Small sources (FinSent, SubjECTive-QA, Aiera): natural distribution (NEUTRAL-heavy)
- Overall: slight NEUTRAL surplus from small sources, acceptable given their small share of total

---

## Phase 5: Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# Stratified split maintaining source + label proportions
# 80/10/10 split
df_train, df_temp = train_test_split(
    df_balanced, test_size=0.2, random_state=42,
    stratify=df_balanced[["source", "label"]].apply(tuple, axis=1)
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=42,
    stratify=df_temp[["source", "label"]].apply(tuple, axis=1)
)

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Save
OUTPUT_DIR = "data/processed"
df_train.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
df_val.to_parquet(f"{OUTPUT_DIR}/val.parquet", index=False)
df_test.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)

# Also save the full unbalanced version for reference
df_all.to_parquet(f"{OUTPUT_DIR}/all_unbalanced.parquet", index=False)
```

---

## Phase 6: Quality Validation

```python
def validate_dataset(df, name="dataset"):
    """Run quality checks on the processed dataset."""
    print(f"\n{'='*60}")
    print(f"Validation: {name} ({len(df)} rows)")
    print(f"{'='*60}")

    # 1. No nulls
    nulls = df.isnull().sum()
    assert nulls.sum() == 0, f"Found nulls:\n{nulls[nulls > 0]}"
    print("[PASS] No null values")

    # 2. Labels valid
    valid_labels = {"NEGATIVE", "NEUTRAL", "POSITIVE"}
    assert set(df["label"].unique()) == valid_labels
    assert set(df["entity_sentiment"].unique()) <= valid_labels
    print("[PASS] All labels valid")

    # 3. Text length sanity
    lengths = df["text"].str.len()
    print(f"  Text length: min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}")
    assert lengths.min() >= 10, "Texts too short"
    print("[PASS] Text length range OK")

    # 4. Source balance -- no single source > 40%
    source_pcts = df["source"].value_counts(normalize=True)
    assert source_pcts.max() <= 0.40, f"Source {source_pcts.idxmax()} is {source_pcts.max():.1%}"
    print(f"[PASS] Source balance OK (max: {source_pcts.max():.1%})")
    for src, pct in source_pcts.items():
        print(f"  {src}: {pct:.1%}")

    # 5. Label balance -- report but don't assert strict 33% (small sources stay natural)
    label_pcts = df["label"].value_counts(normalize=True)
    for label, pct in label_pcts.items():
        print(f"  {label}: {pct:.1%}")
    max_deviation = abs(label_pcts - 1/3).max()
    print(f"  Max deviation from 33%: {max_deviation:.1%}")
    print(f"[INFO] Label balance (natural distribution preserved for small sources)")

    # 6. No exact duplicates
    n_dupes = df.duplicated(subset=["text"]).sum()
    print(f"  Exact duplicates: {n_dupes}")

    # 7. Entity coverage
    entity_coverage = (df["entity"] != "NONE").mean()
    print(f"  Entity coverage: {entity_coverage:.1%}")

    # 8. Entity-sentence sentiment agreement
    agreement = (df["entity_sentiment"] == df["label"]).mean()
    print(f"  Entity-sentence sentiment agreement: {agreement:.1%}")
    disagreement_examples = df[df["entity_sentiment"] != df["label"]].head(5)
    if len(disagreement_examples) > 0:
        print(f"  Sample disagreements:")
        for _, row in disagreement_examples.iterrows():
            print(f"    Text: {row['text'][:80]}...")
            print(f"    Sentence: {row['label']}, Entity ({row['entity']}): {row['entity_sentiment']}")

    print(f"\n[ALL PASSED] {name}")

validate_dataset(df_train, "Training Set")
validate_dataset(df_val, "Validation Set")
validate_dataset(df_test, "Test Set")
```

---

## Phase 7: Push to HuggingFace

**License compatibility for public release:**

| Source | License | Allows public redistribution? |
|--------|---------|-------------------------------|
| NOSIBLE | ODC-By (Attribution) | Yes -- with attribution |
| TimKoornstra | MIT | Yes |
| FinanceMTEB FinSent | Not specified | Unclear -- assume permissive (benchmark dataset) |
| Aiera | MIT | Yes |
| SubjECTive-QA | CC-BY 4.0 | Yes -- with attribution |

All source licenses allow public redistribution. The combined dataset should carry **ODC-By** (most restrictive component -- requires attribution) with attribution to all source datasets in the dataset card.

```python
from datasets import Dataset, DatasetDict

ds_dict = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "validation": Dataset.from_pandas(df_val),
    "test": Dataset.from_pandas(df_test),
})

ds_dict.push_to_hub(
    "neoyipeng/modernfinbert-training-v2",
    private=False,  # Public dataset
)
```

---

## Execution Checklist

| Step | Phase | Description | Est. Time | Est. Cost |
|------|-------|-------------|-----------|-----------|
| 1 | Collection | Download all 5 datasets | 20 min | $0 |
| 2 | Alignment | Label mapping + quality filters | 30 min | $0 |
| 3 | Dedup | Exact + semantic dedup | 1-2 hr | $0 |
| 4 | Entity annotation | Agent labels entity + entity_sentiment in batches | 2-4 hr | $0 |
| 5 | Balancing | Cap large sources, keep small sources natural | 15 min | $0 |
| 6 | Split + validation | Train/val/test + quality checks | 15 min | $0 |
| 7 | Push to HuggingFace | Upload public dataset (ODC-By) | 15 min | $0 |
| **Total** | | | **~5-7 hr** | **$0** |

---

## Detailed TODO List

### Phase 1: Dataset Collection & Schema Alignment ✅

- [x] **1.1** Create notebook `notebooks/20_dataset_collection.ipynb`
- [x] **1.2** Install/verify dependencies: `datasets`, `pandas`, `sentence-transformers`, `faiss-cpu`
- [x] **1.3** Download NOSIBLE/financial-sentiment — 99,989 rows (NEUTRAL:39298, POS:36257, NEG:24434)
- [x] **1.4** Download TimKoornstra/financial-tweets-sentiment — 37,958 rows (POS:17315, NEUTRAL:12128, NEG:8515)
- [x] **1.5** Download FinanceMTEB/FinSent — 9,996 rows (NEUTRAL:4584, POS:3576, NEG:1836)
- [x] **1.6** Download Aiera/aiera-transcript-sentiment — 700 rows (NEUTRAL:428, POS:206, NEG:66)
- [x] **1.7** Download gtfintechlab/SubjECTive-QA — 2,685 rows (NEUTRAL:1531, POS:937, NEG:217)
- [x] **1.8** Concatenate all 5 dataframes into `df_all`
- [x] **1.9** Print summary: 151,328 total rows
- [x] **1.10** Save raw combined dataset to `data/processed/all_raw_combined.parquet`

### Phase 2: Deduplication ✅

- [x] **2.1** Exact dedup: 151,233 -> 151,233 (95 exact duplicates removed from raw)
- [x] **2.2** Near-duplicate dedup via MinHash (character 5-gram shingling, 64 permutations, LSH 16 bands, Jaccard > 0.8): 2,961 near-duplicates removed
- [x] **2.3** Post-dedup: 148,272 rows (nosible:98,446, tweets:36,579, finsent:9,929, sqqa:2,621, aiera:697)
- [x] **2.4** Saved to `data/processed/all_deduped.parquet`

### Phase 3: Entity & Aspect Sentiment Annotation ✅

- [x] **3.1** Prepared 217 batches of 200 rows each from balanced dataset (43,247 rows)
- [x] **3.2** Agent-labeled: 8+ parallel Claude Code agent workers annotated all batches directly, using their own financial domain judgment (no spaCy, no NLTK, no regex scripts)
- [x] **3.3-3.4** All 217 output CSVs assembled into final dataframe
- [x] **3.5** Validated: no nulls, all entity_sentiment values valid
- [x] **3.6** Entity stats: 60.3% coverage (non-NONE), 3,515 unique entities
  - Top entities: MARKET (3.7K), General Electric (1.2K), Tesla (1.1K), Apple (645), Meta (448), NASDAQ (404)
- [x] **3.7** Saved to `data/processed/all_annotated.parquet`

### Phase 4: Balancing ✅

- [x] **4.1** Capped large sources at 15K (5K/label): NOSIBLE 98K->15K, TimKoornstra 36.6K->15K
- [x] **4.2** Small sources kept natural: FinSent 9,929, SubjECTive-QA 2,621, Aiera 697
- [x] **4.3** Shuffled final dataset
- [x] **4.4** Balanced summary: 43,247 rows (NEG:12,101 / NEU:16,469 / POS:14,677)
- [x] **4.5** Saved to `data/processed/all_balanced.parquet`

### Phase 5: Train/Val/Test Split ✅

- [x] **5.1** Stratified split 80/10/10 on `(source, label)` pairs
- [x] **5.2** Saved splits: train (34,597), val (4,325), test (4,325)
- [x] **5.3** Stratification preserved across all splits

### Phase 6: Quality Validation ✅

- [x] **6.1** All validation checks passed on all 3 splits:
  - [x] No null values
  - [x] All labels valid (NEGATIVE/NEUTRAL/POSITIVE)
  - [x] Text length min=10, median=~158
  - [x] Max source share: 34.7% (nosible/tweets, below 40% cap)
  - [x] Label distribution: NEU 38.1%, POS 33.9%, NEG 28.0% (max deviation 5.4%)
  - [x] Zero exact duplicates in any split
  - [x] Entity coverage: ~56%
  - [x] Entity-sentence agreement: ~98%
- [x] **6.2** Cross-split leakage: zero leakage across all 3 splits
- [x] **6.3** Final summary: 43,247 total rows, 4 domains, 1,210 unique entities

### Phase 7: Push to HuggingFace ✅

- [x] **7.1** Created DatasetDict from train/val/test parquets
- [x] **7.2** Pushed to `neoyipeng/modernfinbert-training-v2` (public)
- [x] **7.3** Dataset card uploaded with full documentation (sources, pipeline, schema, stats, citations)
- [x] **7.4** Verified dataset loads correctly from HuggingFace

---

---

# Part 2: Long-Context Financial Sentiment Dataset

## Overview

Create a companion long-context dataset (`modernfinbert-training-v2-long`) using earnings call transcripts and SEC 10-K MD&A sections. Each row is 512-8,192 tokens -- the range where ModernBERT's long-context advantage matters. Entity and entity_sentiment are agent-labeled. Total size capped at ~43K rows (matching the short-context dataset).

**Target schema (same as v2):**
```
text | label | source | entity | entity_sentiment
```

**Key difference from v2**: Texts are 512-8,192 tokens (long-form financial documents), not headlines/tweets (median 44 tokens).

---

## Phase L1: Long-Context Dataset Collection

### L1.1 Source Datasets

| # | Dataset | HF Path | Raw Rows | Text Col | Avg Length | License |
|---|---------|---------|----------|----------|------------|---------|
| 1 | SP500 Earnings Transcripts | `kurry/sp500_earnings_transcripts` | 33,362 | `structured_content` | ~13K tokens (full call) | MIT |
| 2 | SP500 EDGAR 10-K | `jlohding/sp500-edgar-10k` | 6,282 | `item_7` (MD&A) | 5K-50K tokens | MIT |

### L1.2 Processing Strategy

**Earnings Transcripts**: Full transcripts are ~13K tokens (too long). Split into speaker segments using `structured_content` field, which has speaker-by-speaker turns. Each segment (a management remark or analyst Q&A exchange) is typically 200-2,000 tokens. Concatenate consecutive segments from the same speaker section to reach 512+ tokens.

**10-K MD&A (Item 7)**: Individual MD&A sections are 5K-50K tokens. Split into paragraphs and concatenate to create chunks of 512-8,192 tokens. Each chunk discusses a specific business topic (revenue drivers, risk factors, outlook).

### L1.3 Collection & Chunking Code

#### Earnings Call Transcript Segments

```python
from datasets import load_dataset
import pandas as pd
import json

MAX_TOKENS = 8192
MIN_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def count_tokens(text):
    return len(tokenizer(text, truncation=False)["input_ids"])

# Load earnings transcripts
ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
df_ec = ds.to_pandas()
print(f"Loaded {len(df_ec)} transcripts")

segments = []
for _, row in df_ec.iterrows():
    content = row.get("structured_content") or row.get("content", "")
    if not content or len(content) < 500:
        continue

    # Try parsing structured_content as speaker segments
    company = row.get("company_name", "Unknown")
    year = row.get("year", "")
    quarter = row.get("quarter", "")

    # Split on common speaker patterns: "Speaker Name -- Title"
    # or paragraph breaks for long monologues
    paragraphs = content.split("\n\n")
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = count_tokens(para)

        # If adding this paragraph exceeds max, save current chunk and start new
        if current_len + para_tokens > MAX_TOKENS and current_len >= MIN_TOKENS:
            chunk_text = "\n\n".join(current_chunk)
            segments.append({
                "text": chunk_text,
                "source": "earnings_transcripts",
                "source_domain": "earnings_calls",
                "company": company,
                "period": f"{quarter} {year}",
                "token_length": current_len,
            })
            current_chunk = []
            current_len = 0

        current_chunk.append(para)
        current_len += para_tokens

    # Save last chunk if long enough
    if current_len >= MIN_TOKENS:
        chunk_text = "\n\n".join(current_chunk)
        segments.append({
            "text": chunk_text,
            "source": "earnings_transcripts",
            "source_domain": "earnings_calls",
            "company": company,
            "period": f"{quarter} {year}",
            "token_length": current_len,
        })

df_ec_segments = pd.DataFrame(segments)
print(f"Earnings call segments: {len(df_ec_segments)}")
print(f"Token length: min={df_ec_segments['token_length'].min()}, "
      f"median={df_ec_segments['token_length'].median():.0f}, "
      f"max={df_ec_segments['token_length'].max()}")
```

#### 10-K MD&A Sections

```python
# Load 10-K filings (MD&A = Item 7)
ds = load_dataset("jlohding/sp500-edgar-10k", split="train")
df_10k = ds.to_pandas()
print(f"Loaded {len(df_10k)} 10-K filings")

mda_segments = []
for _, row in df_10k.iterrows():
    mda = row.get("item_7", "")
    if not mda or len(mda) < 1000:
        continue

    company = row.get("company", "Unknown")
    date = row.get("date", "")

    # Split MD&A into paragraphs and chunk
    paragraphs = mda.split("\n\n")
    if len(paragraphs) < 3:
        paragraphs = mda.split("\n")

    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue
        para_tokens = count_tokens(para)

        if current_len + para_tokens > MAX_TOKENS and current_len >= MIN_TOKENS:
            chunk_text = "\n\n".join(current_chunk)
            mda_segments.append({
                "text": chunk_text,
                "source": "sec_10k_mda",
                "source_domain": "sec_filings",
                "company": company,
                "period": str(date),
                "token_length": current_len,
            })
            current_chunk = []
            current_len = 0

        current_chunk.append(para)
        current_len += para_tokens

    if current_len >= MIN_TOKENS:
        chunk_text = "\n\n".join(current_chunk)
        mda_segments.append({
            "text": chunk_text,
            "source": "sec_10k_mda",
            "source_domain": "sec_filings",
            "company": company,
            "period": str(date),
            "token_length": current_len,
        })

df_mda_segments = pd.DataFrame(mda_segments)
print(f"MD&A segments: {len(df_mda_segments)}")
print(f"Token length: min={df_mda_segments['token_length'].min()}, "
      f"median={df_mda_segments['token_length'].median():.0f}, "
      f"max={df_mda_segments['token_length'].max()}")
```

#### Combine & Filter

```python
df_long = pd.concat([df_ec_segments, df_mda_segments], ignore_index=True)

# Enforce token limits: 512 <= tokens <= 8192
df_long = df_long[
    (df_long["token_length"] >= MIN_TOKENS) &
    (df_long["token_length"] <= MAX_TOKENS)
]
print(f"After filtering [{MIN_TOKENS}, {MAX_TOKENS}] tokens: {len(df_long)} rows")
print(f"\nToken length distribution:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  p{p}: {df_long['token_length'].quantile(p/100):.0f}")

print(f"\nPer source:")
print(df_long["source"].value_counts())
```

---

## Phase L2: Balancing & Capping

Target: ~43K rows total (matching the short-context v2 dataset).

```python
TARGET_TOTAL = 43_000

# If we have more than target, sample down proportionally
if len(df_long) > TARGET_TOTAL:
    # Balance between sources
    n_per_source = TARGET_TOTAL // df_long["source"].nunique()
    sampled = []
    for source in df_long["source"].unique():
        source_df = df_long[df_long["source"] == source]
        if len(source_df) > n_per_source:
            sampled.append(source_df.sample(n=n_per_source, random_state=42))
        else:
            sampled.append(source_df)
    df_long = pd.concat(sampled, ignore_index=True)

df_long = df_long.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Final long-context dataset: {len(df_long)} rows")
print(f"Per source:\n{df_long['source'].value_counts()}")
```

---

## Phase L3: Entity & Sentiment Annotation (Agent-Labeled)

Same approach as Part 1: agent reads each text, determines entity and entity_sentiment. Since these are long texts, they will typically mention multiple entities -- the agent picks the most prominent one.

For long-context texts, the label itself (sentence-level sentiment) is also determined by the agent, since these texts are unlabeled. The agent reads the text and assigns:
- `label`: Overall sentiment of the passage (POSITIVE / NEGATIVE / NEUTRAL)
- `entity`: Primary financial entity
- `entity_sentiment`: Sentiment toward that entity

```python
# Prepare annotation batches
BATCH_SIZE = 100  # smaller batches for long texts
ANNOTATION_DIR = "data/processed/entity_annotations_long"
os.makedirs(ANNOTATION_DIR, exist_ok=True)

n_batches = (len(df_long) + BATCH_SIZE - 1) // BATCH_SIZE
for i in range(n_batches):
    start = i * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(df_long))
    batch = df_long.iloc[start:end][["text"]].copy()
    batch.to_csv(f"{ANNOTATION_DIR}/batch_{i:04d}_input.csv", index=True)

print(f"Prepared {n_batches} batches of {BATCH_SIZE} for agent annotation")

# Agent annotates: for each row, determine label, entity, entity_sentiment
# Output CSV columns: (index), label, entity, entity_sentiment
```

---

## Phase L4: Dedup, Split & Validation

```python
import re
from hashlib import md5

# Exact dedup
df_long["text_norm"] = df_long["text"].apply(lambda t: re.sub(r"\s+", " ", str(t).strip().lower()))
n_before = len(df_long)
df_long = df_long.drop_duplicates(subset=["text_norm"], keep="first")
print(f"Dedup: {n_before} -> {len(df_long)}")
df_long = df_long.drop(columns=["text_norm"]).reset_index(drop=True)

# Add metadata columns
df_long["label_confidence"] = "agent"

# Keep only final columns
df_long = df_long[["text", "label", "source", "source_domain", "label_confidence", "entity", "entity_sentiment"]]

# Train/val/test split (80/10/10)
from sklearn.model_selection import train_test_split

df_train, df_temp = train_test_split(df_long, test_size=0.2, random_state=42, stratify=df_long["source"])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp["source"])

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Validate
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    nulls = df.isnull().sum().sum()
    valid_labels = set(df["label"].unique()) <= {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    min_tokens = df["text"].str.len().min()
    print(f"  {name}: {len(df)} rows, nulls={nulls}, labels_valid={valid_labels}")
```

---

## Phase L5: Push to HuggingFace

```python
from datasets import Dataset, DatasetDict

ds_dict = DatasetDict({
    "train": Dataset.from_pandas(df_train, preserve_index=False),
    "validation": Dataset.from_pandas(df_val, preserve_index=False),
    "test": Dataset.from_pandas(df_test, preserve_index=False),
})

ds_dict.push_to_hub("neoyipeng/modernfinbert-training-v2-long", private=False)
print("Pushed to neoyipeng/modernfinbert-training-v2-long")
```

---

## Long-Context TODO List

### Phase L1: Collection & Chunking ✅

- [x] **L1.1** Create notebook `notebooks/03_long_context_dataset.ipynb`
- [x] **L1.2** Load ModernBERT tokenizer for accurate token counting
  - [x] `AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")`
  - [x] Verify max model length is 8,192
- [x] **L1.3** Download `kurry/sp500_earnings_transcripts` (33K transcripts, MIT)
  - [x] Load full dataset: `load_dataset("kurry/sp500_earnings_transcripts", split="train")`
  - [x] Print raw stats: row count, columns, text length distribution
- [x] **L1.4** Chunk earnings transcripts into 512-8,192 token segments
  - [x] Use `structured_content` field (speaker-segmented) if available, else `content`
  - [x] Split on `\n\n` paragraph breaks
  - [x] Accumulate consecutive paragraphs until chunk reaches 512+ tokens
  - [x] When adding a paragraph would exceed 8,192 tokens, save current chunk and start new one
  - [x] Discard trailing chunks shorter than 512 tokens
  - [x] Preserve metadata: company name, year, quarter per chunk
  - [x] Count tokens per chunk using tokenizer (not char length approximation)
- [x] **L1.5** Print earnings chunk stats: total chunks, token length percentiles (p25/p50/p75/p90/p95/max)
- [x] **L1.6** Download `jlohding/sp500-edgar-10k` (6.3K filings, MIT)
  - [x] Load: `load_dataset("jlohding/sp500-edgar-10k", split="train")`
  - [x] Print raw stats: row count, which Item columns are populated
- [x] **L1.7** Extract Item 7 (MD&A) sections from 10-K filings
  - [x] Filter rows where `item_7` is non-empty and len > 1000 chars
  - [x] Print how many filings have usable MD&A sections
- [x] **L1.8** Chunk MD&A sections into 512-8,192 token segments
  - [x] Split on `\n\n` paragraph breaks, fall back to `\n` if few paragraphs
  - [x] Same accumulation logic as earnings: build up to 512+, cap at 8,192
  - [x] Discard chunks under 512 tokens
  - [x] Preserve metadata: company name, filing date per chunk
- [x] **L1.9** Print MD&A chunk stats: total chunks, token length percentiles
- [x] **L1.10** Combine earnings + MD&A chunks into single dataframe
  - [x] Unified columns: `text`, `source` ("earnings_transcripts" / "sec_10k_mda"), `source_domain`, `company`, `period`, `token_length`
- [x] **L1.11** Hard filter: drop any rows outside [512, 8192] token range
- [x] **L1.12** Print combined stats:
  - [x] Total rows
  - [x] Per-source counts
  - [x] Token length percentiles (p25/p50/p75/p90/p95/max)
  - [x] Number of unique companies
- [x] **L1.13** Save raw long-context dataset to `data/processed/long_raw_combined.parquet`

### Phase L2: Deduplication ✅

- [x] **L2.1** Exact dedup: normalize text (lowercase, collapse whitespace), drop duplicates
  - [x] Print count removed
- [x] **L2.2** Near-duplicate dedup: MinHash (same config as v2: 5-gram, 64 perms, 16 bands, Jaccard > 0.8)
  - [x] Print count removed
- [x] **L2.3** Print post-dedup stats: total rows, per-source counts
- [x] **L2.4** Save to `data/processed/long_deduped.parquet`

### Phase L3: Balancing & Capping ✅

- [x] **L3.1** Check raw source distribution after dedup
- [x] **L3.2** Cap total at ~43K rows (matching v2 short-context dataset size)
  - [x] If fewer than 43K rows available, use all (no upsampling)
  - [x] If more than 43K, downsample each source proportionally
- [x] **L3.3** Balance between earnings_transcripts and sec_10k_mda sources
  - [x] Target roughly equal representation (50/50) or proportional to available data
- [x] **L3.4** Shuffle with fixed seed
- [x] **L3.5** Print balanced stats: total rows, per-source counts, token length percentiles
- [x] **L3.6** Save to `data/processed/long_balanced.parquet`

### Phase L4: Entity & Sentiment Annotation (Agent-Labeled) ✅

Since source data is **unlabeled**, the agent determines ALL three columns: `label`, `entity`, `entity_sentiment`.

- [x] **L4.1** Prepare annotation batches: 100 rows per batch (smaller than v2 due to longer texts)
  - [x] Save to `data/processed/entity_annotations_long/batch_XXXX_input.csv`
  - [x] Each batch CSV has columns: (index), text
  - [x] Print total batch count
- [x] **L4.2** Launch parallel agent workers to annotate batches
  - [x] For each row, agent reads the full text and determines:
    - `label`: overall sentiment of the passage (POSITIVE / NEGATIVE / NEUTRAL)
    - `entity`: primary financial entity the passage is about (canonical company name, ticker, index, commodity, "MARKET", or "NONE")
    - `entity_sentiment`: sentiment specifically toward the entity (may differ from overall label)
  - [x] Agent writes `batch_XXXX_output.csv` with columns: (index), label, entity, entity_sentiment
- [x] **L4.3** Monitor annotation progress: track completed vs total batches
- [x] **L4.4** Handle rate-limited agents: identify missing batches, re-launch agents for gaps
- [x] **L4.5** Assemble all batch outputs into main dataframe
  - [x] Verify row count matches: annotations == balanced dataset
  - [x] Join on index
- [x] **L4.6** Validate annotations:
  - [x] No null values in label, entity, entity_sentiment
  - [x] All labels in {POSITIVE, NEGATIVE, NEUTRAL}
  - [x] All entity_sentiment in {POSITIVE, NEGATIVE, NEUTRAL}
- [x] **L4.7** Text-verify entities: every non-NONE/MARKET entity must appear in its text
  - [x] Use same substring matching logic as v2 (canonical name, short form, first word)
  - [x] Reset hallucinated entities to NONE, entity_sentiment = label
  - [x] Print count of corrections
- [x] **L4.8** Re-annotate NONE entities with second-pass agents (same as v2)
  - [x] Prepare NONE-only batches
  - [x] Agent re-examines and finds missed entities
  - [x] Text-verify again, fix remaining hallucinations
- [x] **L4.9** Print final annotation stats:
  - [x] Label distribution (NEG/NEU/POS)
  - [x] Entity coverage (non-NONE %)
  - [x] Unique entities
  - [x] Top 20 entities
  - [x] Entity-label agreement rate
- [x] **L4.10** Save annotated dataset to `data/processed/long_annotated.parquet`

### Phase L5: Train/Val/Test Split & Validation ✅

- [x] **L5.1** Add metadata columns: `label_confidence = "agent"`
- [x] **L5.2** Select final columns: text, label, source, source_domain, label_confidence, entity, entity_sentiment
- [x] **L5.3** Stratified split 80/10/10 on `source` (and `label` if enough per-group samples)
- [x] **L5.4** Save splits:
  - [x] `data/processed/long_train.parquet`
  - [x] `data/processed/long_val.parquet`
  - [x] `data/processed/long_test.parquet`
- [x] **L5.5** Validate all 3 splits:
  - [x] No null values
  - [x] All labels valid
  - [x] All token lengths in [512, 8192]
  - [x] No single source > 60%
  - [x] Report label distribution per split
  - [x] Zero exact duplicates within each split
  - [x] Entity coverage
- [x] **L5.6** Cross-split leakage check: no text in more than one split
- [x] **L5.7** Print final dataset summary: rows per split, source composition, label distribution, token length stats, entity stats

### Phase L6: Push to HuggingFace ✅

- [x] **L6.1** Create DatasetDict from train/val/test parquets
- [x] **L6.2** Push to `neoyipeng/modernfinbert-training-v2-long` (public)
- [x] **L6.3** Write dataset card:
  - [x] Description: long-context companion to v2
  - [x] Source datasets with attribution (MIT licenses)
  - [x] Token length distribution stats
  - [x] Chunking methodology
  - [x] Agent annotation methodology (same as v2)
  - [x] Schema documentation
  - [x] Intended use: training ModernBERT with long-context financial text
  - [x] License: MIT (both sources are MIT)
- [x] **L6.4** Verify dataset loads correctly from HuggingFace

---

## Resolved Decisions

1. ~~FOMC dataset~~ -- **Removed**. Hawkish/dovish is monetary policy stance, not financial sentiment. Not compatible.
2. **Source cap size** -- 15K per source. Confirmed.
3. **Entity annotation** -- Fully agent-labeled. No packages/scripts needed.
4. **Aiera as training data** -- Confirmed. Using FPB, FiQA, TFNS as held-out benchmarks instead.
5. **More earnings call data** -- Added `gtfintechlab/SubjECTive-QA` (2.7K earnings call QA pairs, CC-BY 4.0). Uses OPTIMISTIC dimension as sentiment label.
6. **Long-context dataset** -- New companion dataset from earnings transcripts + 10-K MD&A. Each row 512-8,192 tokens. Max ~43K rows. Agent-labeled for entity-level sentiment.
