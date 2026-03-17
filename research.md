# Research Report: Full Fine-Tuning + DataBoost for FPB Performance

## Executive Summary

**There is no notebook that combines full fine-tuning (all parameters unfrozen) with DataBoost augmentation.** These two techniques exist in separate, independent notebooks:

- **`02A_databoost_vs.ipynb`** — DataBoost augmentation, but uses **LoRA r=16** (not full fine-tuning)
- **`09e_full_finetune.ipynb`** (archived) — Full fine-tuning (all params), but uses the **base dataset only** (no DataBoost)

This is a notable gap. The DataBoost notebook (02A) is the best-performing setup in the paper for FPB test performance, but it operates with LoRA. The full fine-tuning notebook (09e) was designed to diagnose whether the BERT vs. ModernBERT gap was a LoRA artifact, not to maximize FPB performance. No experiment combines both.

---

## Deep Analysis of Each Notebook

### 1. DataBoost Notebook (`02A_databoost_vs.ipynb`) — The Best Setup for FPB

This is the most important notebook for FPB test performance. It implements a four-stage pipeline:

#### Stage 1: Data Preparation
- **Dataset**: `neoyipeng/financial_reasoning_aggregated` from HuggingFace
- **Filtering**: Sentiment task only, FPB (source 5) entirely excluded
- **Training set size**: ~8,643 samples after filtering
- **Label encoding**: One-hot vectors via `np.eye(3)`, mapping `NEGATIVE→0, NEUTRAL/MIXED→1, POSITIVE→2`
- **Key specificity**: Labels are stored as **soft one-hot vectors** (e.g., `[1,0,0]`), not integer class indices. This means the loss function computes against a distribution, not a hard target — but since these are one-hot, it's functionally equivalent to cross-entropy with hard labels.

#### Stage 2: Baseline Model Training
The `train_model()` function is reused for both baseline and boosted training. Its specifics:

- **Architecture**: `answerdotai/ModernBERT-base` (149M total params)
- **LoRA config**: r=16, alpha=32, dropout=0.05, bias="none"
  - **Target modules**: `["Wqkv", "out_proj", "Wi", "Wo"]`
  - These are ModernBERT-specific: `Wqkv` is the **fused QKV projection** (one matrix for query+key+value), `out_proj` is the attention output, `Wi`/`Wo` are the GeGLU FFN layers
  - **Critical detail**: Because ModernBERT fuses Q/K/V into a single `Wqkv` matrix, a LoRA r=16 adapter on it effectively gives ~r=5.3 per component (Q, K, V), while BERT's separate `query`/`key`/`value` each get full r=16. This is a known asymmetry acknowledged in the paper's Limitations section.
- **Optimizer**: AdamW with lr=2e-4, weight_decay=0.001, cosine schedule
- **Batch config**: batch_size=8, gradient_accumulation=4 (effective batch 32)
- **Warmup**: 10 steps (not ratio-based)
- **Epochs**: 10, with early stopping on validation loss (`load_best_model_at_end=True`)
- **Precision**: FP16 mixed precision
- **Attention**: SDPA (Scaled Dot-Product Attention, PyTorch native)
- **Gradient checkpointing**: Enabled (saves memory, trades compute)
- **Seed**: 3407

#### Stage 3: Error Mining
After baseline training, inference runs on the **validation set** (not test, not FPB):
- Collects all samples where `pred != true_label`
- **Result**: 82 misclassified samples out of ~480 validation samples (17.1% error rate)
- Error breakdown typically shows POSITIVE→NEUTRAL and NEUTRAL→POSITIVE as dominant confusion patterns
- Errors saved to `validation_errors.csv`

#### Stage 4: Verbalized Sampling Augmentation
This is where 02A differs from the original 02. Instead of calling the Claude API at runtime:

- **Pre-generated data**: The augmentation data is **embedded directly in the notebook** as a base64-encoded gzip blob (`VS_DATA_B64`). This makes the notebook self-contained and reproducible without API access.
- **Generation method**: VS-CoT (Verbalized Sampling with Chain-of-Thought), based on Zhang et al. 2025
- **How VS-CoT works**:
  1. For each misclassified sample, Claude analyzes *why* a classifier would confuse the predicted label with the true label
  2. Identifies distinguishing linguistic cues for the correct sentiment
  3. Plans coverage across financial sub-domains (earnings calls, analyst notes, SEC filings, social media, press releases, etc.)
  4. Generates **k=5 candidate texts with explicit probability estimates** — this is the key innovation over simple paraphrasing, which mode-collapses to near-identical outputs
  5. Each candidate spans a different financial register to maximize diversity
- **Volume**: 82 seed errors × ~3 paraphrases each = **246 augmented samples** (some seeds get more, some fewer)
- **Data structure**: Each augmented sample has: `text`, `label` (int), `confusion_type` (e.g., "POSITIVE→NEUTRAL")

#### Stage 5: Augmented Retraining
- Original training data (~8,643) + VS augmented data (246) = ~8,889 samples
- Augmentation ratio: ~2.8% of original training size
- **Same `train_model()` function** is called — identical hyperparameters, fresh model initialization
- **Key design choice**: The model is trained from scratch on augmented data, NOT fine-tuned further from the baseline checkpoint

#### Stage 6: Evaluation on Three Test Sets
1. **Validation set**: Measures whether augmentation fixed the specific errors it targeted
2. **FPB 50agree/allAgree**: The held-out benchmark (model never sees FPB during training)
3. **Aggregated test set**: In-distribution performance

#### Reported Results (from paper, Table 5 — Self-Training section gives the clearest numbers):
| Stage | FPB 50agree | FPB allAgree | Agg Test |
|-------|-------------|--------------|----------|
| Baseline | 80.91% | 93.24% | 79.79% |
| **DataBoosted** | **82.56%** | **95.14%** | **80.83%** |
| Delta | +1.65pp | +1.90pp | +1.04pp |

On the isolated DataBoost experiment (Table 3, aggregated test only):
- Accuracy: +2.92pp (77.29% → 80.21%)
- Macro F1: +7.84pp (66.60% → 74.43%)

**The F1 gain is disproportionately large**, meaning DataBoost primarily helps minority classes (NEGATIVE, POSITIVE).

---

### 2. Full Fine-Tuning Notebook (`09e_full_finetune.ipynb`) — Archived

This notebook was designed to answer one specific question: **Is the BERT > ModernBERT gap under LoRA caused by the fused QKV asymmetry, or by pre-training differences?**

#### Key Differences from the DataBoost Setup

| Aspect | 02A (DataBoost) | 09e (Full FT) |
|--------|----------------|---------------|
| Parameters trained | ~1.1M (LoRA) | ~149M (all) |
| Augmented data | Yes (246 VS samples) | No |
| Models compared | ModernBERT only | BERT-base vs ModernBERT |
| Batch config | bs=8, grad_accum=4 | bs=8, grad_accum=2 |
| Warmup | 10 steps | 0.1 ratio |
| Learning rate | 2e-4 | 2e-5 (10x lower) |
| Weight decay | 0.001 | 0.01 (10x higher) |
| Seed | 3407 | 42 |
| Optimizer | AdamW | AdamW |
| Scheduler | Cosine | Cosine |
| Epochs | 10 | 10 |

**Critical hyperparameter differences**: Full FT uses lr=2e-5 (standard for full fine-tuning of transformers) vs. 2e-4 for LoRA. Weight decay is 0.01 vs. 0.001. Effective batch size is 16 vs. 32. These are appropriate adjustments for full fine-tuning but mean the results are not directly comparable to 02A beyond architecture conclusions.

#### Full FT Results
The notebook compares against prior LoRA results:
```
Config                        Params    FPB 50agree  FPB allAgree
BERT + LoRA r=16 (NB05)      ~0.89M    95.19%       99.16%
BERT + Full FT (this exp)    ~110M     [computed]   [computed]
ModernBERT + LoRA r=16       ~1.1M     91.19%       99.03%
ModernBERT + Full FT         ~149M     [computed]   [computed]
```

The notebook's diagnostic logic:
- If gap < 1.5pp under full FT → LoRA asymmetry was the main cause
- If gap > 2pp under full FT → Pre-training alignment is the cause
- Otherwise → Both factors contribute

**Note**: This notebook is in `archive/not_in_paper/`, meaning its results were not included in the final paper. The paper acknowledges the LoRA asymmetry as a limitation but doesn't resolve it experimentally.

---

### 3. Original DataBoost Notebook (`02_databoost.ipynb`) — Archived

The predecessor to 02A. Key differences:
- Uses the **Anthropic API at runtime** to generate paraphrases (requires API key and credits)
- Has `PARAPHRASES_PER_SAMPLE = 3` as a configurable parameter
- Generates paraphrases on-the-fly rather than using pre-embedded data
- Also evaluates baseline on FPB *before* deleting the model (to measure delta on same test)
- Evaluates on aggregated test set as well

The prompt template (from CSV files in `data/raw/`):
```
HEADLINE: [original misclassified text]

TASK: Create another headline that is similar in theme, but different
in terms of entities mentioned and the same in sentiment ([LABEL]).
If the sentiment is ambiguous, make it more clear. Modify the headline
such that a financial analyst would 100% certainly classify the
headline sentiment as [LABEL].
```

This is **not** the VS-CoT approach — it's simple paraphrasing. The 02A notebook upgraded to VS-CoT for better diversity.

---

## The Training Data Pipeline (Shared Across Notebooks)

### Dataset: `neoyipeng/financial_reasoning_aggregated`

| Source | Domain | N_train | NEG% | NEU% | POS% | Median Words | Annotation |
|--------|--------|---------|------|------|------|-------------|------------|
| 3 | Earnings calls (narrative) | 462 | 11.3 | 53.4 | 35.3 | 32 | LLM |
| 4 | Press releases / news | 1,557 | 3.4 | 58.7 | 37.9 | 60 | LLM |
| 5 | **FPB (excluded)** | 4,361 | 12.5 | 59.4 | 28.1 | 21 | Human |
| 8 | Earnings calls (Q&A) | 2,440 | 7.9 | 57.7 | 34.4 | 161 | LLM |
| 9 | Financial tweets | 4,184 | 13.9 | 68.4 | 17.6 | 15 | LLM |

**Total training (excl FPB): 8,643 samples**

### Known Data Issues (from provenance audit)
1. **Source 4 is 68% Canadian mining**: TSX-V listings, drill results, assay values — extremely narrow sub-domain
2. **Source 8 truncation**: 68.1% of earnings call Q&A samples exceed 512 tokens and get truncated — model trains on incomplete text
3. **Class distribution mismatch**: Training has 10.2% NEGATIVE vs. FPB's 12.5%, creating a distribution shift
4. **All non-FPB labels are LLM-generated**: Sources 3, 4, 8, 9 all use LLM annotation, only FPB (excluded) has human labels

### Evaluation Protocol
- **FPB sentences_50agree**: 4,846 sentences, ≥50% annotator agreement. The harder benchmark.
- **FPB sentences_allAgree**: 2,264 sentences, unanimous agreement. Easier (less ambiguous).
- **Aggregated test set**: ~480 samples from the same sources as training (in-distribution).

---

## Augmentation Data Files

Multiple generations of augmentation data exist in `data/raw/`:

### Manual DataBoost (simple paraphrasing, various batch sizes)
- `Manual_DataBoost - 100.csv` through `Manual_DataBoost - 940.csv`
- Structure: `Original Text, Prompt, Synthetic, Label`
- Generated via Claude API with simple paraphrase prompts

### ModernFinBERT DataBoost v0 (iterated paraphrasing)
- `ModernFinBERT_DataBoost_v0 - 1.csv`, `- 3.csv`, `- 1033.csv`, `- v0-2.csv`
- Structure: `Original Text, Prompt, Synthetic, Label, Stage`
- Includes stage markers (`v0-1`, `v0-2`) showing iterative augmentation rounds

### VS-CoT Augmentation (used in 02A)
- Embedded as `VS_DATA_B64` blob in the notebook itself
- 246 samples generated using Verbalized Sampling with Chain-of-Thought
- Targets 82 misclassified validation samples across multiple confusion patterns
- Spans 6+ financial sub-domains per sample

---

## What Would a "Full Fine-Tuning + DataBoost" Notebook Need?

To create the missing experiment, you would need to combine:

### From 09e (Full FT):
- All parameters unfrozen (no LoRA)
- Gradient checkpointing for memory
- Lower learning rate (2e-5 vs 2e-4)
- Higher weight decay (0.01 vs 0.001)
- Possibly larger effective batch size for stability

### From 02A (DataBoost):
- Error mining on validation set after baseline training
- VS-CoT augmentation data (either pre-generated or API-based)
- Concatenation of original + augmented data
- Fresh model retraining on augmented set
- Evaluation on all three test sets (val, FPB 50agree/allAgree, aggregated test)

### Key Considerations:
1. **Error mining must be redone** — the misclassified samples from full FT baseline will differ from those of the LoRA baseline, because the models have different capacity
2. **Augmentation data should be regenerated** — the confusion patterns may be different under full FT (e.g., different boundary errors)
3. **Memory requirements** — full FT of ModernBERT-base (149M params) + gradient checkpointing needs substantial GPU RAM (~12-16GB in FP16)
4. **Risk of overfitting** — full FT with only ~8,643 training samples (+ ~246 augmented) may overfit more aggressively than LoRA; the lower LR and higher weight decay in 09e are designed to mitigate this
5. **The LoRA asymmetry question** — if full FT closes the BERT-ModernBERT gap (as 09e was designed to test), then DataBoost on full FT ModernBERT might show different improvement dynamics

---

## Performance Landscape (All Configurations)

| Config | Protocol | FPB 50agree | FPB allAgree |
|--------|----------|-------------|--------------|
| ProsusAI/finbert | In-domain | 88.96% | 97.17% |
| FinBERT-IJCAI | In-domain | 94% | — |
| finbert-lc | In-domain | 89% | 97% |
| ModernBERT + LoRA (CV) | 10-fold CV | 86.88% | — |
| BERT + LoRA (CV) | 10-fold CV | 85.27% | — |
| ModernBERT + LoRA (held-out) | Held-out | 80.44-80.93% | 92.98-93.29% |
| **ModernBERT + LoRA + DataBoost** | **Held-out** | **82.56%** | **95.14%** |
| BERT + LoRA (held-out) | Held-out | 73.09% | 83.66% |
| ModernBERT + LoRA + DataBoost + Self-Train | Held-out | 80.54% | 94.08% |
| finbert-tone | Zero-shot | 79.14% | 91.52% |
| Claude Opus 4.6 + skill | Zero-shot | 72.75% (723 samples) | — |

**DataBoosted ModernBERT + LoRA is the best held-out configuration.** Self-training degrades performance. Full fine-tuning + DataBoost has never been tested.

---

## Key Insights

1. **DataBoost is the single most impactful technique** for improving FPB held-out performance (+1.65pp accuracy, +7.84pp F1 on aggregated test)
2. **Only 246 samples** (2.8% of training data) drive the improvement — extremely data-efficient
3. **VS-CoT is critical** — standard paraphrasing mode-collapses; VS-CoT forces distributional diversity across financial registers
4. **The model trains from scratch** each time — no incremental fine-tuning from baseline checkpoint
5. **Error mining targets validation, not test** — this prevents data leakage but means the augmented data may not perfectly target FPB-specific confusion patterns
6. **The protocol gap (6.4pp)** between in-domain CV and held-out evaluation is the central finding of the paper — DataBoost partially closes it
7. **Full FT + DataBoost is untested** — this is the natural next experiment and could potentially yield the best held-out performance, but needs careful hyperparameter tuning to avoid overfitting on the small dataset
