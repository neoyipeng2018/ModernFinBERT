# ModernFinBERT — Detailed Research Report

This report documents the full scope of the ModernFinBERT project: every notebook, the published paper, supporting infrastructure, experimental results, and the narrative arc connecting them.

---

## 1. Project Overview

ModernFinBERT is a systematic empirical study applying ModernBERT (Warner et al., 2024) to three-class financial sentiment analysis (POSITIVE / NEGATIVE / NEUTRAL) on the FinancialPhraseBank (FPB) benchmark. The project uses LoRA-based parameter-efficient fine-tuning and runs on Kaggle GPUs (T4/P100).

**Core question**: Does ModernBERT's modernized architecture (RoPE, Flash Attention, GeGLU activations, 149M params) outperform legacy BERT-base (110M params) on domain-specific financial text?

**Published model**: `neoyipeng/ModernFinBERT-base` on HuggingFace.

**Paper**: "ModernFinBERT: A Systematic Study of ModernBERT for Financial Sentiment Analysis" by Yipeng Neo, reporting eight controlled experiments across 13 pages.

---

## 2. Datasets

### 2.1 Training Data

- **Source**: `neoyipeng/financial_reasoning_aggregated` from HuggingFace
- **Filtering**: sentiment task only, FPB (source=5) excluded from all splits
- **Size after filtering**: 8,643 train / 480 val / 480 test
- **Labels**: 3-class one-hot encoded — NEGATIVE (0), NEUTRAL/MIXED (1), POSITIVE (2)

### 2.2 Evaluation Data — FinancialPhraseBank (FPB)

- **sentences_50agree** (≥50% annotator agreement): 4,846 sentences. Distribution: NEGATIVE 604 (12.5%), NEUTRAL 2,879 (59.4%), POSITIVE 1,363 (28.1%).
- **sentences_allAgree** (unanimous): 2,264 sentences. Distribution: NEGATIVE 303 (13.4%), NEUTRAL 1,391 (61.4%), POSITIVE 570 (25.2%).

### 2.3 Augmentation Data

- **VS-CoT augmentation** (`data/vs_augmented_errors.csv`): 410 samples generated via Verbalized Sampling, targeting 82 misclassified validation samples (5 paraphrases each). Spans 6 financial sub-domains: earnings calls, analyst notes, news headlines, press releases, social media, SEC filings.
- **Raw DataBoost files** (`data/raw/`): ~4,566 rows across Manual_DataBoost and ModernFinBERT_DataBoost_v0 CSV files (earlier augmentation attempts).

### 2.4 Unlabeled Data (for self-training)

- Twitter Financial News datasets from HuggingFace (`zeroshot/twitter-financial-news-sentiment` and `zeroshot/twitter-financial-news-topic`)
- Filtered to 5-50 word sentences, deduplicated, capped at 50,000
- ~30,000 sentences after cleaning

---

## 3. Shared Experimental Configuration

All notebooks share a consistent infrastructure unless otherwise noted:

| Parameter | Value |
|---|---|
| Base model | `answerdotai/ModernBERT-base` (149M params) |
| LoRA rank | r=16, alpha=32 |
| LoRA targets (ModernBERT) | `Wqkv`, `out_proj`, `Wi`, `Wo` |
| LoRA targets (BERT-base) | `query`, `key`, `value`, `dense` |
| LoRA dropout | 0.05, bias="none" |
| Trainable params (ModernBERT) | 3,381,507 (2.21% of 153M) |
| Trainable params (BERT) | 2,681,091 (2.39% of 112M) |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.001 |
| Scheduler | Cosine with 10 warmup steps |
| Batch size | 8, gradient_accumulation=4 (effective 32) |
| Precision | FP16, gradient checkpointing |
| Epochs | 10, early stopping on eval_loss |
| Seed | 3407 |
| Max sequence length | 512 |
| Attention | SDPA |

---

## 4. Notebook-by-Notebook Analysis

### NB01: Architecture Comparison (`01_architecture_comparison.ipynb`)

**Purpose**: Fine-tune ModernBERT-base on the aggregated dataset (FPB excluded) and evaluate on FPB as a completely held-out benchmark. Compare against published FinBERT baselines.

**Published baselines (hardcoded from literature)**:
- FPB 50agree: LSTM+ELMo 75%, ULMFit 83%, ProsusAI/finbert 86%, FinBERT-FinVocab 87.2%
- FPB allAgree: LSTM+ELMo 84%, ULMFit 93%, ProsusAI/finbert 97%

**Key methodological note**: ModernFinBERT never sees FPB during training. All cited baselines used in-domain FPB data, making this a stricter evaluation.

**Results** (paper, Table 1): 80.44% on 50agree, 92.98% on allAgree — trained on the clean 8,643-sample dataset.

**Visualizations**: Horizontal bar chart comparing accuracy; confusion matrices for both FPB variants.

---

### NB02: DataBoost (`02_databoost.ipynb`)

**Purpose**: Implement the DataBoost pipeline — train baseline, mine validation errors, paraphrase with LLM, retrain.

**Status**: Incomplete. Only baseline training and error mining are implemented. The LLM paraphrasing step is absent (installs `anthropic` but never uses it). NB02A completes this work.

**What is implemented**:
1. `train_model()` function encapsulating full training pipeline
2. `run_inference()` helper for batched inference (batch_size=32, max_length=512)
3. Error mining on validation set
4. Saves errors to `validation_errors.csv`
5. Baseline evaluation on FPB 50agree, allAgree, and aggregated test

**Error mining result**: 82 of 480 validation samples misclassified (17.1% error rate).

---

### NB02A: DataBoost with Verbalized Sampling (`02A_databoost_vs.ipynb`)

**Purpose**: Complete DataBoost implementation using pre-generated VS-CoT augmentation data.

**Key innovation**: Instead of calling an LLM at runtime, the notebook contains a gzip-compressed, base64-encoded CSV blob (`VS_DATA_B64`) with 410 pre-generated augmented texts. These target specific confusion patterns (POSITIVE→NEUTRAL, NEUTRAL→POSITIVE, NEGATIVE→NEUTRAL).

**Experimental flow**:
1. Train baseline model (identical hyperparams to NB01)
2. Evaluate baseline on 4 evaluation sets
3. Decode embedded VS augmentation data (410 samples)
4. Create augmented training set: original + VS augmentations (shuffled, seed=42)
5. Retrain fresh model on augmented data
6. Evaluate "DataBoosted" model on all 4 sets
7. Produce comparison table with published baselines

**Results** (paper, Table 2 — aggregated test set):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 0.7729 | 0.6660 |
| DataBoosted | 0.8021 | 0.7443 |
| **Delta** | **+0.029** | **+0.078** |

The disproportionate F1 gain (+7.8pp vs +2.9pp accuracy) confirms DataBoost primarily helps minority classes (NEGATIVE, POSITIVE).

---

### NB03: Claude Comparison (`03_claude_comparison.ipynb`)

**Purpose**: Compare fine-tuned ModernFinBERT (149M params) against Claude Opus 4.5/4.6 on FPB using zero-shot classification, with cost analysis.

**Claude setup**:
- Model: `claude-opus-4-20250514` (later described as Opus 4.6 in paper)
- System prompt forces JSON output `{"sentiment": "POSITIVE"|"NEGATIVE"|"NEUTRAL"}`
- Rate limiting: 1-second pause every 20 samples
- Fallback: substring matching if JSON parse fails; defaults to NEUTRAL on exception
- Used a purpose-built "financial sentiment engine" skill with investor-lens definitions, text-length-dependent decision framework (direct classification for short text, avoiding CoT per Vamvourellis & Mehta 2025)
- **Blind evaluation**: four independent Claude agents each classify a chunk

**Results** (paper, Table 3):

| Model | Accuracy | Macro F1 | Cost/sample | Latency/sample |
|---|---|---|---|---|
| ModernFinBERT (149M) | 0.8044 | 0.7705 | ~$0.00001 | ~2ms |
| Claude Opus 4.6 + skill | 0.7280 | 0.7030 | ~$0.008 | ~1-2s |

Fine-tuned model is **800x cheaper** and **500-1000x faster** while being 7.6pp more accurate.

**Error analysis**: POSITIVE↔NEUTRAL boundary is the dominant confusion source (131 of 197 errors). NEGATIVE recall strongest at 0.819.

---

### NB03A: Test Evaluation (`03A_test_evaluation.ipynb`)

**Purpose**: Load best saved checkpoints from NB01 and NB02, evaluate all three models on the aggregated test set in a single consistent run.

**Models evaluated**:
1. NB01 ModernFinBERT
2. NB02 Baseline (same setup, separate run)
3. NB02 DataBoosted (augmented retraining)

**Requires**: Pre-saved Kaggle dataset at `/kaggle/input/modernfinbert-best-checkpoints/` with LoRA adapter files.

**Results** (from kaggle_output_03A):
- NB01: 77.71% / 0.6958 F1
- NB02 Baseline: 77.29% / 0.666 F1
- NB02 DataBoosted: **80.21% / 0.7443 F1**

DataBoost provides clear improvement on the held-out test set.

---

### NB04: 10-Fold Cross-Validation (`04_kfold_cv.ipynb`)

**Purpose**: Properly replicate the FinBERT paper's evaluation protocol using stratified 10-fold CV on FPB sentences_50agree. In-domain evaluation where the model trains and tests on FPB itself.

**Fixes over earlier k-fold attempt**:
- Model + LoRA re-initialized from pretrained weights each fold (previous version accumulated weights)
- `StratifiedKFold` instead of random shuffle
- Per-fold output directories
- Sanity check: epoch-1 training loss consistency across folds

**Configuration**: 5 epochs per fold (not 10), seed=42 for splits.

**Results** (paper, Table 4):

| Fold | Accuracy | Macro F1 |
|---|---|---|
| 1 | 0.8639 | 0.8438 |
| 2 | 0.8866 | 0.8765 |
| 3 | 0.8619 | 0.8529 |
| 4 | 0.8619 | 0.8494 |
| 5 | 0.8825 | 0.8664 |
| 6 | 0.8701 | 0.8718 |
| 7 | 0.8554 | 0.8323 |
| 8 | 0.8719 | 0.8441 |
| 9 | 0.8678 | 0.8469 |
| 10 | 0.8657 | 0.8562 |
| **Mean ± Std** | **0.8688 ± 0.0096** | **0.8540 ± 0.0139** |

Per-class F1: NEUTRAL 0.8984 ± 0.0082 (strongest), NEGATIVE 0.8556 ± 0.0355, POSITIVE 0.8081 ± 0.0217 (weakest, due to class imbalance).

**Comparable to published FinBERT results** (86-87% range).

---

### NB05: Architecture and Baseline Comparison

**Note**: The original NB05 has been removed. The pre-trained baseline comparison (ProsusAI/finbert, finbert-tone) and the BERT-base controlled comparison are now handled by NB09B and NB09C. See those sections below.

---

### NB06: Multi-Seed Robustness (`06_multi_seed.ipynb`)

**Purpose**: Run the NB01 protocol with 5 different seeds to produce confidence intervals.

**Seeds**: [3407, 42, 123, 456, 789]

**Results** (paper, Table 6):

| Evaluation Set | Mean Accuracy | Mean Macro F1 |
|---|---|---|
| FPB 50agree | 0.8044 ± 0.0089 | 0.7705 ± 0.0098 |
| FPB allAgree | 0.9298 ± 0.0025 | 0.9148 ± 0.0033 |
| Aggregated test | 0.7771 ± 0.0053 | 0.6958 ± 0.0062 |

Standard deviations below 1% across all metrics — confirms stable, reproducible results.

---

### NB07: Self-Training (`07_self_training.ipynb`)

**Purpose**: Iterative self-training using unlabeled financial tweets on top of the DataBoosted model.

**Pipeline**:
1. Train baseline on aggregated data
2. Train DataBoosted teacher (from NB02A)
3. Source ~30K unlabeled financial sentences from Twitter
4. Per-class top-k pseudo-label selection with increasing confidence thresholds: [15%, 25%, 40%] across rounds
5. Fresh student each round (no weight inheritance — anti-confirmation-bias measure)
6. Early stopping if validation accuracy does not improve

**Results** (paper, Table 7):

| Stage | FPB 50agree Acc | FPB 50agree F1 | FPB allAgree Acc |
|---|---|---|---|
| Baseline | 0.8091 | 0.7810 | 0.9324 |
| DataBoosted | **0.8256** | **0.8052** | **0.9514** |
| SelfTrain R1 | 0.8054 | 0.7700 | 0.9408 |

**Negative result**: Self-training degraded performance. Validation accuracy dropped from 83.54% to 82.92% in Round 1, triggering early stopping.

**Root causes**:
1. **Domain mismatch**: Twitter financial text is informal/abbreviated; FPB is formal press-release language
2. **Overconfident teacher**: Mean and min confidence both near 1.0, so pseudo-labels carried no meaningful uncertainty signal, amplifying the teacher's existing biases

---

---

### NB09A: Deduplication Audit (`09a_dedup_audit.ipynb`)

**Purpose**: Verify zero data leakage between aggregated training data and FPB before the controlled experiments (NB09B-E).

**Three checks**:
1. **Exact match**: Normalized (lowercased, stripped) text comparison
2. **Fuzzy match** (>90%): `difflib.SequenceMatcher.ratio()` with length filter
3. **Semantic near-duplicates** (cosine >0.95): `all-MiniLM-L6-v2` embeddings, chunked computation

**Dataset sizes**: 9,123 training samples vs. 4,846 FPB samples.

**Result**: **ALL CLEAN** — 0 exact, 0 fuzzy, 0 semantic matches. No contamination.

---

### NB09B: FPB-Only 10-Fold Cross-Validation (`09b_fpb_crossval.ipynb`)

**Purpose**: Head-to-head comparison on identical data. Trains all models on the same FPB folds to eliminate training data composition as a confound.

**Three configurations**:
1. `bert-base-uncased` + LoRA r=16
2. `ModernBERT-base` + LoRA r=16
3. `ModernBERT-base` + LoRA r=48

**Configuration differences from other notebooks**: batch_size=16, max_length=128, warmup_ratio=0.05 (instead of fixed warmup steps), integer labels (not one-hot).

**Statistical analysis**: Paired t-tests (`scipy.stats.ttest_rel`) for all three pairwise comparisons.

**Results** (paper, Table 8 — BERT vs ModernBERT r=16):

| Fold | BERT-base | ModernBERT | Delta |
|---|---|---|---|
| 0 | 0.8557 | 0.8763 | +0.0206 |
| 1 | 0.8227 | 0.8474 | +0.0247 |
| 2 | 0.8536 | 0.8639 | +0.0103 |
| 3 | 0.8619 | 0.8330 | -0.0289 |
| 4 | 0.8701 | 0.8680 | -0.0021 |
| 5 | 0.8515 | 0.8619 | +0.0103 |
| 6 | 0.8678 | 0.8636 | -0.0041 |
| 7 | 0.8657 | 0.8843 | +0.0186 |
| 8 | 0.8574 | 0.8884 | +0.0310 |
| 9 | 0.8202 | 0.8492 | +0.0289 |
| **Mean ± Std** | **0.8527 ± 0.0175** | **0.8636 ± 0.0171** | **+0.0109** |

ModernBERT wins 7 of 10 folds. Paired t-test: t=1.88, **p=0.093** — marginally significant. No evidence BERT outperforms ModernBERT.

**Note**: ModernBERT r=48 only completed 2 of 10 folds (likely hit Kaggle time limit).

---

### NB09C: Clean Held-Out Evaluation (`09c_clean_holdout.ipynb`)

**Purpose**: The definitive architecture comparison. Re-runs the held-out protocol (aggregated data minus FPB → evaluate on FPB) with verified-clean data for both BERT and ModernBERT.

**Results** (paper, Table 9):

| Model | FPB 50agree Acc | FPB 50agree F1 | FPB allAgree Acc |
|---|---|---|---|
| ModernBERT + LoRA r=16 | **80.93%** | **77.93%** | **93.29%** |
| BERT-base + LoRA r=16 | 73.09% | 60.52% | 83.66% |
| **Delta (MB - BERT)** | **+7.84pp** | **+17.41pp** | **+9.63pp** |

ModernBERT is decisively better than BERT-base on clean data across all metrics.

---

### NB09D: Sample Efficiency Curves (`09d_sample_efficiency.ipynb`)

**Purpose**: Map how BERT and ModernBERT scale with training data size. Determines whether the gap is data-dependent or fundamental.

**Design**: 2 models × 6 sample sizes (500, 1K, 2K, 4K, 8K, 13K) × 3 seeds = **36 training runs**.

**Interpretation framework**:
- Final gap < 50% of initial → "ModernBERT needs more data"
- Final gap > 80% of initial → "Pre-training alignment is the cause"

**Status**: Not yet executed (no kaggle_output_09d). Listed as future work in the paper's Limitations section.

---

### NB09E: Full Fine-Tuning (`09e_full_finetune.ipynb`)

**Purpose**: The definitive LoRA ablation. Eliminates LoRA entirely by training ALL parameters unfrozen. If ModernBERT catches up under full fine-tuning, the gap was a LoRA artifact. If BERT still wins, the gap is from pre-training.

**Key hyperparameter differences from LoRA experiments**:
- Learning rate: 2e-5 (10x lower — standard for full fine-tuning)
- Weight decay: 0.01 (10x higher)
- Warmup ratio: 0.1
- Gradient accumulation: 2 (effective batch 16)

**Interpretation thresholds**:
- Gap < 1.5pp → "LoRA asymmetry was the main cause"
- Gap > 2pp → "Pre-training alignment is the cause"

**Status**: Not yet executed. Listed as future work in the paper's Limitations section.

---

## 6. The Protocol Gap

A key analytical finding is the 6.4pp gap between in-domain and held-out evaluation of the same model:

| Protocol | FPB 50agree Accuracy |
|---|---|
| 10-fold CV on FPB (NB04) | 86.88% |
| Held-out evaluation (NB01) | 80.44% |
| **Gap** | **6.44pp** |

This reflects the difference between testing on FPB-distributed data (CV) vs. true generalization from non-FPB financial text. The paper argues both numbers should always be reported, and the held-out number better reflects real-world deployment performance.

---

## 7. Skills (Claude Code Prompts)

### 7.1 Financial Sentiment Engine (`skills/financial-sentiment-engine/`)

A structured prompt for Claude to classify financial sentiment with:
- **Investor-lens definitions**: POSITIVE = value creation/alpha signals, NEGATIVE = risk/value destruction, NEUTRAL = no clear investment implication
- **Text-length-dependent strategy**: direct classification for short text (based on Vamvourellis & Mehta 2025 finding that CoT hurts accuracy on short financial text), label-first reasoning for medium text, analogical reasoning for long text
- **Calibrated probability guidelines** and batch mode
- **Edge case handling**: sarcasm, relative statements, forward-looking language

**Blind benchmark**: 72.8% accuracy / 0.703 macro F1 on 723-sample diverse test set. POSITIVE↔NEUTRAL confusion dominates (131/197 errors).

### 7.2 Verbalized Sampling Augment (`skills/verbalized-sampling-augment/`)

A prompt implementing the VS-CoT approach (Zhang et al., 2025) for generating diverse synthetic training data:
- Instead of single paraphrases (which mode-collapse), generates k candidates with explicit probability distributions
- Each candidate grounded in a different financial sub-domain
- Three variants: VS-CoT (chain-of-thought), VS-Standard, VS-Multi
- Includes confusion-type analysis and tail sampling for diversity

---

## 8. Paper Structure and Claims

The paper (`paper/main.tex`) is organized into 7 sections:

1. **Introduction**: Motivates the study, lists 7 experiments, introduces the protocol gap
2. **Related Work**: FPB history, ModernBERT architecture, LoRA, data augmentation (VS), self-training
3. **Experimental Setup**: Datasets (3.1), model config (3.2), evaluation metrics (3.3)
4. **Experiments and Results**: 7 subsections (NB01-NB07/NB09), each with protocol, tables, and analysis
5. **Analysis and Discussion**: Protocol gap (5.1), architecture comparison (5.2), fine-tuning vs LLMs (5.3), DataBoost (5.4), self-training failure (5.5)
6. **Limitations**: Single benchmark, base models only, English only, LoRA only, single training size, tweet-only self-training data
7. **Conclusion**: Five key findings

### Five Key Claims

1. **Evaluation protocol matters**: 6.4pp gap between in-domain CV and held-out eval
2. **ModernBERT outperforms BERT**: +7.84pp held-out, +1.09pp cross-validation on verified-clean data
3. **DataBoost works**: +2.9pp accuracy, +7.8pp F1 from only 246 generated samples
4. **Fine-tuning beats prompted LLMs**: 80.4% vs 72.8%, 800x cheaper, 500-1000x faster
5. **Self-training requires domain match**: Twitter pseudo-labels degraded performance

---

## 9. Key Results Summary Table

| Experiment | FPB 50agree Acc | FPB allAgree Acc | Notes |
|---|---|---|---|
| ModernBERT held-out (NB01) | 80.44% | 92.98% | Paper's primary result |
| ModernBERT 10-fold CV (NB04) | 86.88% ± 0.96% | — | In-domain, comparable to FinBERT literature |
| ModernBERT DataBoosted (NB07) | 82.56% | 95.14% | Best overall result |
| ProsusAI/finbert | 88.96% | 97.17% | In-domain (trained on FPB) |
| finbert-tone | 79.14% | 91.52% | Zero-shot |
| BERT-base held-out (NB09C) | 73.09% | 83.66% | Controlled comparison |
| Claude Opus 4.6 + skill (NB03) | 72.80% | — | 800x more expensive |
| ModernBERT multi-seed (NB06) | 80.44% ± 0.89% | 92.98% ± 0.25% | 5 seeds, stable |

---

## 10. Infrastructure Notes

- **Compute**: All experiments run on Kaggle with T4/P100 GPUs
- **Kaggle push directories**: Each notebook has a corresponding `kaggle_push_XX/` with the notebook and `kernel-metadata.json` for submission
- **Output directories**: `kaggle_output_XX/` contain training logs, checkpoints, and result files
- **No scripts**: `scripts/evaluation/`, `scripts/preprocessing/`, `scripts/training/` are all empty — all code lives in notebooks
- **Dependencies**: PyTorch, Transformers, PEFT, TRL, Unsloth, WandB, accelerate, bitsandbytes, xformers

---

## 11. What Remains Unfinished

| Item | Status | Purpose |
|---|---|---|
| NB09D (sample efficiency) | Not yet run | Scaling curves to determine if gap is data-dependent |
| NB09E (full fine-tuning) | Not yet run | Definitive LoRA ablation |
| NB09B r=48 | Incomplete (2/10 folds) | Hit Kaggle time limit |
| `data/cleaned/`, `data/processed/` | Empty | Never populated |
| `scripts/` subdirectories | Empty | All code in notebooks |

Both 09D and 09E are acknowledged in the paper's Limitations section as "future work."

---

---

## 13. Verbalized Sampling — The Augmentation Methodology

The DataBoost augmentation uses Verbalized Sampling (Zhang et al., 2025), specifically the VS-CoT variant:

1. **Error analysis**: Identify confusion patterns (e.g., "POSITIVE misclassified as NEUTRAL")
2. **Sub-domain planning**: For each error, plan paraphrases across 6 financial registers (earnings calls, analyst notes, news, press releases, social media, SEC filings)
3. **Candidate generation**: Generate k candidates with explicit probability estimates
4. **Tail sampling**: Select diverse candidates, not just the most probable

This approach yields 1.6-2.1x more diversity than standard paraphrasing and avoids mode collapse. The 410 generated samples (targeting 82 errors × 5 paraphrases) improved macro F1 by 7.8pp — a remarkably efficient use of synthetic data.

---

## 14. Critical Assessment — Blindspots, Flaws & Weaknesses

*Added 2026-03-14 after deep paper review.*

### 14.1 The Uploaded Model vs. Experimental Models Are Different

The uploaded HuggingFace model (`neoyipeng/ModernFinBERT-base`) achieves **97.13%** on FPB 50agree (per `results/nb10_parts_ab.json`), while the held-out experimental model achieves **80.44%** (paper Table 1). A 17pp gap on the same benchmark is impossible from the same model. The uploaded model was almost certainly trained on FPB data. The Claude comparison (Table 3) used the uploaded model, making the "held-out FPB" framing misleading for that experiment.

### 14.2 LoRA Rank Asymmetry Confound

ModernBERT's fused `Wqkv` receives effectively ~r=5.3 per Q/K/V component (16/3) vs BERT's r=16 per separate projection. ModernBERT wins *despite* this disadvantage, but the magnitude of its advantage is unknowable. NB09e (full fine-tuning) was designed to eliminate this confound but those results aren't in the paper.

### 14.3 Statistical Significance

The head-to-head CV comparison reports p=0.093, which does not meet p<0.05. The paper nevertheless claims ModernBERT "consistently outperforms BERT."

### 14.4 Multi-Seed Results Are Suspiciously Identical to Single-Seed

Table 4 5-seed means match Experiment 1 single-seed results to 4 decimal places across all three metrics (0.8044, 0.9298, 0.7771). Statistically improbable.

### 14.5 DataBoost Evaluated on In-Distribution Data

Table 2 reports DataBoost results on the aggregated test set (same distribution as training), not on FPB. The FPB gain (from Table 7) is +1.65pp, not the +2.9pp reported.

### 14.6 Claude Comparison Issues

- Agents received truncated skill instructions (not full SKILL.md with examples)
- Single stochastic run with no confidence intervals
- 66% of test set is in-distribution for the fine-tuned model, not for Claude
- Cost analysis ignores training/maintenance costs

### 14.7 Training Data Provenance ~~(Was a Black Box)~~

**Status: ADDRESSED.** NB11 (`notebooks/11_data_provenance_audit.ipynb`) completed a full provenance audit. Paper Section 3.1 now includes Table 1 (data provenance) documenting all 4 training sources, their LLM-generated annotation method, class distributions, and text lengths. Three key findings are integrated into the paper: (1) all non-FPB labels are LLM-generated, (2) Source 4 is 68% Canadian mining press releases, (3) Source 8 samples are truncated at 512 tokens. New limitation items added to Section 6. The protocol gap discussion (Section 5.1) now connects the 10.2% vs 12.5% NEGATIVE class mismatch to the held-out performance drop.

### 14.8 Missing Experiments from Paper

NB09d (sample efficiency curves), NB09e (full fine-tuning), and NB10 (gap-widening techniques) were conducted but excluded from the paper despite directly addressing its biggest limitations.

### 14.9 Other Issues

- Single benchmark (FPB only, from 2014)
- No domain-specific pre-training (unlike original FinBERT)
- Label ambiguity treated as model failure (50agree threshold means up to 50% disagreed)
- No error analysis or confusion matrices in the paper (though NB10 has them)
- Inconsistent numbers across experiments for the same protocol (~0.5pp variation)
