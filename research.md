# ModernFinBERT: Deep Research Analysis

## Paper Overview

**Title**: ModernFinBERT: A Systematic Study of ModernBERT for Financial Sentiment Analysis
**Author**: Yipeng Neo
**Model**: `neoyipeng/ModernFinBERT-base` (HuggingFace)
**Dataset**: `neoyipeng/financial_reasoning_aggregated` (HuggingFace)
**Paper**: `paper/main.pdf`

This paper presents nine controlled experiments evaluating ModernBERT-base (149M parameters) for three-class financial sentiment analysis (POSITIVE / NEUTRAL / NEGATIVE) on the FinancialPhraseBank (FPB) benchmark. The central claim is methodological: **evaluation protocol choice inflates reported accuracy by 6.4 percentage points**, and the paper proposes a held-out protocol that gives a more honest picture of model generalization.

The production model is fine-tuned from vanilla `answerdotai/ModernBERT-base` using LoRA -- no domain-specific pre-training -- isolating the contribution of architecture alone.

---

## 1. The Protocol Gap (Core Contribution)

The paper's most important finding is the **protocol gap**: the same model achieves 86.88% accuracy under 10-fold cross-validation on FPB but only 80.44% when FPB is excluded from training entirely. That 6.4pp difference is the gap between "how well the model learns FPB's patterns" and "how well the model generalizes to FPB from other financial text."

Against published in-domain SOTA (89-94% accuracy), the gap widens to 9-14pp. This means most published FPB results are not comparable because they train on FPB data and evaluate on FPB data, conflating memorization with generalization.

**Why this matters**: The held-out protocol (train on aggregated financial corpus excluding FPB, evaluate on full FPB) creates a genuine transfer learning test. The model must learn "financial sentiment" from earnings calls, press releases, tweets, and other sources, then transfer that understanding to FPB's press-release style sentences. This is closer to real-world deployment where the model encounters unseen text.

**Key numbers**:
| Protocol | FPB 50agree Acc | FPB 50agree F1 |
|---|---|---|
| 10-fold CV (in-domain) | 86.88% +/- 0.96% | 85.40% +/- 1.39% |
| Held-out (FPB excluded from training) | 80.44% | 77.05% |
| Held-out + DataBoost | 82.56% | 80.52% |
| FPB allAgree (held-out + DataBoost) | 95.14% | 94.17% |

The allAgree subset (unanimous annotator agreement) reaches 95.14% -- showing that the gap between 80% and 93% on FPB largely reflects human annotator disagreement, not model failure.

---

## 2. Architecture: ModernBERT vs BERT

ModernBERT-base (149M params) incorporates three architectural modernizations over BERT-base (110M params):

1. **Rotary Positional Embeddings (RoPE)**: Replaces absolute positional embeddings. Encodes relative position through rotation matrices applied to query/key vectors. Enables extrapolation to longer sequences (up to 8192 tokens) without retraining.

2. **Flash Attention 2**: Hardware-aware attention implementation that fuses the attention computation into a single GPU kernel, reducing memory from O(n^2) to O(n) and significantly improving throughput.

3. **GeGLU Activations**: Replaces GELU with a gated variant (GeGLU) in the feed-forward network. The gating mechanism `GeGLU(x) = GELU(xW1) * xW2` provides additional expressiveness.

4. **Unpadding**: Removes padding tokens from the computation graph entirely, processing only real tokens. Particularly beneficial for variable-length financial text.

5. **Fused QKV Projection**: Q, K, V projections are fused into a single `Wqkv` matrix. This is computationally efficient but has implications for LoRA (discussed below).

### Empirical Results

**Held-out evaluation**: ModernBERT + LoRA achieves 80.93% accuracy vs BERT-base + LoRA at 73.09% -- a +7.84pp advantage. This is the clearest signal that the architectural modernizations matter.

**Head-to-head 10-fold CV** (Table 7): ModernBERT wins 7 of 10 folds with a mean advantage of +1.09pp (85.27% vs 86.36%). Paired t-test: t=1.88, p=0.093. The smaller gap in-domain suggests ModernBERT's advantage is disproportionately in *transfer* -- its representations are more robust to distribution shift.

**On allAgree**: ModernBERT + LoRA reaches 93.29% vs BERT's 83.66% -- a +9.63pp gap on the cleanest FPB subset. This suggests BERT struggles more with clear-cut sentiment, while ModernBERT handles both easy and ambiguous cases better.

---

## 3. Training Configuration

### LoRA Setup
- **Rank**: r=16, alpha=32 (effective scaling = alpha/r = 2.0)
- **Target modules**: `Wqkv`, `out_proj`, `Wi`, `Wo` (ModernBERT-specific names)
- **Dropout**: 0.05
- **Trainable parameters**: 1.1M out of 149M (~0.7%)

A critical subtlety: ModernBERT's fused QKV projection means LoRA rank 16 is shared across Q, K, and V. The effective rank per component is ~5.3. This creates an implicit bottleneck that may act as additional regularization (see Section 6).

### Training Hyperparameters (LoRA)
- Optimizer: AdamW
- Learning rate: 2e-4 with cosine schedule
- Warmup: 10 steps
- Effective batch size: 32 (per-device 8 x gradient accumulation 4)
- Precision: FP16
- Epochs: 10 with early stopping on validation loss
- Gradient checkpointing: enabled

### Training Hyperparameters (Full Fine-Tuning)
- Learning rate: 2e-5 (10x lower than LoRA)
- Weight decay: 0.01
- Warmup ratio: 0.1
- Effective batch size: 16 (per-device 8 x gradient accumulation 2)
- Gradient checkpointing: enabled

### Training Data
Source: `neoyipeng/financial_reasoning_aggregated`, filtered to sentiment task, FPB (source 5) excluded.

| Source | Domain | N_train | NEG% | NEU% | POS% | Med. Words |
|---|---|---|---|---|---|---|
| 3 | Earnings calls (narrative) | 462 | 11.3% | 53.4% | 35.3% | 32 |
| 4 | Press releases / news | 1,557 | 3.4% | 58.7% | 37.9% | 60 |
| 8 | Earnings calls (Q&A) | 2,440 | 7.9% | 57.7% | 34.4% | 161 |
| 9 | Financial tweets | 4,184 | 13.9% | 68.4% | 17.6% | 15 |
| **Total** | | **8,643** | **10.2%** | **62.7%** | **27.1%** | --- |

Three critical data characteristics:
1. **Class imbalance**: Training NEGATIVE rate is 10.2% vs FPB's 12.5%. The model sees fewer negative examples than what FPB expects.
2. **Sub-domain bias**: Source 4 is 68% Canadian mining/resource press releases (mentions of TSX, mining, resource extraction). This creates a narrow sub-domain that may not transfer broadly.
3. **Length mismatch**: Source 8 (earnings Q&A) has median 161 words (mean 189, max 2,596). With max_length=512, a substantial fraction is silently truncated -- the model trains on incomplete texts.

FPB uses crowd-sourced human annotation (>=50% agreement threshold). The annotation methodology for other sources in the aggregated dataset is documented in their respective original papers.

---

## 4. DataBoost: Targeted Augmentation via Verbalized Sampling

DataBoost is a targeted data augmentation strategy that combines error mining with Verbalized Sampling (VS), a technique from Zhang et al. (2025).

### The Verbalized Sampling Technique

Standard LLM paraphrasing suffers from **mode collapse** -- the model generates near-identical outputs because RLHF training sharpens outputs toward the mode of the distribution. VS counteracts this by explicitly asking the model to:

1. Generate k candidate texts (default k=5)
2. Assign probability estimates to each candidate (must sum to ~1.0)
3. Order from most common (highest probability) to most unusual (lowest)

This forces the model to think about the full distribution of possible texts, not just the most likely one. VS achieves 1.6-2.1x more diverse outputs than direct prompting.

### VS-CoT (Chain of Thought) Variant

The paper uses the VS-CoT variant, which adds a reasoning phase before generation:

**Phase 1 (Analysis)**: Why did the classifier confuse this sample? What distinguishing linguistic cues separate the true label from the predicted label? Plan sub-domain coverage across 11 financial text types (earnings calls, analyst notes, news, SEC filings, press releases, social media, credit ratings, M&A, central bank communications, trading desk commentary, etc.).

**Phase 2 (Generation)**: Produce k diverse candidates spanning different sub-domains, each with a probability estimate. Texts are 10-40 words (matching FPB style).

**Phase 3 (Output)**: text, label, probability, source_variant, confusion_type.

### DataBoost Protocol

1. Train baseline model on aggregated data (FPB excluded)
2. Run inference on validation set -- identify 82 misclassified samples (17.1% error rate on 480 validation samples)
3. For each misclassified sample, generate 3 diverse paraphrases using VS-CoT via Claude, preserving the gold label but varying financial sub-domain, register, and sentence structure
4. This yields 246 augmented samples
5. Retrain with augmented data

### Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 77.29% | 66.60% |
| DataBoosted | 80.21% | 74.43% |
| **Delta** | **+2.92pp** | **+7.84pp** |

The disproportionate F1 gain (+7.84pp) vs accuracy gain (+2.92pp) confirms that augmentation disproportionately helps **minority classes** (NEGATIVE and POSITIVE). This makes sense: the misclassified samples are biased toward the tails, and VS-CoT generates diverse examples in exactly those underrepresented regions.

On FPB evaluation:

| Metric | Baseline | DataBoosted | Delta |
|---|---|---|---|
| FPB 50agree Acc | 80.91% | 82.56% | +1.65pp |
| FPB 50agree F1 | 78.10% | 80.52% | +2.42pp |
| FPB allAgree Acc | 93.24% | 95.14% | +1.90pp |
| FPB allAgree F1 | 91.92% | 94.17% | +2.25pp |

Only 246 synthetic samples move the needle by nearly 2pp on 4,846 FPB sentences -- high augmentation efficiency.

### Augmented Data Structure

The VS-augmented data (`data/vs_augmented_errors.csv`) contains:
- `text`: Generated financial sentence (10-40 words)
- `label`: Numeric label (0/1/2)
- `label_name`: NEGATIVE/NEUTRAL/POSITIVE
- `probability`: VS probability estimate (batches of 5 sum to ~1.0)
- `source_variant`: vs-cot
- `confusion_type`: e.g., "POSITIVE->NEUTRAL" (what the model got wrong)

Examples target specific confusion patterns like POSITIVE sentences containing cautious/factual language that a classifier might read as NEUTRAL. Each batch of 5 spans different financial sub-domains (equity research, trading commentary, press releases, analyst notes, SEC filings).

---

## 5. Fine-Tuned Model vs Claude Opus 4 (LLM Comparison)

### Setup

Claude Opus 4.6 was equipped with a purpose-built **financial sentiment engine skill** -- a structured prompt that:
- Uses direct classification (no CoT) for short texts, following Vamvourellis & Mehta (2025) finding that reasoning causes "overthinking" on financial sentiment
- Defines labels from an investor lens (POSITIVE = likely increases stock price)
- Adapts decision framework by text length (short: direct, medium: label-first, long: analogical reasoning)
- Includes calibrated probability guidelines (never 1.00 or 0.00)

Evaluation was blind: four independent Claude agents classified chunks of 723 samples (243 FPB + 478 non-FPB + 2 external) without seeing ground truth.

### Results

| Model | Accuracy | Macro F1 | Cost/sample | Latency/sample |
|---|---|---|---|---|
| ModernFinBERT (149M) | 83.13% | 79.57% | ~$0.00001 | ~2ms |
| Claude Opus 4.6 + skill | 72.75% | 70.33% | ~$0.008 | ~1-2s |

**Gap**: +10.38pp accuracy, +9.24pp F1 in favor of the fine-tuned model. ModernFinBERT is 800x cheaper and 500-1000x faster.

### Per-Domain Breakdown

| Text Type | ModernFinBERT | Claude | Gap |
|---|---|---|---|
| Earnings Calls (n=136) | 69.12% | 47.79% | +21.32pp |
| Mining/TSX-V (n=35) | 88.57% | 82.86% | +5.71pp |
| Press Release/Other (n=125) | 88.00% | 77.60% | +10.40pp |
| Social Media (n=182) | 87.36% | 77.47% | +9.89pp |

The largest gap is on earnings calls (+21.32pp) -- the domain where both models struggle most. ModernFinBERT's advantage is narrower on clean domains (mining press releases) and largest where the text is most complex and ambiguous.

### Claude's Error Patterns

From the blind benchmark results:
- **POSITIVE/NEUTRAL boundary confusion** dominates: 131 of 197 Claude errors
- NEGATIVE recall is highest (0.819) but over-triggers on neutrals
- NEUTRAL/MIXED is the most reliable class for Claude (F1=0.779)
- Rare polarity flips (POSITIVE->NEGATIVE: 7, NEGATIVE->POSITIVE: 5)

The non-blind evaluation (where Claude could see labels) scored 95.7% accuracy -- confirming the blind evaluation was fair and that the gap is about classification ability, not prompt issues.

---

## 6. LoRA as Implicit Regularizer (2x2 Experiment)

The paper runs a 2x2 factorial experiment: {LoRA, Full Fine-Tuning} x {Baseline, DataBoosted}.

| Fine-tuning | Data | Params | FPB 50agree Acc | FPB 50agree F1 |
|---|---|---|---|---|
| LoRA r=16 | Baseline | 1.1M | 80.91% | 78.10% |
| LoRA r=16 | DataBoosted | 1.1M | **82.56%** | **80.52%** |
| Full FT | Baseline | 149M | 77.22% | 69.37% |
| Full FT | DataBoosted | 149M | 81.37% | 79.77% |

Three findings:

1. **LoRA outperforms full FT**: +3.69pp on baseline, +1.19pp with DataBoost. LoRA's low-rank constraint prevents overfitting on the small (8,643 sample) training set -- it acts as an implicit regularizer.

2. **DataBoost is more impactful under full FT**: Full FT gains +4.15pp accuracy and +10.4pp F1 from DataBoost, vs +1.65pp accuracy under LoRA. Full FT has more capacity to overfit, so the augmented data provides a proportionally larger regularization benefit.

3. **LoRA + DataBoost remains the best configuration** at 82.56% accuracy -- matching the best of both worlds (regularization from LoRA + diversity from DataBoost).

The fused QKV projection in ModernBERT adds another dimension: LoRA r=16 shared across Q, K, V gives ~5.3 effective rank per component -- a tighter bottleneck than LoRA on standard BERT where each component gets its own rank-16 adapter.

---

## 7. Self-Training: A Negative Result

Self-training with pseudo-labels was attempted and failed. The protocol:

1. Train teacher model on DataBoosted set
2. Source ~30k unlabeled financial tweets (filtered 5-50 words, deduplicated)
3. Per-class top-k confidence thresholds (15%, 25%, 40% across rounds) to prevent NEUTRAL domination
4. Train fresh student on labeled + pseudo-labeled data
5. Repeat 3 rounds

**Result**: Round 1 added 3,217 pseudo-labeled samples but accuracy dropped from 82.56% to 80.54%. Validation accuracy decreased from 83.54% to 82.92%, triggering early stopping.

**Why it failed**:
1. **Domain mismatch**: Unlabeled data was Twitter financial text (informal, short, slang-heavy) while FPB evaluation is formal press-release sentences. The training data already contained 48% tweets + earnings calls, and adding more tweets pushed the distribution further from FPB's register.
2. **Overconfident teacher**: Mean and minimum confidence scores were ~1.0, meaning the model was nearly deterministic in its predictions. The pseudo-labels were essentially hard labels with no uncertainty signal, reinforcing existing decision boundaries rather than expanding coverage.

This is a well-documented negative result: more data doesn't help when it's from the wrong distribution, and overconfident pseudo-labels amplify rather than correct errors.

---

## 8. Multi-Seed Robustness

Five seeds (3407, 42, 123, 456, 789) were evaluated on the held-out protocol. All standard deviations are below 1%:

| Metric | Mean +/- Std |
|---|---|
| FPB 50agree Accuracy | 80.44% +/- 0.89% |
| FPB 50agree F1 | 77.05% +/- 0.98% |
| FPB allAgree Accuracy | 92.98% +/- 0.25% |
| Aggregated Test Accuracy | 77.71% +/- 0.53% |

The model is stable: seed choice does not meaningfully affect results. The LoRA + cosine schedule configuration provides robust convergence across random initializations.

---

## 9. Data Provenance and Contamination Audit

### Contamination Check

Three levels of deduplication were tested between the 9,123 training samples and 4,846 FPB evaluation samples:
- **Exact string match**: 0 matches
- **Fuzzy match** (>0.90 similarity): 0 matches
- **Semantic similarity** (cosine >0.95): 0 matches

The training set is **clean** -- no data leakage from FPB into the held-out training corpus.

### Sub-Domain Bias

Source 4 (press releases, 1,557 samples) is **68% Canadian mining and resource extraction** content. These texts frequently mention TSX, TSX-V, mining companies, and resource extraction terminology. This creates a narrow sub-domain bias: the model learns "financial press release" sentiment disproportionately from one industry and one country.

Despite this, ModernFinBERT achieves 88.57% accuracy on mining/TSX-V test samples and 88.00% on general press releases -- suggesting the bias hasn't dramatically hurt generalization to non-mining press releases.

### Annotation Quality

FPB uses crowd-sourced human annotation with a >=50% agreement threshold. The annotation methodology for other training sources is documented in their respective original datasets.

---

## 10. Confidence Calibration

Post-hoc temperature scaling was applied to make model confidence scores reflect actual accuracy.

### Method

Temperature T is learned by minimizing negative log-likelihood on calibration data:

T* = argmin_T [ -(1/N) * sum_i log( exp(z_{i,y_i} / T) / sum_c exp(z_{i,c} / T) ) ]

Temperature scaling divides all logits by T before softmax. T > 1 softens probabilities (reduces overconfidence), T < 1 sharpens them. Critically, temperature scaling preserves all predictions (argmax is unchanged).

### Results

Four calibration methods were compared:

| Method | ECE | Parameters |
|---|---|---|
| Pre-calibration (softmax) | 0.000242 | 0 |
| Temperature scaling (T=1.0003) | 0.000242 | 1 |
| Vector scaling | 0.000242 | 6 |
| Platt scaling | 0.000166 | 12 |

### Methodological Issue

The calibration experiment was run on the model's own training data (in-sample), not held-out data. The 5-fold CV approach described in the paper was skipped (commented out in the notebook). As a result:
- Accuracy on calibration data: 99.97%
- ECE: 0.0002 (near-perfect)
- Learned temperature: T=1.0003 (essentially no change)

These numbers are not informative for real-world calibration. The model appears perfectly calibrated because it's being evaluated on data it memorized. A proper calibration study would use the planned 5-fold CV or a held-out split.

The calibration config (`T=1.0003`) is released alongside the model but should be treated as a placeholder pending proper held-out calibration.

---

## 11. Inference Performance

Benchmarked on `neoyipeng/ModernFinBERT-base` with 100 iterations and 10 warmup:

| Device | Batch Size | p50 Latency | Throughput |
|---|---|---|---|
| CPU | 1 | 26.2ms | 27.4 samples/sec |
| CPU | 32 | 14.0ms/sample | 71.6 samples/sec |
| MPS (Apple Silicon) | 1 | 43.8ms | 22.5 samples/sec |
| MPS (Apple Silicon) | 32 | 9.4ms/sample | 106.7 samples/sec |

Peak throughput: **107 samples/sec on Apple Silicon at batch 32**. Single-sample latency is ~26ms on CPU and ~44ms on MPS.

MPS is slower than CPU for single samples (GPU kernel launch overhead) but scales much better with batching. The optimal batch size is 32 on both devices -- throughput plateaus beyond that.

For comparison, Claude Opus 4.6 processes at ~0.5-1.0 samples/sec. ModernFinBERT is 100-200x faster at batch inference.

---

## 12. Error Analysis (NB15 -- Designed but Unexecuted)

NB15 was designed to systematically categorize misclassifications by:

1. **Confusion type**: Which label pairs are most confused (e.g., NEG->NEU, POS->NEU)
2. **Text length**: Short (<15 words), medium (15-30), long (>30) error rates
3. **Linguistic patterns**: Regex-based detection of:
   - Hedging language: "may," "could," "might," "possibly," "likely"
   - Conditional statements: "if," "although," "however," "despite," "while"
   - Comparative language: "more than," "less than," "higher," "lower"
   - Implicit sentiment: absence of explicit sentiment keywords
4. **Financial domain**: Mining/resources, earnings, market/trading, general business
5. **Confidence analysis**: Do high-confidence errors cluster in specific patterns?

The paper's discussion section mentions these categories but marks the concrete results as "TBD." The error analysis framework exists but has not been executed to produce numbers.

---

## 13. Earnings Call Analysis (NB17a/17b -- Designed but Unexecuted)

Earnings calls are the model's weakest domain (69.12% accuracy vs 87-88% elsewhere). Two notebooks were designed to address this:

**NB17a (Error Mining)**: Train baseline, analyze truncation impact per source (Source 8 median 161 words, max 2,596), mine errors specifically from earnings call data, export tagged by linguistic pattern for VS augmentation.

**NB17b (Improvement)**: Use VS-CoT to generate targeted augmentation for earnings call errors, retrain, and measure improvement.

Neither has been executed. This is the most promising avenue for improvement given that earnings calls represent both the largest domain gap and the most practically valuable text type.

---

## 14. Long-Context Ablation (NB18 -- Designed but Unexecuted)

ModernBERT supports 8192 tokens via RoPE, but all experiments hardcode max_length=512. NB18 was designed to test 512 vs 1024 vs 2048 token contexts across 3 seeds (9 training runs total).

Key hypothesis: Earnings call Q&A texts (Source 8, median 161 words, max 2,596) are silently truncated at 512 tokens. Extending context should improve accuracy on these long texts.

The training function adapts batch size to context length:
- 512: batch 8, grad accum 4 (effective 32)
- 1024: batch 4, grad accum 8 (effective 32)
- 2048: batch 2, grad accum 16 (effective 32)

Estimated runtime: ~9 hours on T4. Not yet executed.

---

## 15. Multi-Benchmark Evaluation (NB19 -- Designed but Unexecuted)

NB19 extends evaluation beyond FPB to three additional benchmarks:

1. **FPB sentences_50agree** (4,846 press-release sentences) -- primary benchmark
2. **FPB sentences_allagree** (2,264 unanimous-agreement) -- difficulty control
3. **Twitter Financial News Sentiment** -- social media domain
4. **FiQA 2018 Task 1** -- financial opinions with continuous sentiment scores mapped to 3-class via thresholds (default +/-0.2)

Compares ModernFinBERT against ProsusAI/finbert (in-domain baseline) and finbert-tone (zero-shot baseline). Includes:
- Automatic label remapping detection (different models use different label orderings)
- Sanity checks on obvious sentences
- FiQA threshold sensitivity analysis (+/-0.1, +/-0.2, +/-0.3)
- LaTeX table generation for paper insertion

Estimated runtime: ~1 hour on T4 (evaluation only). Not yet executed.

---

## 16. Skills Architecture

The project includes two Claude Code skills that serve as reusable tools:

### Financial Sentiment Engine Skill

A structured prompt for Claude that implements direct classification of financial sentiment. Key design decisions:
- **No chain-of-thought for short text** -- based on Vamvourellis & Mehta (2025) finding that reasoning causes "overthinking"
- **Investor-lens labels**: POSITIVE = likely increases stock price, not general positivity
- **Text-length-dependent framework**: Short (direct), medium (label-first/LIRA), long (analogical reasoning)
- **Calibrated probabilities**: High 0.80-0.95, Moderate 0.60-0.79, Low 0.40-0.55

In the blind benchmark, this skill achieved 72.8% accuracy / 0.703 F1 on 723 samples -- significantly below ModernFinBERT's 83.1%.

### Verbalized Sampling Augment Skill

A data augmentation skill that implements VS-CoT for generating diverse financial sentiment training data. Used to produce the DataBoost augmentation (246 samples in one experiment, 410 in another).

---

## 17. Limitations and Open Questions

### Acknowledged Limitations (from paper)
1. Single benchmark evaluation (FPB only -- NB19 designed but unrun)
2. Base models only (110-149M parameters)
3. English only
4. Single full fine-tuning configuration (hyperparameters not tuned)
5. 68% Canadian mining bias in Source 4
6. Heterogeneous training data (15-161 median words, 3.4-13.9% NEG rates across sources)
7. Limited architecture comparison (DeBERTa-v3 collapsed due to LoRA + gradient checkpointing incompatibility)
8. Self-training used only financial tweets
9. Calibration scope limited (temperature learned on training distribution)

### Additional Observations

1. **The calibration experiment needs re-running**: The NB16 results (ECE=0.0002, T=1.0003) are from in-sample evaluation and are not informative. The planned 5-fold CV was skipped.

3. **Error analysis sections in the paper are incomplete**: Sections 5.6 (Error Analysis) and Table 11 (Calibration) are marked TBD, awaiting NB15 and NB16 results respectively.

4. **Multi-seed results in the paper are partial**: Table 8 shows summary statistics but individual per-seed values are marked TBD.

5. **Earnings call performance (69%) is the largest gap**: This is both the weakest domain and the most potentially valuable for finance applications. The NB17a/17b pipeline was designed to address this but hasn't been run.

6. **No domain pre-training comparison**: The paper explicitly fine-tunes from vanilla ModernBERT to isolate architecture contribution, but doesn't compare against a version with continued pre-training on financial text. The TODOS.md lists this as a P2 priority requiring 10-50GB of financial text and ~2 weeks.

---

## 18. What the Paper Does Well

1. **Methodological rigor**: The held-out protocol, 3-level deduplication audit, and 2x2 factorial design show careful experimental control. Negative results (self-training) are reported honestly.

2. **Practical framing**: The Claude comparison answers the real question practitioners ask: "should I fine-tune or just prompt GPT-4?" The answer (fine-tune, 10pp better, 800x cheaper) is clear and actionable.

3. **DataBoost efficiency**: 246 samples producing 7.8pp F1 improvement is remarkably sample-efficient. The VS-CoT technique for targeted augmentation of misclassified examples is a practical contribution.

4. **LoRA regularization insight**: The 2x2 experiment provides clean evidence that LoRA isn't just a computational convenience -- it actively improves generalization on small datasets through implicit regularization.

5. **Reproducibility**: All notebooks run on Kaggle free GPU tier (~20 GPU hours total), all code/data/models are public. Total compute cost is minimal (~$11 for training + Claude API).

---

## 19. Project Status Summary

| Component | Status | Notes |
|---|---|---|
| Paper (main.tex) | ~85% complete | Error analysis, calibration, multi-seed details TBD |
| NB01-NB04 | Complete | Core experiments (held-out, DataBoost, Claude, CV) |
| NB06-NB07 | Complete | Multi-seed, self-training |
| NB09b | Complete | BERT vs ModernBERT CV |
| NB12 | Complete | LoRA vs full FT (2x2) |
| NB13 | Complete | DeBERTa baseline (failed due to incompatibility) |
| NB14 | Complete | Production model training |
| NB15 | Code ready, unexecuted | Error analysis |
| NB16 | Executed but flawed | Calibration on in-sample data |
| NB17a/17b | Code ready, unexecuted | Earnings call improvement |
| NB18 | Code ready, unexecuted | Long-context ablation (~9hr on T4) |
| NB19 | Code ready, unexecuted | Multi-benchmark eval (~1hr on T4) |
| Model on HuggingFace | Published | `neoyipeng/ModernFinBERT-base` |
| Dataset on HuggingFace | Published | `neoyipeng/financial_reasoning_aggregated` |
| Demo app | Built | Streamlit app in `demo/` |

---

## 20. Key Numerical Claims (Quick Reference)

| Claim | Number | Source |
|---|---|---|
| Protocol gap | 6.44pp (86.88% vs 80.44%) | NB01 vs NB04 |
| ModernBERT vs BERT (held-out) | +7.84pp accuracy | NB09 |
| ModernBERT vs BERT (CV) | +1.09pp, p=0.093 | NB09b |
| DataBoost F1 improvement | +7.84pp | NB02 |
| DataBoost accuracy improvement | +2.92pp | NB02 |
| Fine-tuned vs Claude | +10.38pp accuracy | NB03 |
| Cost advantage vs Claude | 800x cheaper | NB03 |
| Speed advantage vs Claude | 500-1000x faster | NB03 |
| LoRA vs Full FT (baseline) | +3.69pp | NB12 |
| LoRA vs Full FT (DataBoost) | +1.19pp | NB12 |
| DataBoost under Full FT | +4.15pp acc, +10.4pp F1 | NB12 |
| Multi-seed std (accuracy) | <1% across all metrics | NB06 |
| FPB allAgree (best) | 95.14% acc, 94.17% F1 | NB02 |
| Training data contamination | 0 matches (exact, fuzzy, semantic) | NB09a/11 |
| Total GPU compute | ~20 hours on T4 | Paper |
| Total training cost | ~$11 | Paper |
| Trainable parameters (LoRA) | 1.1M / 149M (0.7%) | Paper |
| Inference throughput (peak) | 107 samples/sec (MPS batch 32) | Benchmark |
| Earnings call accuracy | 69.12% (weakest domain) | NB03 |
