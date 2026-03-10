# Financial Sentiment Engine — Benchmark Results

## Overview

Blind evaluation of the `financial-sentiment-engine` skill on a 723-row test dataset covering diverse financial text types: Financial PhraseBank sentences, earnings call Q&A transcripts, social media/tweets, press releases, and Canadian mining/TSX-V announcements.

## Methodology

- **Model**: Claude Opus 4.6 (via Claude Code subagents)
- **Evaluation type**: Blind — agents received only the text column, no ground truth labels
- **Skill instructions**: Agents were given the skill's classification rules (3-label taxonomy, text-length decision framework, investor-lens definitions) but NOT the full SKILL.md file
- **Parallelization**: Dataset split into 4 chunks of ~180 rows, processed by 4 independent agents concurrently
- **Date**: 2026-03-11

## Dataset

- **Source**: `MFB-SENTIMENT-SKILL - test_dataset.csv`
- **Rows**: 723
- **Label distribution**:
  - NEUTRAL/MIXED: 432 (59.8%)
  - POSITIVE: 197 (27.2%)
  - NEGATIVE: 94 (13.0%)

## Results

| Metric | Score |
|---|---|
| **Accuracy** | 72.8% (526/723) |
| **Macro F1** | 0.703 |
| **Macro Precision** | 0.686 |
| **Macro Recall** | 0.732 |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| POSITIVE | 0.642 | 0.619 | 0.630 | 197 |
| NEGATIVE | 0.611 | 0.819 | 0.700 | 94 |
| NEUTRAL/MIXED | 0.803 | 0.757 | 0.779 | 432 |

### Confusion Matrix

| | Pred POSITIVE | Pred NEGATIVE | Pred NEUTRAL/MIXED |
|---|---|---|---|
| **GT POSITIVE** | 122 | 7 | 68 |
| **GT NEGATIVE** | 5 | 77 | 12 |
| **GT NEUTRAL/MIXED** | 63 | 42 | 327 |

## Error Analysis

**Total errors**: 197 (27.2%)

| Pattern | Count | Notes |
|---|---|---|
| POSITIVE → NEUTRAL/MIXED | 68 | Mild positives classified as neutral |
| NEUTRAL/MIXED → POSITIVE | 63 | Neutral announcements classified as positive |
| NEUTRAL/MIXED → NEGATIVE | 42 | Neutral items flagged as negative |
| NEGATIVE → NEUTRAL/MIXED | 12 | Subtle negatives missed |
| POSITIVE → NEGATIVE | 7 | Rare polarity flips |
| NEGATIVE → POSITIVE | 5 | Rare polarity flips |

### Key Observations

1. **POSITIVE ↔ NEUTRAL/MIXED confusion dominates** (131 of 197 errors). The boundary between mild positive news and neutral factual reporting is the hardest distinction.
2. **NEGATIVE recall is strongest** (0.819) — the skill catches bad news well, but over-triggers on some neutral items (precision 0.611).
3. **NEUTRAL/MIXED is the most reliable class** (F1=0.779), which is notable since it's typically the hardest to classify.

## Reference Benchmarks

| Method | Macro F1 | Source |
|---|---|---|
| GPT-4o No-CoT | 0.78 | Vamvourellis & Mehta, 2025 |
| **This skill (blind)** | **0.703** | This evaluation |
| DistilRoBERTa fine-tuned | ~0.85 | Domain-tuned baselines |

## Notes

- The blind eval agents received a condensed version of the skill rules, not the full SKILL.md with examples. Performance may improve when the full skill is loaded.
- The dataset contains non-English text and highly domain-specific TSX-V mining announcements, which add difficulty.
- A prior non-blind evaluation (where ground truth was visible during classification) scored 95.7% accuracy / 0.944 Macro F1 — confirming significant label leakage bias.

## Files

- `SKILL.md` — The skill definition
- `blind_benchmark_results.csv` — Full predictions with ground truth (723 rows: text, labels, blind_predicted_label)
