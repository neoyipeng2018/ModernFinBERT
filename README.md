# ModernFinBERT

A systematic empirical study of ModernBERT for financial sentiment analysis, comprising nine controlled experiments on the FinancialPhraseBank (FPB) benchmark.

**Paper**: [ModernFinBERT: A Systematic Study of ModernBERT for Financial Sentiment Analysis](paper/main.pdf)
**Model**: [neoyipeng/ModernFinBERT-base](https://huggingface.co/neoyipeng/ModernFinBERT-base)
**Dataset**: [neoyipeng/financial_reasoning_aggregated](https://huggingface.co/datasets/neoyipeng/financial_reasoning_aggregated)

## Key Results

| Protocol | Accuracy | Macro F1 |
|----------|----------|----------|
| FPB 50agree (10-fold CV) | 86.88% +/- 0.96% | 85.40% +/- 1.39% |
| FPB 50agree (held-out) | 80.44% | 77.05% |
| FPB 50agree (held-out + DataBoost) | 82.56% | 80.52% |
| FPB allAgree (held-out + DataBoost) | 95.14% | 94.17% |
| vs Claude Opus 4.6 (723 samples) | 83.13% vs 72.75% | +10.4pp |

## Experiments

| NB | Experiment | Key Finding |
|----|-----------|-------------|
| 01 | Held-out evaluation | 80.44% on FPB with FPB excluded from training |
| 02 | DataBoost augmentation | +2.9pp accuracy, +7.8pp F1 from targeted paraphrases |
| 03 | vs Claude Opus 4.6 | Fine-tuned model wins by 10.4pp, 800x cheaper |
| 04 | 10-fold CV on FPB | 86.88% +/- 0.96%, comparable to published FinBERT |
| 06 | Multi-seed robustness | Stable across 5 seeds (std < 1%) |
| 07 | Self-training | Negative result: domain-mismatched pseudo-labels hurt |
| 09b | BERT vs ModernBERT CV | ModernBERT wins 7/10 folds, +1.09pp mean |
| 12 | LoRA vs full fine-tuning | LoRA outperforms full FT by +3.69pp (regularization) |
| 15 | Error analysis | Linguistic patterns driving misclassification |
| 16 | Confidence calibration | Model already well-calibrated (ECE ~0.02%) |
| 18 | Long-context ablation | Context length effect on earnings call accuracy |
| 19 | Multi-benchmark eval | Generalization across FPB, FiQA, Twitter Financial |

## Quick Start

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="neoyipeng/ModernFinBERT-base")
result = classifier("The company reported strong quarterly earnings, beating analyst expectations.")
print(result)
# [{'label': 'POSITIVE', 'score': 0.95}]
```

## Project Structure

```
ModernFinBERT/
├── notebooks/           # Experiment notebooks (NB01-NB19)
├── scripts/             # Inference, benchmarking, data audit scripts
├── paper/               # LaTeX paper source and figures
├── data/                # Raw, cleaned, and processed datasets
├── results/             # JSON results from all experiments
├── demo/                # Production demo app
├── MODEL_CARD.md        # HuggingFace model card
├── plan.md              # Implementation plan for next experiments
└── research.md          # Detailed research findings
```

## Architecture

- **Base**: ModernBERT-base (149M parameters) with RoPE, Flash Attention 2, GeGLU
- **Fine-tuning**: LoRA (r=16, alpha=32) targeting Wqkv, out_proj, Wi, Wo
- **Training**: AdamW, lr=2e-4, cosine schedule, FP16, gradient checkpointing

## Links

- [HuggingFace Model](https://huggingface.co/neoyipeng/ModernFinBERT-base)
- [Training Dataset](https://huggingface.co/datasets/neoyipeng/financial_reasoning_aggregated)
- [FinancialPhraseBank](https://huggingface.co/datasets/financial_phrasebank)
