---
language: en
license: apache-2.0
tags:
  - finance
  - sentiment-analysis
  - modernbert
  - financial-nlp
datasets:
  - neoyipeng/financial_reasoning_aggregated
  - financial_phrasebank
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
model-index:
  - name: ModernFinBERT-base
    results:
      - task:
          type: text-classification
          name: Financial Sentiment Analysis
        dataset:
          name: FinancialPhraseBank (sentences_50agree)
          type: financial_phrasebank
          config: sentences_50agree
        metrics:
          - name: Accuracy (5-fold CV)
            type: accuracy
            value: TBD
          - name: Macro F1 (5-fold CV)
            type: f1
            value: TBD
---

# ModernFinBERT: Financial Sentiment Analysis

ModernFinBERT is a financial sentiment analysis model based on the [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) architecture (149M parameters). It classifies financial text into three sentiment categories: **NEGATIVE**, **NEUTRAL**, and **POSITIVE**.

## Key Features

- Built on ModernBERT-base with rotary positional embeddings, Flash Attention, and GeGLU activations
- Full fine-tuning on a diverse financial corpus spanning earnings calls, press releases, financial tweets, and analyst communications
- Augmented with DataBoost (targeted paraphrases of difficult examples via Verbalized Sampling)
- Fast inference: suitable for high-volume production use

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="neoyipeng/ModernFinBERT-base")

result = classifier("The company reported strong quarterly earnings, beating analyst expectations.")
print(result)
# [{'label': 'POSITIVE', 'score': 0.95}]
```

### Batch inference

```python
texts = [
    "Revenue declined 15% year-over-year due to market headwinds.",
    "The board approved a new share buyback program.",
    "The company maintained its quarterly dividend at $0.50 per share.",
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{result['label']} ({result['score']:.2f}): {text[:60]}...")
```

## Training Data

The model was trained on the complete [neoyipeng/financial_reasoning_aggregated](https://huggingface.co/datasets/neoyipeng/financial_reasoning_aggregated) dataset including FinancialPhraseBank, augmented with 410 DataBoost samples generated via Verbalized Sampling.

| Source | Domain | Samples |
|--------|--------|---------|
| Earnings calls (narrative) | Corporate transcripts | 462 |
| Press releases / news | Financial news | 1,557 |
| Earnings calls (Q&A) | Analyst Q&A | 2,440 |
| Financial tweets | Social media | 4,184 |
| FinancialPhraseBank | Press releases | 4,846 |
| DataBoost augmentation | Synthetic paraphrases | 410 |
| **Total** | | **~13,900** |

## Evaluation

### 5-Fold Cross-Validation (Full Dataset)
| Metric | Score |
|--------|-------|
| Accuracy | 81.41% ± 1.09% |
| Macro F1 | 77.66% ± 0.79% |

### Held-Out Evaluation (FPB excluded from training)
Results from the research paper using LoRA fine-tuning on non-FPB data only:

| Evaluation Set | Accuracy | Macro F1 |
|---------------|----------|----------|
| FPB 50agree | 82.56% | 80.52% |
| FPB allAgree | 95.14% | 94.17% |

## Training Configuration

- **Architecture:** ModernBERT-base (149M parameters, all unfrozen)
- **Learning rate:** 2e-5
- **Weight decay:** 0.01
- **Batch size:** 16 effective (8 per device, gradient accumulation 2)
- **Epochs:** 10 with early stopping
- **Precision:** FP16 mixed precision
- **Hardware:** NVIDIA Tesla T4 (16GB)

## Limitations

- **English only**: Trained and evaluated exclusively on English financial text.
- **Three-class taxonomy**: Classifies as NEGATIVE/NEUTRAL/POSITIVE only. Does not provide fine-grained sentiment or aspect-level analysis.
- **Domain bias**: Training data includes 68% Canadian mining press releases in the news subset, which may affect performance on other financial sub-domains.
- **Not domain-pre-trained**: Unlike FinBERT variants, this model does not include continued MLM pre-training on financial text. It is fine-tuned directly from the general-purpose ModernBERT-base checkpoint.

## Citation

```bibtex
@article{neo2026modernfinbert,
  title={ModernFinBERT: A Systematic Study of ModernBERT for Financial Sentiment Analysis},
  author={Neo, Yipeng},
  journal={arXiv preprint},
  year={2026}
}
```

## Paper

For full experimental details, evaluation protocols, and analysis, see the [ModernFinBERT paper](https://github.com/neoyipeng2018/ModernFinBERT).
