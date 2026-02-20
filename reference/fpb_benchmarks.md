# FinancialPhraseBank (FPB) Benchmark Results from Literature

## Papers

### 1. ProsusAI/finbert (Araci, 2019)
- **Title**: FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models
- **Author**: Dogu Araci
- **Year**: 2019
- **URL**: https://arxiv.org/abs/1908.10063
- **HuggingFace**: https://huggingface.co/ProsusAI/finbert
- **Eval protocol**: Trained on FPB (80/20 train-test, 20% of train for val)

#### FPB Sentences_50Agree (~4,845 samples)
| Model          | Accuracy | F1   |
|----------------|----------|------|
| LPS            | 0.71     | 0.71 |
| HSC            | 0.71     | 0.76 |
| LSTM           | 0.71     | 0.64 |
| LSTM+ELMo      | 0.75     | 0.70 |
| ULMFit         | 0.83     | 0.79 |
| **FinBERT**    | **0.86** | **0.84** |

#### FPB Sentences_AllAgree (~2,264 samples)
| Model          | Accuracy | F1   |
|----------------|----------|------|
| LPS            | 0.79     | 0.80 |
| HSC            | 0.83     | 0.86 |
| LSTM           | 0.81     | 0.74 |
| LSTM+ELMo      | 0.84     | 0.77 |
| ULMFit         | 0.93     | 0.91 |
| FinSSLX        | 0.91     | 0.88 |
| **FinBERT**    | **0.97** | **0.95** |

**Note**: These results use in-domain train/test splits of FPB itself.

---

### 2. yiyanghkust/finbert-tone (Yang, UY, Huang, 2020)
- **Title**: FinBERT: A Pretrained Language Model for Financial Communications
- **Authors**: Yi Yang, Mark Christopher Siy UY, Allen Huang
- **Year**: 2020 (published in Contemporary Accounting Research, 2023)
- **URL**: https://arxiv.org/abs/2006.08097
- **HuggingFace**: https://huggingface.co/yiyanghkust/finbert-tone
- **Eval protocol**: 90/10 random split, averaged over 10 runs. FPB agreement level unspecified.

#### FPB Results (Table 2, 10-run average accuracy)
| Model                          | Accuracy |
|--------------------------------|----------|
| BERT (cased)                   | 0.755    |
| BERT (uncased)                 | 0.835    |
| FinBERT-BaseVocab (cased)      | 0.856    |
| FinBERT-BaseVocab (uncased)    | 0.870    |
| FinBERT-FinVocab (cased)       | 0.864    |
| **FinBERT-FinVocab (uncased)** | **0.872**|

**Note**: finbert-tone was fine-tuned on 10K analyst report sentences, not FPB. FPB used as eval benchmark.

---

### 3. FinBERT-IJCAI (Liu et al., 2020)
- **Title**: FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining
- **Authors**: Zhuang Liu, Degen Huang, Kaiyu Huang, Zhuang Li, Jun Zhao
- **Year**: 2020
- **Venue**: IJCAI 2020
- **URL**: https://www.ijcai.org/proceedings/2020/622

| Metric   | Score |
|----------|-------|
| Accuracy | 0.94  |
| F1       | 0.93  |

---

### 4. Recent LLM Evaluation (2025)
- **Title**: Reasoning or Overthinking: Evaluating Large Language Models on Financial Sentiment Analysis
- **URL**: https://arxiv.org/html/2506.04574v1

#### Macro F1 by Agreement Level
| Model              | 50-65% | 66-74% | 75-99% | 100% agree |
|--------------------|--------|--------|--------|------------|
| FinBERT-Prosus     | 0.722  | 0.786  | 0.885  | 0.962      |
| FinBERT-Tone       | 0.436  | 0.566  | 0.736  | 0.897      |
| GPT-4o (No-CoT)    | 0.561  | 0.677  | 0.727  | 0.895      |
| GPT-4.1 (No-CoT)   | 0.552  | 0.658  | 0.734  | 0.890      |

---

### 5. finbert-lc (2024)
- **Title**: Financial Sentiment Analysis: Leveraging Actual and Synthetic Data
- **URL**: https://arxiv.org/html/2412.09859v1

#### FPB Sentences_AllAgree
| Model          | Accuracy | F1   |
|----------------|----------|------|
| FinBERT (Araci)| 0.97     | 0.95 |
| **finbert-lc** | **0.97** | **0.96** |

#### FPB Sentences_50Agree
| Model          | Accuracy | F1   |
|----------------|----------|------|
| FinBERT (Araci)| 0.86     | 0.84 |
| **finbert-lc** | **0.89** | **0.88** |

---

## Important Evaluation Context

**ProsusAI/finbert and most baselines** train and test on FPB using in-domain splits (80/20 or 90/10). This means the model sees FPB training data during fine-tuning.

**ModernFinBERT (ours)** is trained on aggregated financial sentiment data *excluding* FPB (source 6 filtered out). FPB serves as a fully held-out, zero-shot transfer benchmark. This is a stricter evaluation setting.

Achieving comparable or better accuracy under this held-out protocol is a stronger result than matching in-domain fine-tuning numbers.
