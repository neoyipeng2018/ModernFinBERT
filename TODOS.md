# TODOS

## P2: Domain-Specific Continued Pre-Training (ModernFinBERT v2)
**What:** Continue masked language model pre-training on a large financial corpus (SEC filings, earnings transcripts, financial news) before fine-tuning for sentiment.
**Why:** Current ModernFinBERT is fine-tuned from vanilla ModernBERT-base without domain adaptation. Domain pre-training (as done by FinBERT variants) could close the gap to in-domain SOTA (89-94%). This is the natural "v2" of the model.
**Effort:** L (human: ~2 weeks / CC: ~4-6 hours excluding data collection)
**Depends on:** Access to a large financial text corpus (10-50GB). Consider using SEC-EDGAR, financial news APIs, or existing corpora like FinCorpus.

## ~~P2: Multi-Benchmark Evaluation~~ DONE
Implemented in NB19 (`notebooks/19_multi_benchmark.ipynb`). Evaluates ModernFinBERT, ProsusAI/finbert, and finbert-tone on FPB 50agree, FPB allAgree, Twitter Financial News Sentiment, and FiQA 2018. Includes label remapping verification, FiQA threshold sensitivity, and LaTeX table generation.

## P3: Statistical Significance for All Comparisons
**What:** Run DataBoost and held-out experiments with multiple seeds to compute confidence intervals and significance tests for all key claims (not just the CV comparison).
**Why:** Currently only the 10-fold CV comparison (ModernBERT vs BERT) has a significance test (p=0.093). The DataBoost improvement (+2.9pp) and held-out architecture gap (+7.84pp) lack significance testing.
**Effort:** M (human: ~3 days / CC: ~2 hours GPU time)
**Depends on:** Kaggle GPU quota.

## P3: Community Held-Out Evaluation Leaderboard
**What:** Create a standardized benchmark where researchers submit held-out evaluation results using the protocol from this paper. Host on GitHub or HuggingFace.
**Why:** The held-out evaluation protocol is the paper's key methodological contribution. A community leaderboard would drive adoption and establish it as a standard.
**Effort:** L (human: ~2 weeks / CC: ~4 hours)
**Depends on:** Paper publication and community interest.

## P3: Update NB03 Model String for Reproducibility
**What:** Change `model="claude-opus-4-20250514"` in NB03 to match the actual Claude Opus 4.6 model ID used via agent, or add a comment explaining that the experiment was run via Claude Code agent.
**Why:** Someone running the notebook as-is would use a different model than what the paper reports.
**Effort:** S (5 min)
**Depends on:** Nothing.
