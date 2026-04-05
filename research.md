# Deep Research: Data Sources for Retraining ModernFinBERT

> **Goal**: Find all possible in-domain data sources (labeled and unlabeled) to train the best financial sentiment model, given that Financial PhraseBank, FiQA, and Financial Twitter are considered out-of-domain / already known.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State: What ModernFinBERT Uses](#2-current-state)
3. [Tier 1: High-Impact Labeled Datasets](#3-tier-1-labeled)
4. [Tier 2: Moderate-Impact Labeled Datasets](#4-tier-2-labeled)
5. [Tier 3: Niche / Specialized Labeled Datasets](#5-tier-3-labeled)
6. [Unlabeled Corpora for Domain-Adaptive Pretraining](#6-unlabeled-corpora)
7. [Market-Reaction Labeled Datasets (Distant Supervision)](#7-market-reaction)
8. [What the Best Financial LMs Used for Training](#8-what-top-models-used)
9. [State-of-the-Art Training Methods](#9-sota-methods)
10. [Recommended Training Pipeline](#10-recommended-pipeline)
11. [Complete Dataset Reference Table](#11-reference-table)

---

## 1. Executive Summary <a id="1-executive-summary"></a>

The research uncovered **40+ labeled datasets** and **15+ unlabeled corpora** beyond the three out-of-domain benchmarks (FPB, FiQA, Twitter). The single highest-impact finding is:

**The labeled training pool can grow from ~14K samples to ~250K+ clean examples** by combining:
- NOSIBLE Financial Sentiment (100K, multi-LLM consensus labels, ODC-By license)
- FinGPT Sentiment aggregation (76K instruction-formatted, MIT)
- TimKoornstra aggregated tweets (38K, MIT)
- JanosAudran SEC reports (20.5M sentence-level, market-reaction labels, Apache 2.0)
- SEntFiN entity-aware headlines (10.7K)
- StockEmotions (10K, 12 emotion classes)
- Gold Commodity Headlines (11.4K)
- FinEntity (979 entity-level)
- FOMC Hawkish-Dovish (496)

**For domain-adaptive pretraining**, the open-source ceiling is:
- PleIAs/SEC: 7.25 billion words of 10-K filings (1993-2024)
- EDGAR-CORPUS: 220K filings, billions of tokens (Apache 2.0)
- Financial-News-Multisource: 57M rows across 24 news sources
- Earnings call transcripts: 33K+ transcripts across 685 companies

**Key methodological insight**: The optimal pipeline is DAPT (continued MLM on financial corpus) -> TAPT (MLM on task-specific unlabeled data) -> supervised fine-tuning with contrastive loss. This consistently produces the best results across all recent papers.

---

## 2. Current State: What ModernFinBERT Uses <a id="2-current-state"></a>

| Source | Domain | Samples | Median Length |
|--------|--------|---------|---------------|
| Earnings Calls (Narrative) | Corporate transcripts | 513 | 32 words |
| Press Releases & News | Financial news | 1,730 | 60 words |
| Financial PhraseBank | Press releases | 4,846 | 21 words |
| Earnings Calls (Q&A) | Analyst Q&A | 2,711 | 161 words |
| Financial Tweets | Social media | 4,649 | 15 words |
| **Total** | | **14,449** | |

**Current best results**: 86.88% accuracy (10-fold CV on FPB), 80.44% (held-out). The model uses ModernBERT-base (149M params) with LoRA fine-tuning.

**The gap**: Current training data is small (14K) and domain-narrow (68% of news = Canadian mining companies). There's no domain-adaptive pretraining phase. The model goes straight from general ModernBERT to supervised fine-tuning.

---

## 3. Tier 1: High-Impact Labeled Datasets <a id="3-tier-1-labeled"></a>

These are the datasets most likely to produce the largest performance gains.

### 3.1 NOSIBLE Financial Sentiment (100K)

- **HuggingFace**: `NOSIBLE/financial-sentiment`
- **Size**: 100,000 examples (train split)
- **Labels**: 3-class (positive, negative, neutral)
- **Domain**: Financial news headlines
- **Labeling**: Multi-LLM consensus pipeline (8 LLM models) with active learning refinement and GPT-5.1 oracle validation. Cleaned and deduplicated.
- **License**: ODC-By (Open Data Commons Attribution) -- commercial use OK
- **Why it matters**: 20x larger than FPB with comparable label quality. This is the single biggest available labeled dataset for financial sentiment. Multi-LLM consensus reduces individual model bias. Includes source URL and domain metadata.
- **Risk**: LLM-generated labels may have systematic biases different from human annotators. Need to validate against FPB human labels.

### 3.2 FinGPT Sentiment Training Set (76K)

- **HuggingFace**: `flwrlabs/fingpt-sentiment-train`
- **Size**: 76,772 examples
- **Labels**: Two schemes -- 3-level (positive/neutral/negative) AND 7-level (strong negative through strong positive)
- **Domain**: Mixed financial news + tweets
- **Format**: Instruction-tuned (instruction/input/output)
- **License**: MIT
- **Why it matters**: Large, diverse, and includes fine-grained 7-level labels that can be collapsed to 3-class. The instruction format can be stripped to get raw text + label pairs. Aggregates multiple source datasets.

### 3.3 TimKoornstra Aggregated Financial Tweets (38K)

- **HuggingFace**: `TimKoornstra/financial-tweets-sentiment`
- **Size**: 38,091 tweets (Neutral=12,181, Bullish=17,368, Bearish=8,542)
- **Labels**: 3-class (Neutral/Bullish/Bearish)
- **Domain**: Financial social media
- **Source**: Aggregated from 9 sources: FiQA, IEEE DataPort, Kaggle, GitHub, Surge AI crypto/stock, HuggingFace. Deduplicated, sentiment-mapped.
- **License**: MIT
- **Why it matters**: The most comprehensive aggregation of financial tweet sentiment data. Multi-source reduces single-source bias. Clean deduplication.

### 3.4 JanosAudran SEC Financial Reports (20.5M sentences)

- **HuggingFace**: `JanosAudran/financial-reports-sec`
- **Size**: 20.5M sentences (large_lite config); 240K (small configs)
- **Labels**: Binary (positive/negative) derived from market reaction at 1-day, 5-day, and 30-day windows
- **Domain**: 10-K annual reports, segmented into 20 sections
- **Metadata**: CIK, tickers, SIC codes, filing dates, actual return percentages
- **License**: Apache 2.0
- **Why it matters**: Massive scale with market-derived labels. Can be used for distant supervision pretraining even if labels are noisy. The 30-day window labels may be more reliable than 1-day (less noise). Includes rich metadata for filtering.
- **Risk**: Market-reaction labels are noisy -- stock moves reflect many factors beyond filing sentiment. Best used as auxiliary training signal, not primary labels.

### 3.5 SEntFiN 1.0 (10.7K entity-aware)

- **Paper**: Sinha et al. (2022) -- "SEntFiN 1.0: Entity-Aware Sentiment Analysis for Financial News"
- **Size**: 10,753 news headlines; 2,847 headlines with multiple entities having conflicting sentiments; 5,000+ entity phrases
- **Labels**: Entity-level sentiment (positive, negative, neutral)
- **Domain**: Financial news headlines
- **Why it matters**: The only large-scale dataset capturing entity-level conflicting sentiment in financial text. Critical for real-world deployment where "Company A gained market share from struggling Company B" needs different labels per entity. Directly addresses a known weakness of sentence-level models.
- **Availability**: Via paper authors / GitHub

### 3.6 FinanceMTEB FinSent (10K)

- **HuggingFace**: `FinanceMTEB/FinSent`
- **Size**: 9,996 examples (train=9,000, test=1,000)
- **Labels**: 3-class (positive, negative, neutral)
- **Domain**: Analyst report style text
- **Why it matters**: Part of the FinanceMTEB benchmark suite. Analyst report domain is underrepresented in current training data. Clean benchmark-quality labels.

---

## 4. Tier 2: Moderate-Impact Labeled Datasets <a id="4-tier-2-labeled"></a>

### 4.1 StockEmotions (10K, 12 emotions)

- **Paper**: AAAI 2023
- **Size**: 10,000 StockTwits comments
- **Labels**: 12 fine-grained emotion classes + bullish/bearish sentiment
- **Domain**: StockTwits investor commentary
- **Why it matters**: Only dataset capturing investor emotions beyond binary/ternary sentiment. The 12 emotions (fear, greed, excitement, etc.) can be collapsed to 3-class but also enable multi-task training. Captures retail investor psychology.

### 4.2 FinGPT Sentiment Classification (47.5K)

- **HuggingFace**: `FinGPT/fingpt-sentiment-cls`
- **Size**: 47,557 examples
- **Labels**: Binary (positive/negative) with 20 instruction variations
- **Domain**: Mixed news + tweets
- **Why it matters**: Large binary dataset. Can supplement ternary training as auxiliary task or for pretraining the classification head.

### 4.3 Gold Commodity Headlines (11.4K)

- **Paper**: Commodity market NLP literature
- **Size**: 11,412 annotated commodity market headlines
- **Labels**: Sentiment labels
- **Domain**: Commodity markets (gold, oil, metals)
- **Why it matters**: Unique commodity market domain that's underrepresented in financial NLP. Commodity-specific language ("supply glut", "demand surge") differs from equity-focused text.

### 4.4 FOMC Hawkish-Dovish (496)

- **HuggingFace**: `TheFinAI/finben-fomc`
- **Size**: 496 examples (test set)
- **Labels**: 3-class (Hawkish, Dovish, Neutral)
- **Domain**: FOMC statements and minutes (1996-2022)
- **Paper**: "Trillion Dollar Words" (Shah et al., ACL 2023)
- **Why it matters**: Central bank communication domain. Monetary policy sentiment is distinct from corporate sentiment. Small but unique domain coverage.
- **Note**: The larger FOMC dataset (from the full paper) covers 1996-2022 and may be available from the authors.

### 4.5 FinEntity (979, entity-level)

- **HuggingFace**: `yixuantt/FinEntity`
- **Size**: 979 examples
- **Labels**: 3-class per entity span (Positive/Neutral/Negative), multiple entities per sentence
- **Domain**: Financial text, EMNLP 2023
- **License**: ODC-BY
- **Why it matters**: Character-offset entity annotations with per-entity sentiment. Complementary to SEntFiN for entity-level modeling.

### 4.6 FinMarBa (61K, market-reaction labels)

- **Paper**: arXiv 2507.22932 (2025)
- **Size**: 61,252 annotated headlines (Jan 2010 - Jan 2024), from Bloomberg Market Wraps
- **Labels**: Market-reaction-based sentiment (derived from actual stock/index movements)
- **Domain**: Bloomberg Market Wrap headlines
- **Why it matters**: Novel approach using objective market movements rather than human annotation. Eliminates annotator bias. 14 years of temporal coverage. May be available from authors.
- **Risk**: Same caveat as all market-reaction labels -- stock movements ≠ text sentiment. But Bloomberg Market Wraps are specifically *about* market movements, so alignment is higher than for generic news.

### 4.7 Kaggle Finance News Sentiments (32K)

- **Kaggle**: `antobenedetti/finance-news-sentiments`
- **Size**: 32,000+ labeled news items
- **Labels**: Sentiment labels
- **Domain**: Financial news
- **Why it matters**: Reasonably large, directly accessible. Quality may vary -- needs validation.

### 4.8 Reddit WallStreetBets Sentiment

- **HuggingFace**: `SocialGrep/reddit-wallstreetbets-aug-2021`
- **Size**: Full month of r/WallStreetBets (Aug 2021 -- peak meme stock era)
- **Labels**: Sentiment from in-house pipeline (on comments)
- **Domain**: Retail investor social media
- **Why it matters**: Captures meme stock / retail investor language. Highly informal register that stress-tests models.
- **Risk**: Pipeline-generated labels, quality uncertain. Peak meme stock era may not be representative.

### 4.9 TimKoornstra Synthetic Financial Tweets (1.4M)

- **HuggingFace**: `TimKoornstra/synthetic-financial-tweets-sentiment`
- **Size**: ~1.4M synthetic tweets
- **Labels**: 3-class (Neutral/Bullish/Bearish)
- **Domain**: Synthetic financial social media
- **Generator**: Nous-Hermes-2-Mixtral-8x7B-DPO
- **License**: MIT
- **Why it matters**: Massive scale. Synthetic data can help with rare class examples and distribution coverage. MIT license.
- **Risk**: Synthetic text may lack the noise, typos, and idiosyncratic patterns of real tweets. Could introduce model-specific artifacts. Best used as supplementary data, not primary.

---

## 5. Tier 3: Niche / Specialized Labeled Datasets <a id="5-tier-3-labeled"></a>

### 5.1 SemEval-2017 Task 5

- **Size**: Train: 1,142 headlines + 1,694 microblog posts; Test: 491 headlines + 794 posts
- **Labels**: Continuous sentiment score per entity target
- **Domain**: Financial microblogs and news headlines
- **Why it matters**: Standard academic benchmark with continuous scores and entity targets.

### 5.2 CryptoBERT / Crypto Sentiment Datasets

- Various crypto-specific sentiment datasets on HuggingFace and Kaggle
- **Domain**: Cryptocurrency markets
- **Why it matters**: Crypto language overlaps with but differs from traditional finance. Adds register diversity.
- **Risk**: Crypto vocabulary ("HODL", "mooning", "rug pull") may not transfer well to corporate finance.

### 5.3 Auditor Sentiment

- **HuggingFace**: `FinanceInc/auditor_sentiment`
- **Size**: 4,846 (train=3,880, test=969)
- **Labels**: 3-class
- **Domain**: Auditing
- **License**: Proprietary (DO NOT SHARE) -- limits usage
- **Note**: Derived from FPB with >75% agreement filter. Not independent data.

### 5.4 Fin-Fact (3.1K financial claims)

- **HuggingFace**: `amanrangapur/Fin-Fact`
- **Size**: 3,121 claims
- **Labels**: Fact-check labels, visualization bias labels, justifications
- **Domain**: Financial fact-checking
- **Why it matters**: Useful for multi-task training (fact verification as auxiliary task). Not directly sentiment but teaches financial reasoning.

### 5.5 FinQA (8.2K financial QA pairs)

- **HuggingFace**: `dreamerdeo/finqa`
- **Size**: 8,281 QA pairs over 2,800 financial reports (S&P 500, 1999-2019)
- **Labels**: Numerical answers with reasoning programs
- **License**: MIT
- **Why it matters**: Financial numerical reasoning. Useful for multi-task pretraining to build financial understanding.

### 5.6 FiNER-139 (1.1M sentences, NER)

- **HuggingFace**: `nlpaueb/finer-139`
- **Size**: 1,121,256 sentences from ~10K SEC filings
- **Labels**: 139 XBRL entity types (NER, not sentiment)
- **License**: CC-BY-SA-4.0
- **Why it matters**: Multi-task training candidate. Financial NER as auxiliary task during fine-tuning can improve representation quality.

### 5.7 Kaggle Sentiment-Labeled Headlines

- **Kaggle**: `cashbowman/sentiment-labeled-headlines`
- **Labels**: Scores 1-5 (granular)
- **Why it matters**: 5-point scale provides richer signal than 3-class. Can be bucketed for training.

---

## 6. Unlabeled Corpora for Domain-Adaptive Pretraining <a id="6-unlabeled-corpora"></a>

Domain-adaptive pretraining (DAPT) -- continued MLM on unlabeled financial text -- is the single most impactful training technique for financial NLP (Gururangan et al., ACL 2020). The following corpora are the best available.

### 6.1 SEC EDGAR Filings

| Dataset | HuggingFace Path | Size | Coverage | License |
|---------|-----------------|------|----------|---------|
| PleIAs/SEC | `PleIAs/SEC` | 7.25B words (245K filings) | 10-K, 1993-2024 | Not specified |
| EDGAR-CORPUS | `eloukas/edgar-corpus` | 220K filings, billions of tokens | 10-K, 1993-2020 | Apache 2.0 |
| SEC Filings Index | `arthrod/SEC_filings_1994_2024` | Metadata only | All form types, 1994-2024 | Not specified |

**Recommended**: Use `eloukas/edgar-corpus` (Apache 2.0 license, clean text with tables stripped) as primary DAPT corpus. Supplement with `PleIAs/SEC` for 2020-2024 coverage.

**Raw EDGAR access**: SEC EDGAR FULL-TEXT search at `efts.sec.gov/LATEST/search-index` provides direct access to all public filings. Free, no API key needed. Can download 10-K, 10-Q, 8-K filings directly.

### 6.2 Financial News

| Dataset | Path | Size | Coverage |
|---------|------|------|----------|
| Financial-News-Multisource | `Brianferrell787/financial-news-multisource` | 57.1M rows, 24 subsets | Bloomberg, Reuters, CNBC, NYT, Yahoo Finance, Reddit, 1990-2025 |
| FNSPID | `Zihan1004/FNSPID` | 15.7M news + 29.7M stock prices | S&P 500, 1999-2023 |

**Recommended**: `financial-news-multisource` is the largest unified financial news corpus available. Research-only license.

### 6.3 Earnings Call Transcripts

| Source | Size | Access |
|--------|------|--------|
| kurry/sp500_earnings_transcripts (HF) | 33,000+ transcripts, 685 companies | HuggingFace |
| jlh-ibm/earnings_call (HF) | 188 transcripts + stock prices, NASDAQ, 2016-2020 | HuggingFace |
| Seeking Alpha / Motley Fool transcripts | Millions of transcripts | Web scraping (ToS restrictions) |

**Recommended**: `kurry/sp500_earnings_transcripts` provides excellent coverage. Earnings calls capture spoken corporate financial language that's distinct from written news/filings.

### 6.4 Central Bank Communications

| Source | Size | Access |
|--------|------|--------|
| BIS speeches | 1996-present, thousands of speeches | bis.org (free) |
| ECB speeches | 1997-present | ecb.europa.eu (free) |
| Fed FOMC minutes | 1993-present | federalreserve.gov (free) |
| Fed Beige Book | 1996-present | federalreserve.gov (free) |

**Why it matters**: Central bank language has unique characteristics (hedging, forward guidance, policy stance) that make it a valuable pretraining domain. Models trained on this text better understand uncertainty and conditionality.

### 6.5 Stock Market Social Media (Unlabeled)

| Dataset | Path | Size |
|---------|------|------|
| StephanAkkerman/stock-market-tweets-data | HuggingFace | 923,673 tweets (Apr-Jul 2020) |
| Reddit WallStreetBets dumps | Various | Millions of posts |
| StockTwits historical data | StockTwits API | Billions of messages |

### 6.6 Additional Financial Text Sources

- **IPO prospectuses**: Available via SEC EDGAR (S-1 filings). Dense financial language.
- **Credit rating agency reports**: Partially available from S&P, Moody's, Fitch press releases.
- **ESG reports**: Growing corpus, available from company investor relations pages.
- **Financial textbooks**: Public domain older editions available on Project Gutenberg / Internet Archive.
- **Loughran-McDonald financial word lists**: Not a corpus but provides domain vocabulary for augmentation.
- **Financial regulations**: SEC, FINRA, CFTC regulatory texts. Free and public.

---

## 7. Market-Reaction Labeled Datasets (Distant Supervision) <a id="7-market-reaction"></a>

These datasets use stock price movements to automatically assign sentiment labels. Labels are noisy but free and massive in scale.

| Dataset | Size | Label Method | Best For |
|---------|------|-------------|----------|
| JanosAudran/financial-reports-sec | 20.5M sentences | 1/5/30-day return windows | DAPT with weak labels |
| FNSPID | 15.7M news + prices | Price-aligned news | News-price sentiment |
| FinMarBa | 61K Bloomberg headlines | Market-movement labels | High-quality distant labels |

**Best practice**: Use market-reaction labels as auxiliary training signal or for pretraining, not as primary fine-tuning labels. A 30-day return window produces more reliable labels than 1-day for longer documents (10-K filings). For news headlines, 1-day returns may be appropriate.

**Loughran-McDonald lexicon as distant supervisor**: The LM dictionary provides six financial sentiment categories (negative, positive, litigious, uncertainty, constraining, superfluous) derived from 10-K filings. Can be used to auto-label sentences containing LM dictionary words. Recent work (EnhancedFinSentiBERT, 2025) integrates LM signals as a feature branch alongside BERT, achieving 87.0% F1 on FPB.

---

## 8. What the Best Financial Language Models Used <a id="8-what-top-models-used"></a>

Understanding what data the top models trained on reveals what works:

| Model | Base | Pretraining Corpus | Corpus Size | FPB Result |
|-------|------|--------------------|-------------|------------|
| ProsusAI/FinBERT | BERT-base | Reuters TRC2 (financial subset) | Undisclosed | 88.9% acc |
| FinBERT (Yang et al.) | BERT-base | 10-K + financial news + earnings calls | 4.9B tokens | 79.3% acc* |
| BloombergGPT | From scratch (50B) | FinPile: 40% web/news, 40% filings, 20% other | 363B financial + 345B general | N/A |
| FinTral | Mistral-7B | FinSet: C4-finance + SEC EDGAR + news + social media | 20B tokens | Beats GPT-4 on 5/9 tasks |
| EnhancedFinSentiBERT | BERT | Financial news + analyst reports (2010-2024) | ~1GB text, 9M+ sentences | 87.0% F1 |
| FinRoBERTa-FSA | RoBERTa | Financial domain pretraining + FPB fine-tuning | Undisclosed | ~97% acc |

*Yang FinBERT's 79.3% is on analyst report test set, not FPB.

**Key insight**: Every model that achieves >88% on FPB has domain-adaptive pretraining on financial text. The models that skip DAPT and go straight to fine-tuning plateau around 80-85%. This is the single most important gap in current ModernFinBERT training.

### Reproducible Open-Source Ceiling

The maximum openly available pretraining corpus:
- **SEC filings**: PleIAs/SEC (7.25B words) + EDGAR-CORPUS (220K filings, ~6B tokens)
- **Financial news**: Financial-News-Multisource (57M rows)
- **Earnings calls**: kurry/sp500_earnings_transcripts (33K+)
- **Central bank**: BIS + ECB + Fed speeches and minutes
- **Social media**: Stock tweets (923K) + StockTwits + Reddit finance subs

**Estimated total**: 15-20B tokens of financial text, comparable to what ProsusAI and Yang used, and ~5% of BloombergGPT's financial corpus. More than sufficient for continued pretraining of an encoder model.

---

## 9. State-of-the-Art Training Methods <a id="9-sota-methods"></a>

### 9.1 Domain-Adaptive Pretraining (DAPT)

**Foundation**: Gururangan et al. (ACL 2020) "Don't Stop Pretraining" showed that continued MLM pretraining on domain-specific text consistently improves downstream performance. Combining DAPT then TAPT gave the best results.

**For finance specifically**:
- Every top-performing financial model uses DAPT
- Continued pretraining from a strong general checkpoint (BERT/RoBERTa/ModernBERT) is far more cost-effective than pretraining from scratch
- Mix ~70-80% financial text with ~20-30% general text to prevent catastrophic forgetting (GeoGalactica used 8:1:1 ratio)
- Even 10% of a well-selected corpus matches full-corpus performance (AWS FinPythia, EMNLP 2024)

### 9.2 Task-Adaptive Pretraining (TAPT)

Continue MLM pretraining on unlabeled examples from the target task distribution (e.g., financial headlines if the task is headline sentiment). Even a few thousand unlabeled examples help. Run 50-100 epochs of MLM on them before fine-tuning.

**Pipeline**: General pretrain -> DAPT on financial corpus -> TAPT on task-specific data -> supervised fine-tuning

### 9.3 Data Mixing and Selection

- **DoReMi** (NeurIPS 2023): Use a proxy model to find optimal domain weights, then resample training data. +6.5% on average over default weights.
- **Perplexity-based filtering**: Select highest-quality financial samples rather than using everything. AWS FinPythia achieved full performance with 10% of corpus using perplexity + token-type entropy filtering.
- **Anti-catastrophic-forgetting**: Include 5-30% general-domain data during continued pretraining.

### 9.4 LLM-Based Labeling at Scale

- **GPT-4o/Claude as teacher**: Use LLMs to label financial sentiment at scale, then distill into small encoder.
- **Active learning** (Fed Reserve M-RARU, 2025): 80% reduction in labeling samples needed by selecting only the most informative points for LLM labeling.
- **SIEVE** (2024): Fine-tune lightweight encoder on LLM annotations using active learning -- 500 filtering operations for the cost of 1 GPT-4o call.
- **Multi-LLM consensus**: NOSIBLE dataset used 8 LLMs for labeling. Consensus reduces individual model bias.
- **Cost-effective pipeline**: (1) Loughran-McDonald dictionary for initial distant labels on large corpus, (2) GPT-4o/Claude with active learning on ~2-5K high-value examples, (3) distill into target encoder.

### 9.5 Contrastive Learning

- **SuCroMoCo** (Knowledge-Based Systems, 2024): Supervised Cross-Momentum Contrast aligns financial text with prototypical sentiment representations. Outperforms both PLM-based approaches and LLMs on financial benchmarks.
- **Practical approach**: Add SupCon (supervised contrastive) loss alongside cross-entropy during fine-tuning to learn tighter sentiment clusters.

### 9.6 Multi-Task Learning

- Train sentiment + NER + relation extraction jointly (IJCAI 2020 FinBERT used 6 simultaneous pretraining tasks)
- Recent work (2024) shows joint training reduces reliance on external knowledge and prevents error propagation
- Candidate auxiliary tasks: financial NER (FiNER-139), fact verification (Fin-Fact), financial QA (FinQA)

### 9.7 Handling Financial Sentiment Challenges

| Challenge | What It Is | Mitigation |
|-----------|-----------|------------|
| Neutral dominance | ~60% of FPB is neutral | Weighted loss or dedicated neutral feature branch |
| Negation/hedging | "Not unprofitable" = positive | Negation-aware attention or augmentation with negated examples |
| Forward-looking | "Revenue expected to decline" | Include forward-looking statements in training data (earnings calls are rich in these) |
| Entity-level conflict | Same sentence, different sentiment per entity | Train on SEntFiN / FinEntity for entity-aware sentiment |
| Domain vocabulary | "Haircut" ≠ barbershop | DAPT on financial text handles this naturally |
| Class imbalance | Varies by source | Stratified sampling + focal loss |

### 9.8 Base Model Choice

| Model | Params | Context | Speed | Best For |
|-------|--------|---------|-------|----------|
| BERT-base | 110M | 512 | Baseline | Legacy baseline |
| RoBERTa-base | 125M | 512 | ~1x BERT | Better pretraining recipe |
| DeBERTa-v3-base | 184M | 512 | ~0.5x ModernBERT | **Best task accuracy**, best sample efficiency |
| ModernBERT-base | 149M | 8,192 | 2-4x DeBERTa | **Best speed/memory**, long context |

**Critical finding** (April 2025 paper): When trained on identical data, DeBERTa-v3 outperforms ModernBERT on benchmark accuracy and reaches higher performance significantly earlier in training. ModernBERT's advantages may partly stem from superior training data (2T tokens) rather than architecture alone.

**Recommendation**: Train both ModernBERT-base and DeBERTa-v3-base variants. ModernBERT for production (speed + long context for earnings calls), DeBERTa-v3 for maximum accuracy on short text.

### 9.9 Key Recent Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| "Reasoning or Overthinking" | 2025 | GPT-4o *without* CoT beats GPT-4o *with* CoT on financial sentiment -- overthinking hurts |
| FinSentLLM | 2025 | Multi-LLM ensemble + semantic features, +3-6% over FinBERT without fine-tuning |
| EnhancedFinSentiBERT | 2025 | Dictionary knowledge + neutral feature extraction = 87.0% F1 on FPB |
| TinyFinBERT | 2024 | GPT-4o augmentation + knowledge distillation to tiny model |
| Fed Reserve M-RARU | 2025 | Active knowledge distillation, 80% sample reduction |
| SuCroMoCo | 2024 | Supervised contrastive learning beats both PLMs and LLMs |
| AWS FinPythia | 2024 | 10% of corpus = full performance with smart selection |
| ModernBERT vs DeBERTaV3 | 2025 | DeBERTaV3 wins on accuracy with same data; ModernBERT wins on efficiency |

---

## 10. Recommended Training Pipeline <a id="10-recommended-pipeline"></a>

Based on all findings, here is the recommended pipeline for training the best financial sentiment model:

### Phase 1: Domain-Adaptive Pretraining (DAPT)

**Objective**: Continued MLM pretraining to inject financial domain knowledge into the base model.

**Corpus** (combine these, total ~10-15B tokens):
1. `eloukas/edgar-corpus` -- 220K 10-K filings, Apache 2.0 (primary)
2. `PleIAs/SEC` -- 7.25B words of 10-K filings, 1993-2024 (supplement for recent years)
3. `Brianferrell787/financial-news-multisource` -- 57M rows of financial news
4. `kurry/sp500_earnings_transcripts` -- 33K earnings call transcripts
5. Central bank communications (BIS, ECB, Fed speeches/minutes)
6. `StephanAkkerman/stock-market-tweets-data` -- 923K financial tweets (unlabeled)

**Mix ratio**: 75% financial + 25% general (use a Wikipedia/BookCorpus sample to prevent forgetting)

**Data selection**: Use perplexity-based filtering to select highest-quality financial text. Target 2-5B tokens for continued pretraining (sufficient for encoder models).

**Duration**: ~50K-100K steps, batch size 256, learning rate ~1e-4 with linear warmup

### Phase 2: Task-Adaptive Pretraining (TAPT)

**Objective**: Continue MLM on unlabeled text similar to the downstream task.

**Corpus**: Collect all text from your labeled datasets (strip labels), plus:
- Financial headlines from news corpora (matching FPB-style text)
- Short financial commentary (matching tweet-style text)
- Earnings call excerpts (matching EC-style text)

**Duration**: 50-100 epochs on the task-specific unlabeled data

### Phase 3: Supervised Fine-Tuning

**Training data** (combine, ~250K+ total after dedup):
1. **NOSIBLE/financial-sentiment** -- 100K (anchor dataset, highest volume)
2. **flwrlabs/fingpt-sentiment-train** -- 76K (instruction format -> strip to text+label)
3. **TimKoornstra/financial-tweets-sentiment** -- 38K (social media domain)
4. **FinanceMTEB/FinSent** -- 10K (analyst report domain)
5. **SEntFiN** -- 10.7K (entity-aware, resolve to sentence-level for standard training)
6. **StockEmotions** -- 10K (collapse 12 emotions to 3-class)
7. **Gold Commodity Headlines** -- 11.4K (commodity domain)
8. **TheFinAI/finben-fomc** -- 496 (monetary policy domain)
9. **yixuantt/FinEntity** -- 979 (entity-level)
10. **Your existing aggregated dataset** -- 14K (current ModernFinBERT training data)

**Deduplication**: Critical. Many datasets share source data (FPB appears in FinGPT, NOSIBLE may overlap with FPB). Use embedding-based dedup with cosine similarity > 0.95 threshold.

**Label harmonization**: Map all to 3-class {negative, neutral, positive}. For bullish/bearish, map bearish->negative, bullish->positive.

**Loss function**: Cross-entropy + SupCon (supervised contrastive) loss, weighted by inverse class frequency

**Augmentation**: Use Verbalized Sampling (your existing DataBoost method) to augment misclassified validation examples

### Phase 4: Evaluation

**Held-out benchmarks** (not used in training):
- Financial PhraseBank 50agree (standard)
- Financial PhraseBank allagree (high-agreement subset)
- FiQA-SA (aspect-based)
- Twitter Financial News Sentiment
- FOMC Hawkish-Dovish
- Your existing blind test set (723 samples)

**Protocols**:
- 10-fold CV on FPB (for comparability with literature)
- Held-out evaluation (your established protocol)
- Multi-seed for statistical significance

### Phase 5: LLM Labeling for Additional Data (Optional)

If more training data is needed:
1. Take unlabeled financial news headlines from `financial-news-multisource`
2. Use active learning (M-RARU approach) to select most uncertain/informative examples
3. Label ~5-10K with Claude/GPT-4o using multi-LLM consensus
4. Add to training set and retrain

### Expected Outcome

Based on literature, this pipeline should achieve:
- **FPB 50agree**: 89-92% accuracy (up from 86.88% current CV, 80.44% held-out)
- **FPB allagree**: 96-98% accuracy (up from 95.14%)
- The DAPT phase alone should add +3-5pp over current results
- The expanded training data should add another +2-3pp

---

## 11. Complete Dataset Reference Table <a id="11-reference-table"></a>

### Labeled Datasets

| # | Dataset | Path | Size | Labels | Domain | License | Priority |
|---|---------|------|------|--------|--------|---------|----------|
| 1 | NOSIBLE Financial Sentiment | `NOSIBLE/financial-sentiment` | 100K | 3-class | News | ODC-By | **P0** |
| 2 | FinGPT Sentiment Train | `flwrlabs/fingpt-sentiment-train` | 76.7K | 3+7 class | Mixed | MIT | **P0** |
| 3 | TimKoornstra Financial Tweets | `TimKoornstra/financial-tweets-sentiment` | 38K | 3-class | Social | MIT | **P0** |
| 4 | JanosAudran SEC Reports | `JanosAudran/financial-reports-sec` | 20.5M | Binary (market) | SEC 10-K | Apache 2.0 | **P1** |
| 5 | FinanceMTEB FinSent | `FinanceMTEB/FinSent` | 10K | 3-class | Analyst | N/S | **P1** |
| 6 | SEntFiN 1.0 | Paper / GitHub | 10.7K | Entity-level 3-class | News | N/S | **P1** |
| 7 | StockEmotions | Paper (AAAI 2023) | 10K | 12 emotions + 2 sent | StockTwits | N/S | **P1** |
| 8 | Gold Commodity Headlines | Paper | 11.4K | Sentiment | Commodity | N/S | **P1** |
| 9 | FinGPT Sentiment Cls | `FinGPT/fingpt-sentiment-cls` | 47.5K | Binary | Mixed | N/S | **P2** |
| 10 | Kaggle Finance News | `antobenedetti/finance-news-sentiments` | 32K | Sentiment | News | N/S | **P2** |
| 11 | Synthetic Financial Tweets | `TimKoornstra/synthetic-financial-tweets-sentiment` | 1.4M | 3-class | Synthetic | MIT | **P2** |
| 12 | FOMC Hawkish-Dovish | `TheFinAI/finben-fomc` | 496 | 3-class | Monetary | CC-BY-NC 4.0 | **P2** |
| 13 | FinEntity | `yixuantt/FinEntity` | 979 | Entity-level 3-class | Finance | ODC-BY | **P2** |
| 14 | FinMarBa | Paper (2025) | 61K | Market-reaction | Bloomberg | N/S | **P2** |
| 15 | SemEval-2017 Task 5 | Academic | ~4K | Continuous scores | Mixed | Academic | **P2** |
| 16 | Reddit WSB Sentiment | `SocialGrep/reddit-wallstreetbets-aug-2021` | ~100K+ | Pipeline labels | Reddit | N/S | **P3** |
| 17 | Kaggle Labeled Headlines | `cashbowman/sentiment-labeled-headlines` | N/S | 1-5 scale | News | N/S | **P3** |
| 18 | FiQA (already known) | `TheFinAI/fiqa-sentiment-classification` | 1.1K | Continuous | Mixed | MIT | Benchmark |
| 19 | FPB (already known) | `takala/financial_phrasebank` | 4.8K | 3-class | News | CC-BY-NC-SA | Benchmark |
| 20 | TFNS (already known) | `zeroshot/twitter-financial-news-sentiment` | 11.9K | 3-class | Twitter | MIT | Benchmark |

### Unlabeled Corpora for DAPT

| # | Dataset | Path | Size | Domain | License |
|---|---------|------|------|--------|---------|
| 1 | PleIAs/SEC | `PleIAs/SEC` | 7.25B words | 10-K filings | N/S |
| 2 | EDGAR-CORPUS | `eloukas/edgar-corpus` | 220K filings | 10-K filings | Apache 2.0 |
| 3 | Financial-News-Multisource | `Brianferrell787/financial-news-multisource` | 57M rows | News (24 sources) | Research |
| 4 | FNSPID | `Zihan1004/FNSPID` | 15.7M news | News + prices | CC-BY-NC 4.0 |
| 5 | SP500 Earnings Transcripts | `kurry/sp500_earnings_transcripts` | 33K transcripts | Earnings calls | N/S |
| 6 | Stock Market Tweets | `StephanAkkerman/stock-market-tweets-data` | 923K tweets | Social | CC-BY 4.0 |
| 7 | BIS Speeches | bis.org | Thousands | Central bank | Public |
| 8 | ECB Speeches | ecb.europa.eu | Thousands | Central bank | Public |
| 9 | Fed FOMC Minutes | federalreserve.gov | 1993-present | Monetary policy | Public |
| 10 | SEC EDGAR Raw | efts.sec.gov | All filings | All SEC | Public |

---

## Appendix: Key References

1. Gururangan et al. "Don't Stop Pretraining" (ACL 2020) -- DAPT/TAPT methodology
2. Malo et al. "Good debt or bad debt" (2014) -- Financial PhraseBank
3. Shah et al. "Trillion Dollar Words" (ACL 2023) -- FOMC sentiment
4. Wu et al. "BloombergGPT" (2023) -- Financial LLM at scale
5. Bhatia et al. "FinTral" (ACL Findings 2024) -- Multimodal financial LLM
6. Thomas "TinyFinBERT" (2024) -- GPT-4o augmentation + distillation
7. Fed Reserve "LLM on a Budget" (2025) -- Active knowledge distillation
8. EnhancedFinSentiBERT (ScienceDirect 2025) -- Dictionary + neutral features
9. SuCroMoCo (Knowledge-Based Systems 2024) -- Supervised contrastive learning
10. AWS FinPythia (EMNLP 2024) -- Efficient data selection for financial DAPT
11. ModernBERT vs DeBERTaV3 (arXiv 2025) -- Architecture comparison
12. Sinha et al. "SEntFiN 1.0" (2022) -- Entity-aware financial sentiment
13. NOSIBLE Financial Sentiment (2025) -- Multi-LLM consensus labeling at scale
