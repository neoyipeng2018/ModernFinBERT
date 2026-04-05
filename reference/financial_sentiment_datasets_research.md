# Financial Sentiment Analysis: Comprehensive Dataset & Data Source Research

**Date**: 2026-04-01
**Purpose**: Identify all available datasets and data sources for training the best possible financial sentiment model, covering labeled sentiment datasets, unlabeled pretraining corpora, benchmarks, and novel data sources.

---

## 1. LABELED FINANCIAL SENTIMENT DATASETS

### 1.1 Tier 1: High-Quality, Widely-Used Benchmarks

These are the gold-standard datasets used by virtually every financial sentiment paper. You likely already use some of these.

#### Financial PhraseBank (FPB)
- **URL**: https://huggingface.co/datasets/takala/financial_phrasebank
- **Size**: 4,840 sentences
- **Labels**: Positive / Negative / Neutral (3-class)
- **Domain**: English financial news (corporate press releases, earnings reports)
- **Annotation**: 5-8 annotators per sentence; subsets by agreement level (50%, 66%, 75%, 100%)
- **License**: CC BY-NC-SA 3.0
- **Quality**: Very high -- human-annotated with agreement filtering
- **Paper**: Malo et al., 2014
- **Notes**: The single most common benchmark. Small but clean. The 100% agreement subset (~2,264 sentences) is highest quality.

#### FiQA Sentiment Analysis (FiQA-SA)
- **URL**: https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification
- **Size**: 1,173 examples (train: 822, valid: 117, test: 234)
- **Labels**: Continuous sentiment score from -1 (bearish) to +1 (bullish)
- **Domain**: Financial microblogs (StockTwits) + news headlines
- **License**: Public (research use)
- **Quality**: High -- from FiQA 2018 shared task
- **Paper**: Maia et al., 2018
- **Notes**: Unique continuous-scale labels rather than discrete classes. Part of FLUE benchmark.

#### SemEval-2017 Task 5 (TSA)
- **URL**: https://bitbucket.org/ssix-project/semeval-2017-task-5-subtask-1/
- **Size**: ~561 examples (test); Track 1 microblogs + Track 2 news headlines
- **Labels**: Continuous sentiment -1 to +1 per entity/company
- **Domain**: StockTwits microblogs (Track 1) + Yahoo Finance news (Track 2)
- **License**: Research use
- **Quality**: High -- 32 participating teams; well-validated
- **Paper**: Cortis et al., 2017
- **Notes**: Entity-level fine-grained sentiment. Important for aspect-based approaches.

#### Twitter Financial News Sentiment (TFNS)
- **URL**: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
- **Size**: 11,932 tweets (Bearish: 1,789, Bullish: 2,398, Neutral: 7,744)
- **Labels**: Bearish / Bullish / Neutral (3-class)
- **Domain**: Finance-related tweets
- **License**: MIT
- **Quality**: Good -- collected via Twitter API, human-annotated
- **Notes**: Good size for social media domain. Significant class imbalance toward Neutral.

### 1.2 Tier 2: Large-Scale or Novel Labeled Datasets (High Priority for Training)

These are less commonly used but potentially more valuable for training a state-of-the-art model due to their size, domain specificity, or novel annotation approaches.

#### NOSIBLE Financial Sentiment (100K)
- **URL**: https://huggingface.co/datasets/NOSIBLE/financial-sentiment
- **Size**: 100,000 samples
- **Labels**: Positive / Negative / Neutral (3-class, financial-impact focused)
- **Domain**: Financial news from public web (diverse sources)
- **Annotation**: Multi-LLM consensus (8 models including Grok 4, Gemini 2.5 Flash, GPT-5 Nano, Llama 4 Maverick) + active learning relabeling with GPT-5.1 oracle
- **License**: ODC-By v1.0 (commercial use permitted)
- **Quality**: High -- multi-model consensus + active learning refinement; outperforms FPB-only training
- **Notes**: BEST CANDIDATE FOR LARGE-SCALE TRAINING. 20x larger than FPB. LLM-labeled but with sophisticated quality control. Financial-impact focused labels (not general sentiment).

#### FinMarBa (Market-Informed)
- **URL**: https://arxiv.org/abs/2507.22932 / https://ssrn.com/abstract=5365059
- **Size**: 2M+ individual news items from 3,700+ daily Bloomberg Market Wraps (2010-2024)
- **Labels**: Market-driven annotation (sentiment based on actual market responses, not human interpretation)
- **Domain**: Bloomberg Market Wraps -- professional financial journalism
- **License**: Academic (check paper for availability)
- **Quality**: Very high -- eliminates human annotation bias by using market reactions
- **Notes**: NOVEL APPROACH. Labels derived from market behavior rather than human annotators. Addresses the fundamental problem that human-labeled sentiment often doesn't predict market movements.

#### SEntFiN 1.0 (Entity-Aware)
- **URL**: https://arxiv.org/abs/2305.12257
- **Size**: 10,753 news headlines with entity-sentiment annotations; 2,847 with multiple entities
- **Labels**: Entity-level sentiment (positive/negative/neutral per entity)
- **Domain**: Financial news headlines
- **Supplementary**: Database of 1,000+ financial entities with 5,000+ name variations
- **License**: Research use
- **Quality**: Very high -- addresses multi-entity conflicting sentiment problem
- **Paper**: Sinha & Kedas, 2022 (JASIST)
- **Notes**: Critical for real-world use where a headline mentions multiple companies with different sentiment implications.

#### FinEntity (Entity-Level)
- **URL**: https://huggingface.co/datasets/yixuantt/FinEntity
- **Size**: Not publicly specified (check paper)
- **Labels**: Entity span + sentiment (Positive / Neutral / Negative) per entity
- **Domain**: Financial news
- **License**: Research use
- **Quality**: High -- EMNLP 2023 paper
- **Paper**: EMNLP 2023
- **Notes**: Complementary to SEntFiN. Annotates entity spans AND their sentiment.

#### StockEmotions (Fine-Grained Emotions)
- **URL**: https://github.com/adlnlp/StockEmotions
- **Size**: 10,000 labeled comments (+ 50,281 processed samples)
- **Labels**: 12 emotion classes (ambiguous, amusement, anger, anxiety, belief, confusion, depression, disgust, excitement, optimism, panic, surprise) + 2 sentiment classes (bullish 55% / bearish 45%)
- **Domain**: StockTwits comments
- **License**: Research use
- **Quality**: High -- AAAI 2023 paper; balanced bullish/bearish split
- **Paper**: AAAI 2023 Bridge
- **Notes**: Only dataset with fine-grained investor EMOTIONS beyond simple sentiment. Includes time series data.

#### Gold Commodity Headlines (Sinha & Khandait)
- **URL**: https://huggingface.co/datasets/SaguaroCapital/sentiment-analysis-in-commodity-market-gold
- **Size**: 11,412 annotated headlines (2000-2019)
- **Labels**: Multiple subtasks (price up/down/same, comparisons, etc.)
- **Domain**: Gold commodity market news
- **License**: Research use
- **Quality**: Good -- human-annotated, large for headline datasets
- **Paper**: Sinha & Khandait, 2020

### 1.3 Monetary Policy / Central Bank Sentiment

#### FOMC Hawkish-Dovish (Trillion Dollar Words)
- **URL**: https://huggingface.co/datasets/gtfintechlab/fomc_communication | https://github.com/gtfintechlab/fomc-hawkish-dovish
- **Size**: Largest tokenized FOMC dataset (1996-2022); test set of 1,375+ labeled entries
- **Labels**: Hawkish / Dovish / Neutral (3-class)
- **Domain**: FOMC speeches, meeting minutes, press conference transcripts
- **License**: CC BY-NC 4.0
- **Quality**: Very high -- ACL 2023 paper
- **Paper**: Shah, Paturi & Chava, ACL 2023
- **Notes**: Essential for monetary policy sentiment. Includes fine-tuned FOMC-RoBERTa model.

#### Op-Fed (FOMC Opinion & Stance)
- **URL**: https://arxiv.org/abs/2509.13539
- **Size**: 1,044 annotated sentences with context
- **Labels**: Five-stage hierarchical schema (opinion, monetary policy reference, stance directionality)
- **Domain**: Verbatim FOMC meeting transcripts
- **License**: Research use
- **Quality**: Very high -- active learning annotation; addresses inter-sentence dependence
- **Notes**: Fewer than 8% of sentences express non-neutral stance. 65% require context beyond sentence-level.

#### FedNLP
- **URL**: https://github.com/usydnlp/FedNLP
- **Size**: 122 FOMC documents + 1,300 speeches (Jan 2015 - Jul 2020)
- **Labels**: Multiple NLP tasks (sentiment, summarization, rate prediction)
- **Domain**: FOMC statements, minutes, press conferences, member speeches
- **License**: Research use
- **Paper**: SIGIR 2021

### 1.4 ESG and Specialized Finance

#### ML-ESG Shared Task Datasets
- **URL**: https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp-kdf-2024/shared-task-ml-esg-3
- **Size**: Varies by year (ML-ESG-1, 2, 3)
- **Labels**: ESG impact level + duration (English and French)
- **Domain**: ESG news articles
- **License**: Shared task participants
- **Notes**: From FinNLP workshop series. Includes multilingual variants.

#### MultiFin (Multilingual)
- **URL**: https://aclanthology.org/2023.findings-eacl.66/
- **Size**: 546 English examples (+ 14 other languages)
- **Labels**: Multi-class financial headline classification
- **Domain**: Real-world article headlines, 15 languages
- **License**: Research use
- **Paper**: EACL 2023
- **Notes**: Useful if you want multilingual capability.

### 1.5 Crypto-Specific Sentiment

#### CryptoBERT Datasets
- **URL**: https://huggingface.co/ElKulako/cryptobert
- **Pretraining corpus**: 3.2M unique crypto social media posts (1.865M StockTwits, 496K tweets, 172K Reddit, 664K Telegram)
- **Fine-tuning data**: 2M labeled StockTwits posts (Bearish/Neutral/Bullish)
- **Labels**: Bearish / Neutral / Bullish (3-class)
- **Domain**: Cryptocurrency social media
- **Quality**: Good -- 70% accuracy on 200K out-of-sample test
- **Notes**: Both the pretraining corpus and labeled data are valuable. StockTwits labels are user-provided (noisy but massive scale).

### 1.6 Kaggle Financial Sentiment Datasets

#### Financial Sentiment Analysis (sbhatti)
- **URL**: https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis
- **Size**: 5,842 rows
- **Labels**: Positive / Negative / Neutral
- **Domain**: Financial news
- **License**: Kaggle (check terms)

#### Sentiment Analysis for Financial News (ankurzing)
- **URL**: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- **Size**: ~4,840 rows (likely a copy of Financial PhraseBank)
- **Labels**: Positive / Negative / Neutral

#### Aspect-Based Sentiment Analysis for Financial News
- **URL**: https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news
- **Labels**: Aspect-level sentiment

#### Stock-Market Sentiment Dataset
- **URL**: https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset
- **Labels**: Positive / Negative

#### Stock Tweets for Sentiment Analysis
- **URL**: https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction
- **Domain**: Stock-related tweets

#### Daily Financial News for 6000+ Stocks (Benzinga)
- **URL**: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
- **Size**: Massive (6000+ stocks, 2000s-2010s)
- **Domain**: Benzinga headlines + analyst ratings
- **Notes**: Unlabeled for sentiment but includes analyst ratings that could serve as weak labels.

#### Stock News Sentiment Analysis (Massive)
- **URL**: https://www.kaggle.com/datasets/avisheksood/stock-news-sentiment-analysismassive-dataset
- **Size**: Large
- **Domain**: Stock news headlines

---

## 2. UNLABELED FINANCIAL TEXT CORPORA (For Domain-Adaptive Pretraining)

### 2.1 SEC EDGAR Filings

#### EDGAR-CORPUS (Loukas et al.)
- **URL**: https://huggingface.co/datasets/eloukas/edgar-corpus
- **Size**: 6+ billion tokens from 90,000+ annual reports (10-K filings, 1993-2020)
- **Domain**: SEC 10-K annual reports
- **License**: Public domain (SEC filings are public)
- **Paper**: Loukas et al., ECONLP 2021
- **Notes**: BEST OPEN-SOURCE SEC CORPUS. Used for domain-adaptive pretraining by multiple models including FiLM.

#### PleIAs/SEC
- **URL**: https://huggingface.co/datasets/PleIAs/SEC
- **Size**: 7.25 billion words across 245,211 entries (avg 34,324 words/entry)
- **Domain**: SEC 10-K annual reports (1993-2024)
- **License**: Public domain
- **Notes**: LARGEST and most current SEC filing corpus available. Built with EDGAR-Crawler toolkit. More comprehensive than EDGAR-CORPUS.

#### JanosAudran/financial-reports-sec
- **URL**: https://huggingface.co/datasets/JanosAudran/financial-reports-sec
- **Size**: 13.4 GB
- **Domain**: SEC 10-K annual reports (up to 2020)
- **Labels**: Includes positive/negative labels based on market reaction around filing date
- **License**: Public domain
- **Notes**: Uniquely includes sentiment labels derived from stock price reactions. Dual-use: pretraining + weak supervision.

#### EDGAR Crawler (Toolkit)
- **URL**: https://github.com/lefterisloukas/edgar-crawler
- **Purpose**: Download and extract text from specific item sections of SEC filings into structured JSON
- **Notes**: Build your own corpus with full control over filing types, date ranges, and sections.

### 2.2 Earnings Call Transcripts

#### Strux Transcripts Dataset
- **URL**: https://struxdata.github.io/
- **Size**: 11,950 quarterly transcripts (avg 10,187 tokens each) from 869 companies across 11 sectors
- **Domain**: NASDAQ 500 / S&P 500 earnings calls (2017-2024)
- **Labels**: Investment labels (Strongly Buy / Buy / Hold / Sell / Strongly Sell) based on 30-day post-earnings stock performance
- **Notes**: HIGHEST QUALITY earnings call dataset with built-in investment signal labels.

#### kurry/sp500_earnings_transcripts
- **URL**: https://huggingface.co/datasets/kurry/sp500_earnings_transcripts
- **Size**: 33,000+ unique transcripts across 685 companies (2+ decades)
- **Domain**: S&P 500 earnings calls
- **Notes**: Largest freely available earnings call transcript dataset on HuggingFace.

#### glopardo/sp500-earnings-transcripts
- **URL**: https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts
- **Domain**: S&P 500 earnings calls (2014-2024)
- **Notes**: Includes quarterly financial metrics and company fundamentals alongside transcripts.

#### FINOS EarningsCallTranscript
- **URL**: https://huggingface.co/datasets/finosfoundation/EarningsCallTranscript
- **Domain**: Earnings call audio + transcriptions (Voxtral model)
- **Notes**: Multimodal -- includes audio. Useful for audio-text research.

#### lamini/earnings-calls-qa
- **URL**: https://huggingface.co/datasets/lamini/earnings-calls-qa
- **Domain**: Earnings call Q&A pairs
- **Notes**: Structured Q&A format for instruction tuning.

### 2.3 Financial News Corpora

#### Financial-News-Multisource (57M+ rows)
- **URL**: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource
- **Size**: 57,100,000+ rows across 24 subsets (1990-2025)
- **Domain**: Unified financial news corpus from 24 sources (NYT, Reddit finance, broad US outlets)
- **License**: Varies by subset
- **Notes**: THE LARGEST unified financial news corpus. Normalized schema (date, text, extra_fields). Streamable Parquet. Includes Reddit finance subreddits.

#### FNSPID (Financial News + Stock Price Integration)
- **URL**: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
- **Size**: 15.7 million financial news articles + 29.7 million stock price records (1999-2023, 4,775 S&P 500 companies)
- **Domain**: Four stock market news websites
- **License**: Research use
- **Paper**: KDD 2024
- **Notes**: Integrates news with stock prices. Good for market-informed labeling.

#### ashraq/financial-news-articles
- **URL**: https://huggingface.co/datasets/ashraq/financial-news-articles
- **Size**: 306,242 articles (492 MB)
- **Domain**: Financial news articles

#### Reuters TRC2 (Thomson Reuters Text Research Collection)
- **Size**: 1.8M news stories (Jan 2008 - Feb 2009); 290,444 financial articles extracted
- **Domain**: Reuters financial news
- **License**: Research license (apply to NIST/Reuters)
- **Notes**: Used to pretrain ProsusAI/FinBERT. Not publicly downloadable -- must apply for access. High quality professional journalism.

#### philipperemy/financial-news-dataset
- **URL**: https://github.com/philipperemy/financial-news-dataset
- **Domain**: Reuters and Bloomberg financial news
- **Notes**: Scraping toolkit; may have legal restrictions.

### 2.4 Central Bank Communications

#### BIS Central Bank Speeches
- **URL**: https://www.bis.org/cbspeeches/download.htm
- **Size**: Speeches from ~1,000 central bank officials since 1996/1997
- **Domain**: Central bank governor speeches worldwide
- **License**: Free for research
- **Notes**: Precompiled full-text extracts available for download. Covers financial crises, policy regime changes, unconventional monetary policies.

#### ECB Speeches Dataset
- **URL**: https://www.ecb.europa.eu/press/key/html/downloads.en.html
- **Size**: All ECB speeches with metadata
- **Domain**: European Central Bank communications
- **License**: Free for research
- **Notes**: Official ECB release to stimulate NLP research.

#### CB-LMs Corpus (BIS Working Paper)
- **URL**: https://www.bis.org/publ/work1215.pdf
- **Size**: 37,037 research papers + 18,345 speeches
- **Domain**: Central banking corpus
- **Notes**: Used to train CB-LMs (Central Bank Language Models).

### 2.5 Financial Social Media

#### StockTwits (via API)
- **Source**: https://stocktwits.com/
- **Size**: Millions of posts (ongoing)
- **Labels**: User-provided bullish/bearish tags (noisy but free labels)
- **Domain**: Stock market commentary
- **Notes**: THE most valuable social media source for financial sentiment. Users self-label sentiment. Multiple research datasets derived from it.

#### Reddit Financial Subreddits
- Included in Financial-News-Multisource dataset
- Key subreddits: r/wallstreetbets, r/stocks, r/investing, r/cryptocurrency
- Various scraped datasets exist (see WallStreetBets research papers)

---

## 3. DOMAIN-SPECIFIC FINANCIAL NLP BENCHMARKS

### 3.1 FLUE (Financial Language Understanding Evaluation)
- **URL**: https://salt-nlp.github.io/FLANG/ | https://github.com/SALT-NLP/FLANG
- **Paper**: Shah et al., EMNLP 2022
- **Tasks**: 5 financial NLP tasks
  - Financial Sentiment Analysis (FPB)
  - Financial Sentiment Regression (FiQA)
  - News Headline Classification
  - Named Entity Recognition
  - Structure Boundary Detection / Question Answering
- **Notes**: THE standard benchmark suite for financial NLP models. Used by FLANG.

### 3.2 FinBen (formerly PIXIU)
- **URL**: https://github.com/The-FinAI/PIXIU | https://www.thefin.ai/dataset-benchmark/finben
- **Paper**: NeurIPS 2024 Datasets & Benchmarks Track
- **Tasks**: 42 datasets across 24 tasks in 8 categories
- **Sentiment-specific datasets**:
  - FPB (4,845 examples, news)
  - FiQA-SA (1,173 examples, microblogs + headlines)
  - TSA / SemEval-2017 (561 examples, headlines)
  - FOMC (496 examples, hawkish/dovish)
- **Other relevant datasets**: Headlines (11,412), FinArg-ECC (969), MultiFin (546), M&A (500), MLESG (300)
- **Notes**: THE most comprehensive financial LLM benchmark. Open Financial LLM Leaderboard.

### 3.3 FLaME (Finance Language Model Evaluation)
- **URL**: https://arxiv.org/abs/2506.15846
- **Paper**: ACL 2025 Findings
- **Tasks**: 20 core NLP tasks in finance, evaluated across 23 foundation LMs
- **Notes**: Holistic taxonomy for financial NLP. Clear inclusion criteria including domain relevance, licensing, and label quality.

### 3.4 SemEval Financial Tasks
- **SemEval-2017 Task 5**: Fine-grained sentiment on financial microblogs and news (described above)
- 32 participating teams, well-established benchmark

### 3.5 FinNLP Workshop Shared Tasks
- **URL**: https://sigfintech.github.io/finnlp.html
- Annual events at ACL/IJCAI/LREC-COLING
- **Recent tasks**: FinLLM challenge, ML-ESG classification, financial text summarization
- **Notes**: Good source for new datasets each year.

### 3.6 Financial Narrative Processing (FNP/FNS)
- **URL**: http://wp.lancs.ac.uk/cfie/fns2023/
- **Task**: Summarization of UK annual reports
- **Size**: 3,863 annual reports with 2+ gold summaries each
- **Years**: Running since 2020; expanded to Spanish and Greek in 2023
- **Notes**: Not sentiment-specific but valuable for financial narrative understanding.

---

## 4. WHAT EXISTING FINANCIAL LANGUAGE MODELS USED FOR TRAINING

### 4.1 FinBERT (Huang et al., 2020 -- "FinBERT: A Large Language Model")
**Pretraining corpus (4.9B tokens total)**:
- 60,490 10-K filings + 142,622 10-Q filings (Russell 3000, 1994-2019) = 2.5B tokens
- 476,633 analyst reports (S&P 500, 2003-2012, Thomson Investext) = 1.1B tokens
- 136,578 earnings call transcripts (7,740 firms, 2004-2019, SeekingAlpha) = 1.3B tokens

**Fine-tuning**: Financial PhraseBank (Malo et al., 2014)

**Key insight**: This is the most diverse financial pretraining corpus documented in literature. The analyst reports are from Thomson Investext (proprietary). Earnings calls from SeekingAlpha have copyright restrictions.

### 4.2 FinBERT (Araci, 2019 / ProsusAI -- "FinBERT: Financial Sentiment Analysis")
**Pretraining corpus**: Reuters TRC2 subset -- 290,444 financial news articles extracted from 1.8M total articles
**Fine-tuning**: Financial PhraseBank
**Notes**: TRC2 requires research application to NIST. Simpler but effective approach.

### 4.3 BloombergGPT (Wu et al., 2023)
**FinPile (363B tokens)**:
- Web content: 298B tokens (financially relevant web documents)
- News articles: 38B tokens (hundreds of financial news sources)
- SEC filings: 14B tokens
- Press releases: 9B tokens
- Bloomberg content: 5B tokens (proprietary)

**General data**: 345B tokens from The Pile, C4, Wikipedia
**Total**: 708B tokens (51% financial, 49% general)
**Notes**: The FinPile composition is the gold standard for financial pretraining data mix. Most of it is NOT reproducible (proprietary Bloomberg data).

### 4.4 FLANG (Shah et al., EMNLP 2022)
**Pretraining**: Financial keywords/phrases masking, span boundary objective, in-filing objective
**Corpus**: SEC filings (10-K, 10-Q, 8-K) -- details in the paper
**Notes**: Novel masking strategy using financial keywords rather than random masking.

### 4.5 FinGPT (Yang et al., 2023)
**Data collection**: 34 diverse Internet sources via FinNLP pipeline
- Sources: CNBC, Reuters, Yahoo Finance, MarketWatch, etc.
- Social media, regulatory filings, academic datasets
**Training approach**: Instruction tuning (not full pretraining)
- LoRA fine-tuning on news + tweets sentiment data
- Labels: stock price changes as sentiment proxy
**Notes**: Open-source. Uses stock-price-derived labels instead of human annotations.

### 4.6 SEC-BERT (Loukas et al.)
**Pretraining**: EDGAR-CORPUS (6B+ tokens of 10-K filings)
**Models**: sec-bert-base, sec-bert-num, sec-bert-shape
**URL**: https://huggingface.co/nlpaueb/sec-bert-base

### 4.7 FiLM (Financial Language Model)
**Pretraining**: Expanded with 3.1B tokens from EDGAR-CORPUS
**URL**: https://huggingface.co/HYdsl/FiLM

---

## 5. FINANCIAL SENTIMENT LEXICONS

### Loughran-McDonald Master Dictionary
- **URL**: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
- **Size**: 354 positive, 2,355 negative, 297 uncertainty, 904 litigious, 19 strong modal, 27 weak modal, 184 constraining words
- **Domain**: SEC 10-K filings + earnings calls (CapIQ)
- **License**: Free for academic research
- **Notes**: THE standard financial sentiment lexicon. Shows that general-purpose dictionaries (Harvard) misclassify ~75% of "negative" words in financial context.

### NTUSD-Fin
- **URL**: https://nlpfin.github.io/
- **Size**: 8,331 words + 112 hashtags + 115 emojis
- **Domain**: Financial social media (built from 330K+ labeled posts)
- **Notes**: Specifically designed for social media financial text, not formal reports.

---

## 6. NOVEL AND UNDEREXPLORED DATA SOURCES

### 6.1 Market-Informed Labeling (Weak Supervision)
Instead of human annotation, use market reactions as labels:
- **FinMarBa approach**: Use Bloomberg Market Wrap sentiment vs. actual market moves
- **FinGPT approach**: Use relative stock price changes as sentiment labels
- **JanosAudran dataset**: 10-K filings labeled by stock price reaction around filing date
- **Strux approach**: 30-day post-earnings stock performance as investment signal
- **FNSPID**: 15.7M news articles paired with stock prices for 4,775 companies

### 6.2 Earnings Call Transcripts with Signals
- Strux (11,950 transcripts with buy/sell labels)
- glopardo dataset (transcripts + financial metrics)
- These contain management tone, forward-looking statements, Q&A dynamics

### 6.3 Central Bank Communications
- BIS speeches (1996-present, 1,000+ officials worldwide)
- ECB speeches (full corpus, free download)
- FOMC transcripts (multiple labeled datasets exist)
- Unique domain: monetary policy language has direct market impact

### 6.4 IPO Prospectuses
- Research exists on "forward looking statements" in IPO filings (FLS extraction)
- SEC EDGAR contains all prospectus filings
- Underexplored for sentiment analysis

### 6.5 ESG Reports
- ESG report collections exist (14,468 Chinese reports in one study)
- ML-ESG shared task datasets (FinNLP workshop)
- Thomson Reuters Eikon ESG news (10,000+ sources)
- Growing area with increasing investor focus

### 6.6 Credit Rating Reports
- Moody's, S&P, Fitch reports contain rich sentiment
- Generally proprietary -- no open datasets found
- Could be weak-labeled using rating changes

### 6.7 XBRL/Structured Filings
- FinLoRA includes 4 novel XBRL analysis datasets from 150 SEC filings
- Dow Jones 30 companies (2019-2023)
- Structured financial data with text

---

## 7. RECOMMENDED DATA STRATEGY FOR BEST FINANCIAL SENTIMENT MODEL

### Phase 1: Domain-Adaptive Pretraining (Continue pretraining on financial text)
**Priority corpora (all open-source, ordered by impact)**:
1. **PleIAs/SEC** -- 7.25B words of 10-K filings (1993-2024). Largest, most current.
2. **EDGAR-CORPUS** -- 6B+ tokens of 10-K filings. Well-established.
3. **kurry/sp500_earnings_transcripts** -- 33,000+ earnings calls. Captures spoken financial language.
4. **Financial-News-Multisource** -- 57M+ rows of financial news. Broadest news coverage.
5. **BIS + ECB speeches** -- Central bank language. Free download.

**Target**: ~10-15B tokens of mixed financial text (following BloombergGPT's FinPile composition as a guide: ~40% web/news, ~40% filings, ~20% other financial text).

### Phase 2: Supervised Fine-Tuning (Train sentiment classifier)
**Primary training data**:
1. **NOSIBLE Financial Sentiment** (100K) -- Largest high-quality labeled set
2. **Financial PhraseBank** (4,840 at 50% agreement; 2,264 at 100% agreement)
3. **TFNS** (11,932 tweets)
4. **SEntFiN** (10,753 entity-aware headlines)
5. **StockEmotions** (10,000 fine-grained emotions + sentiment)
6. **FiQA-SA** (1,173 continuous-scale)
7. **FOMC** (496 hawkish/dovish)
8. **Gold Headlines** (11,412 commodity-specific)
9. **CryptoBERT labels** (2M StockTwits -- noisy but massive; sample carefully)

**Total clean labeled data**: ~150K+ examples across diverse financial domains

**Weak supervision augmentation**:
- JanosAudran 10-K labels (market reaction-based)
- Strux earnings call labels (stock performance-based)
- StockTwits user labels (bullish/bearish tags)
- FinGPT-style price-derived labels on FNSPID news

### Phase 3: Evaluation
**Benchmarks to report**:
1. FPB (100% agreement subset) -- direct comparison with all prior work
2. FiQA-SA -- regression task
3. FLUE benchmark suite -- comprehensive
4. FinBen sentiment tasks -- modern standard
5. FOMC hawkish/dovish -- monetary policy transfer
6. SEntFiN entity-level -- real-world applicability

---

## 8. KEY SOURCES AND REFERENCES

### Benchmark Platforms
- Papers With Code Financial PhraseBank: https://paperswithcode.com/sota/sentiment-analysis-on-financial-phrasebank
- Open Financial LLM Leaderboard: https://www.thefin.ai/dataset-benchmark/finben
- FLUE/FLANG: https://salt-nlp.github.io/FLANG/

### Key Survey Papers
- Financial Sentiment Analysis: Techniques and Applications (ACM Computing Surveys 2024): https://dl.acm.org/doi/10.1145/3649451
- Large Language Models in Finance (FinLLMs): https://github.com/adlnlp/FinLLMs
- FinBen (NeurIPS 2024): https://arxiv.org/abs/2402.12659

### Key Model Papers
- FinBERT (Huang): https://arxiv.org/abs/2006.08097
- FinBERT (Araci/ProsusAI): https://arxiv.org/abs/1908.10063
- BloombergGPT: https://arxiv.org/abs/2303.17564
- FLANG: https://arxiv.org/abs/2211.00083
- FinGPT: https://arxiv.org/abs/2306.06031

### HuggingFace Collections
- TheFinAI datasets: https://huggingface.co/TheFinAI
- Tonic Financial Datasets Collection: https://huggingface.co/collections/Tonic/financial-datasets-65ac62ffe0ee7990a6ddc031

### Data Tools
- EDGAR Crawler: https://github.com/lefterisloukas/edgar-crawler
- Notre Dame SRAF (Loughran-McDonald): https://sraf.nd.edu/
- FinNLP Pipeline (FinGPT): https://github.com/AI4Finance-Foundation/FinGPT
