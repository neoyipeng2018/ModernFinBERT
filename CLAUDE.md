# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research project for developing ModernFinBERT, a modernized financial sentiment analysis model based on the ModernBERT architecture. The project aims to create a state-of-the-art financial NLP model with better performance than existing FinBERT models, targeting >94% accuracy on FinancialPhraseBank.

## Project Status

**Current State**: Active experimentation — three focused experiments comparing ModernBERT vs FinBERT baselines, measuring data augmentation impact, and benchmarking against Claude Opus 4.5.

**Key Results So Far**:
- ModernFinBERT-base uploaded to HuggingFace (`neoyipeng/ModernFinBERT-base`): 90.47% on FPB (trained on aggregated minus FPB)
- Cross-validation on FPB: 97.63% mean accuracy
- Working Colab setup with Unsloth + ModernBERT-base (149M params)

**Key Objective**: Build and deploy a financial sentiment analysis model that outperforms existing FinBERT variants, with focus on:
- Superior accuracy (>94% vs FinBERT's 93%)
- Fast inference (<50ms per sample)
- Community adoption and practical usage

## Current Experiments

### Notebook 1: Architecture Comparison (`notebooks/01_architecture_comparison.ipynb`)
Fine-tune ModernBERT-base on aggregated dataset (excluding FPB source 6), evaluate on FPB `sentences_allAgree` and `sentences_50agree`. Compare against ProsusAI/finbert and yiyanghkust/finbert-tone baselines. Reports accuracy, macro-F1, per-class metrics, confusion matrices.

### Notebook 2: DataBoost (`notebooks/02_databoost.ipynb`)
Train baseline, mine misclassified validation samples, paraphrase errors with Claude API (preserving correct labels), add to training set, retrain. Measures accuracy improvement from targeted augmentation.

### Notebook 3: Claude Comparison (`notebooks/03_claude_comparison.ipynb`)
Zero-shot classification of FPB test set using Claude Opus 4.5 via API. Compares accuracy/F1 against fine-tuned ModernFinBERT with cost analysis (API cost vs GPU hosting).

## Development Timeline

The project follows a structured 16-week timeline divided into four phases:

### Phase 1: Data Foundation (Weeks 1-3)
- Data audit, cleaning, and harmonization
- GPT-4 assisted quality control
- Dataset publication to HuggingFace

### Phase 2: Model Development (Weeks 4-8)
- ModernBERT architecture adaptation
- Training pipeline setup with wandb/tensorboard
- Base and Large model variants
- Specialized domain variants (earnings calls, news, social media)

### Phase 3: Paper & Tutorial Prep (Weeks 9-12)
- Academic paper writing
- Tutorial and demo development
- Conference submission preparation

### Phase 4: Community Building (Weeks 13-16)
- Public launch and social media campaign
- Integration guides and documentation
- Community engagement and feedback incorporation

## Expected Architecture

Based on the project plan, the codebase will likely develop into:

```
ModernFinBERT/
├── data/                    # Financial text datasets and cleaning scripts
├── models/                  # ModernBERT model variants and weights  
├── training/                # Training scripts and hyperparameter configs
├── evaluation/              # Benchmark suite and evaluation metrics
├── api/                     # Flask API for model serving
├── demo/                    # Gradio/Streamlit demo applications
├── notebooks/               # Jupyter notebooks for experimentation
├── docs/                    # Documentation and tutorials
└── requirements.txt         # Python dependencies
```

## Key Technologies

- **Base Architecture**: ModernBERT (adapted for financial domain)
- **ML Framework**: PyTorch/Transformers
- **Experiment Tracking**: Weights & Biases (wandb) or TensorBoard
- **Data Platform**: HuggingFace datasets and model hub
- **Demo/API**: Gradio, Streamlit, Flask
- **Optimization**: ONNX, TensorRT for inference acceleration

## Development Workflow

**Time Commitment**: 4 hours/week (1 hour × 4 days: Monday, Tuesday, Thursday, Friday)

**Session Structure**:
- Review previous progress at start
- Document clear next steps at end
- Maintain decision log for reference
- Weekly progress sharing for accountability

## Success Metrics

- FinancialPhraseBank accuracy: >94%
- Inference speed: <50ms per sample  
- Community adoption: 100+ GitHub stars in first month
- Conference tutorial acceptance
- Production usage by 3+ companies

## Risk Mitigation

- Backup compute via Colab Pro if training fails
- Focus on specialized variants if general performance lags  
- Workshop track submission if conference rejects main track
- Partnership strategy if adoption is slow

## Phase 1 Implementation Guide: Data Foundation

### Current Dataset Analysis

The `Data.ipynb` file shows you're working with the `neoyipeng/financial_reasoning_aggregated` dataset from HuggingFace:

**Dataset Overview:**
- **Size**: ~31,166 total samples (19,940 train + 4,992 dev + 6,234 test)
- **Labels**: 3-class sentiment (NEGATIVE:0, NEUTRAL/MIXED:1, POSITIVE:2)
- **Format**: Text samples with one-hot encoded labels
- **Filtering**: Already filtered for sentiment task only

**Key Issues Identified:**
1. Missing `NUM_CLASSES` variable (should be 3)
2. Missing numpy import
3. Dataset contains non-financial text (e.g., jewelry heist example)
4. Need to verify label distribution and quality

### Week 1: Data Audit & Strategy

#### Monday (1hr): Dataset Audit
```python
# Essential commands to run:
import numpy as np
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

# Load and inspect dataset
ds = load_dataset("neoyipeng/financial_reasoning_aggregated")
print(f"Dataset splits: {ds.keys()}")
print(f"Total samples: {sum(len(split) for split in ds.values())}")
print(f"Columns: {ds['train'].column_names}")

# Check label distribution
for split_name, split in ds.items():
    labels = [x['label'] for x in split if x['task'] == 'sentiment']
    print(f"{split_name} label distribution:")
    print(pd.Series(labels).value_counts())
```

**Deliverables:**
- `data_sources.md` documenting dataset provenance and licenses
- Sample count per split
- Initial quality assessment

#### Tuesday (1hr): Label Harmonization
```python
# Verify label mapping consistency
label_dict = {'NEUTRAL/MIXED': 1, 'NEGATIVE': 0, 'POSITIVE': 2}

# Check for edge cases and inconsistencies
unique_labels = set()
for split in ds.values():
    for item in split:
        if item['task'] == 'sentiment':
            unique_labels.add(item['label'])

print(f"Unique labels found: {unique_labels}")
```

**Deliverables:**
- `label_mapping.json` with standardized mapping
- Documentation of edge cases and handling strategy

#### Thursday (1hr): Setup Project Structure
```bash
# Create directory structure
mkdir -p data/{raw,processed,cleaned}
mkdir -p models/{checkpoints,configs}
mkdir -p scripts/{preprocessing,training,evaluation}
mkdir -p notebooks/exploratory
mkdir -p logs

# Initialize git and requirements
git init
touch requirements.txt .gitignore
```

**Deliverables:**
- Proper project structure
- Version control setup
- Basic data loader script

#### Friday (1hr): Initial Cleaning
```python
# Remove exact duplicates and fix encoding
def clean_dataset(dataset):
    # Remove duplicates based on text content
    seen_texts = set()
    cleaned = []
    for item in dataset:
        if item['text'] not in seen_texts:
            cleaned.append(item)
            seen_texts.add(item['text'])
    return cleaned

# Generate cleaning statistics
stats = {
    'original_count': len(ds['train']),
    'duplicates_removed': 0,
    'encoding_fixes': 0
}
```

**Deliverables:**
- `cleaning_stats.json` with preprocessing metrics

### Week 2: Deep Cleaning

#### Monday (1hr): Statistical Analysis
```python
# Analyze text lengths and detect anomalies
lengths = [len(text.split()) for text in texts]
print(f"Text length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths)}")

# Check for non-financial content
suspicious_keywords = ['jewelry', 'heist', 'burglary', 'artifacts']
suspicious_samples = [text for text in texts if any(kw in text.lower() for kw in suspicious_keywords)]
```

#### Tuesday (1hr): GPT-4 Quality Control Setup
```python
# Sample suspicious examples for manual review
import random
random.seed(42)
sample_for_review = random.sample(suspicious_samples, min(500, len(suspicious_samples)))

# Create prompt for GPT-4 verification
verification_prompt = """
Analyze this text and determine:
1. Is this financial/business related? (Yes/No)
2. What is the sentiment? (POSITIVE/NEGATIVE/NEUTRAL)
3. Confidence level (1-5)

Text: {text}
"""
```

#### Thursday (1hr): Implement Cleaning Pipeline
```python
# Process with GPT-4 and log changes
def verify_with_gpt4(text_batch):
    # Implementation for batch verification
    # Log all changes for transparency
    pass

# Create verification report
verification_report = {
    'samples_reviewed': 0,
    'samples_removed': 0,
    'label_changes': 0,
    'confidence_distribution': {}
}
```

#### Friday (1hr): Create Train/Val/Test Splits
```python
from sklearn.model_selection import train_test_split

# Stratified split ensuring label balance
def create_stratified_split(dataset, test_size=0.1, val_size=0.1):
    # Ensure balanced representation across labels and sources
    pass

# Save in HuggingFace format
cleaned_ds.save_to_disk('./data/cleaned/financial_sentiment')
```

### Week 3: Data Finalization

#### Monday (1hr): Create Dataset Card
Create comprehensive `dataset_card.md` with:
- Dataset description and intended use
- Data collection methodology
- Preprocessing steps and decisions
- Bias analysis and limitations
- Evaluation benchmarks
- Citation requirements

#### Tuesday (1hr): Validation Suite
```python
# Unit tests for data integrity
def test_label_consistency():
    assert all(label in [0, 1, 2] for label in all_labels)

def test_no_data_leakage():
    train_texts = set(train_data['text'])
    val_texts = set(val_data['text'])
    test_texts = set(test_data['text'])
    assert len(train_texts & val_texts) == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts & test_texts) == 0
```

#### Thursday (1hr): HuggingFace Publication
```python
# Upload to HuggingFace Hub
from datasets import Dataset
from huggingface_hub import HfApi

# Create dataset repository
dataset = Dataset.from_dict({
    'text': texts,
    'labels': labels,
    'split': splits
})

dataset.push_to_hub('your_username/modern_finbert_dataset')
```

#### Friday (1hr): Community Feedback
- Post dataset announcement on relevant communities
- Set up feedback collection mechanisms
- Plan v0.2 improvements based on initial feedback

### Key Files for Phase 1

**Essential Scripts:**
- `scripts/preprocessing/audit_dataset.py`
- `scripts/preprocessing/clean_and_validate.py`
- `scripts/preprocessing/create_splits.py`

**Documentation:**
- `data_sources.md`
- `label_mapping.json`
- `cleaning_stats.json`
- `dataset_card.md`

**Quality Checks:**
- Label distribution balance across splits
- No data leakage between splits
- Financial relevance verification
- Duplicate removal confirmation

## Notes

This repository is currently in the planning phase with raw data available in `Data.ipynb`. The Phase 1 guidance above provides concrete steps to transform the raw `neoyipeng/financial_reasoning_aggregated` dataset into a high-quality, publication-ready financial sentiment dataset for ModernFinBERT training.