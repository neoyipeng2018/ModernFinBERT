# Verbalized Sampling Paper — Key Findings for Financial Sentiment

Reference: Zhang et al. (2025) "Verbalized Sampling: How to Mitigate Mode
Collapse and Unlock LLM Diversity" (arXiv:2510.01171v3)

## Core Problem: Mode Collapse in LLM Data Generation

Post-training alignment (RLHF) causes LLMs to favor a narrow set of
"typical" outputs — mode collapse. When you ask an aligned model to
paraphrase a sentence 5 times, you often get 5 near-identical results.

The root cause is **typicality bias**: human annotators in preference data
systematically favor familiar, fluent, predictable text. This bias gets
amplified during RLHF, sharpening the output distribution toward modes.

## Solution: Verbalized Sampling

Instead of asking for one output at a time (which collapses to the mode),
ask the model to generate a **distribution** of outputs with explicit
probabilities. This recovers the diversity of the pre-training distribution.

### Three Variants

1. **VS-Standard**: "Generate 5 responses with their probabilities"
   - Single call, returns k candidates with probability estimates
   - 1.6-2.1x more diverse than direct prompting

2. **VS-CoT**: "Think step-by-step, then generate 5 with probabilities"
   - Adds chain-of-thought reasoning before generation
   - Best quality-diversity tradeoff (Pareto optimal in the paper)
   - Recommended for synthetic data generation

3. **VS-Multi**: Multi-turn, each turn generates k more conditioned on prior
   - Maximum diversity through iterative exploration
   - Best average downstream accuracy (37.5% on math benchmarks)

## Key Results Relevant to Data Augmentation

### Synthetic Data Generation (Section 8)

The paper directly tested VS for generating synthetic training data:
- Generated 1K synthetic math questions with different prompting methods
- Fine-tuned Qwen models on the synthetic data
- **VS-Multi achieved 37.5% average accuracy** vs. 32.8% baseline
- Direct prompting sometimes *hurt* performance (30.6%) due to lack of
  diversity in generated data

### Diversity Tuning (probability threshold)

A unique feature of VS: you can tune diversity by setting a probability
threshold. "Generate responses with probability below 0.10" forces the
model to explore the distribution tails. Lower threshold = more unusual
outputs. This is orthogonal to temperature — VS + temperature gives
even better diversity-quality tradeoffs.

### Emergent Trend: Larger Models Benefit More

Across all experiments, more capable models benefit more from VS.
Claude-4-Sonnet and GPT-4.1 showed 1.5-2x larger diversity gains than
smaller models. This means using a strong model (Claude Sonnet/Opus)
as the data generator is important.

## Application to Financial Sentiment

### Why VS is Especially Good for Financial Sentiment

1. **Financial text has high intrinsic diversity**: Earnings calls, news,
   social media, SEC filings, analyst notes — same sentiment expressed
   in radically different language. Direct prompting collapses to the
   most "typical" (news headline style).

2. **Boundary cases are critical**: The hardest samples for sentiment
   classifiers are at the NEUTRAL↔POSITIVE and NEUTRAL↔NEGATIVE
   boundaries. VS naturally generates more boundary-region text because
   it samples from the full distribution.

3. **Sub-domain coverage matters**: A classifier trained only on news
   headlines fails on earnings call transcripts. VS's diversity
   ensures training data spans financial sub-domains.

### Recommended Configuration

Based on the paper's findings:
- **Variant**: VS-CoT (best quality-diversity tradeoff)
- **k**: 5 candidates per call
- **prob_threshold**: 0.10 (moderate tail sampling)
- **Model**: Claude Sonnet or better
- **Temperature**: 0.7 (paper's default, can combine with VS)

### Expected Impact

For the ModernFinBERT DataBoost workflow:
- Direct paraphrasing generates ~3 near-identical rewrites per error
- VS-CoT should generate 5 meaningfully different rewrites spanning
  different financial contexts and linguistic styles
- The paper's synthetic data results suggest 3-5% accuracy improvement
  is achievable over naive paraphrasing
