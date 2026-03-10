---
name: verbalized-sampling-augment
description: >
  Apply Verbalized Sampling (VS) to generate diverse synthetic training data
  for financial sentiment classifiers. Use this skill when the user wants to
  augment training data for ModernFinBERT or any sentiment model, boost
  classifier accuracy through data augmentation, generate diverse financial
  text paraphrases, apply the VS paper's method to NLP data generation,
  or cover more of the sentiment space in training data. Trigger on mentions
  of "verbalized sampling", "data augmentation", "databoost", "paraphrase
  errors", "augment misclassified", "diverse training data", or "sentiment
  coverage". Also trigger when the user asks to improve model accuracy by
  generating more training examples or wants to fix misclassification patterns.
---

# Verbalized Sampling for Financial Sentiment Augmentation

You (Claude) are the generator.  You will directly produce diverse financial
sentiment training data using the Verbalized Sampling (VS) prompting
structure.  There is no external API script — you embody the VS approach
by generating a *distribution* of candidate texts with probabilities for
each request, then writing the results to a file.

## Background: Why Verbalized Sampling?

Traditional paraphrasing asks for one rewrite at a time and mode-collapses
to near-identical outputs.  Verbalized Sampling (Zhang et al. 2025) instead
asks for **k candidates with explicit probability estimates** in a single
generation.  This forces you to think about the full distribution of
plausible financial phrasings — common AND unusual — recovering diversity
that alignment normally compresses away.

Key findings from the paper:
- VS boosts diversity 1.6-2.1× over direct prompting
- VS-generated synthetic data improved downstream model accuracy from
  32.8% to 37.5% on math benchmarks (VS-Multi best)
- Larger models benefit more — Claude is ideal for this
- The probability threshold parameter enables "diversity tuning"

## How to Execute This Skill

### Step 1: Understand the Input

The user will provide misclassified samples (or samples they want augmented).
Each sample needs:
- **text**: the original financial sentence
- **true_label**: the correct sentiment (NEGATIVE / NEUTRAL / POSITIVE)
- **pred_label** (optional): what the model predicted (for error-targeted augmentation)

The input may come as:
- A CSV/JSON file of errors from an error-mining step
- A HuggingFace dataset reference
- Samples pasted directly in conversation
- A notebook cell's output

If the user hasn't provided structured errors yet, help them extract the
misclassified samples from their model evaluation.

### Step 2: Clarify Parameters

Ask the user (or use sensible defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| **k** | 5 | Candidates per generation batch |
| **variant** | vs-cot | Which VS variant (see below) |
| **prob_threshold** | 0.10 | Set to focus on tail/unusual phrasings; `null` for full distribution |
| **total_per_sample** | 5 | How many augmented texts per input sample |
| **output_format** | csv | csv, json, or jsonl |

### Step 3: Generate Using VS Structure

For each misclassified sample, generate augmented data following the VS-CoT
approach below.  This is the critical part — you must actually think through
and produce diverse candidates as a distribution, not just paraphrase.

#### VS-CoT Generation Process (recommended)

For each input sample, follow this internal reasoning structure:

**A. Analyze the confusion** (your chain-of-thought):
1. Why might a classifier confuse `{pred_label}` with `{true_label}` for
   this text?
2. What linguistic cues, word choices, or structural patterns distinguish
   `{true_label}` in financial text?
3. What diverse financial sub-domains could express this sentiment
   differently? (earnings calls, analyst notes, SEC filings, social media,
   news headlines, press releases, equity research, credit commentary,
   M&A announcements, central bank communications, retail investor forums)

**B. Generate k diverse candidates with probabilities**:
- Each candidate must clearly express the **true label** sentiment
- Span different financial sub-domains and registers
- Include both typical (high-probability) and unusual (low-probability)
  phrasings
- Probabilities should roughly sum to 1.0 and reflect how common each
  phrasing style is among all ways to express that sentiment in finance
- If `prob_threshold` is set (e.g., 0.10), focus on generating candidates
  whose probability is below that threshold — unusual, tail-distribution
  phrasings that a classifier would struggle with

**C. Output each candidate** as a row with: text, label, probability

#### VS-Standard (faster, less diverse)

Skip the chain-of-thought.  Just generate k candidates with probabilities
directly.  Use when processing many samples and speed matters more.

#### VS-Multi (maximum diversity)

Generate k candidates in a first pass, then generate k more that are
deliberately different from the first batch (different sub-domains,
different registers, different sentence structures).  Use when you need
maximum coverage of the sentiment space.

### Step 4: Write Output File

After generating all augmented samples, write them to a structured file:

**CSV format** (default):
```
text,label,label_name,probability,source_variant,seed_text
"The company reported revenue in line with expectations.",1,NEUTRAL,0.25,vs-cot,"[original seed]"
...
```

**JSON format**:
```json
[
  {
    "text": "The company reported revenue in line with expectations.",
    "label": 1,
    "label_name": "NEUTRAL",
    "probability": 0.25,
    "source_variant": "vs-cot",
    "seed_text": "[original seed]"
  }
]
```

Save the file to the user's workspace folder.  Use a descriptive filename
like `vs_augmented_data.csv` or `vs_augmented_{label}_{timestamp}.csv`.

### Step 5: Summary Statistics

After generating, report:
- Total augmented samples generated
- Breakdown by sentiment class
- Breakdown by financial sub-domain (if trackable)
- Diversity assessment: how many distinct phrasings / sub-domains covered
- Any samples that were difficult to augment (and why)

## Worked Example

**Input error**: "Nokia's revenue grew modestly" — true label NEUTRAL,
predicted POSITIVE.

**VS-CoT reasoning**:
The classifier likely saw "revenue grew" and inferred POSITIVE.  But
"modestly" is a hedging qualifier that makes this factual/neutral reporting.
To help the classifier learn this distinction, I need NEUTRAL texts that
contain superficially positive-sounding financial language but are actually
just reporting facts without evaluative sentiment.  I'll span: earnings
reports, analyst commentary, trading updates, regulatory filings, and
financial journalism.

**Generated candidates**:

| text | label | prob |
|------|-------|------|
| "Quarterly earnings came in at $1.05 per share, slightly above the prior year's $1.02." | NEUTRAL | 0.22 |
| "The firm's operating margin remained broadly stable at 12.3% during the reporting period." | NEUTRAL | 0.20 |
| "Management noted incremental progress in the cloud segment without providing updated guidance." | NEUTRAL | 0.18 |
| "Trading activity in the name picked up modestly following the index rebalance announcement." | NEUTRAL | 0.08 |
| "According to the 10-Q, accounts receivable increased by 4% quarter-over-quarter." | NEUTRAL | 0.06 |

Notice: all five are clearly NEUTRAL despite containing words like "above",
"progress", "picked up" that might superficially suggest POSITIVE.  They span
earnings, operations, management commentary, trading, and SEC filings.

## Processing Large Batches

For many samples (>20), process them in batches:

1. Group errors by confusion type (e.g., all NEUTRAL→POSITIVE errors together)
2. Generate for each group, noting which confusion pattern you're targeting
3. Within each group, vary the financial sub-domain systematically to ensure
   coverage
4. Write results incrementally so partial progress is saved

For very large batches (>100 samples), suggest the user whether they want:
- Full VS-CoT for all (highest quality, slower)
- VS-CoT for hardest confusions + VS-Standard for the rest
- A subset of the most representative errors

## Quality Checks

Before finalizing the augmented dataset:

1. **Label fidelity**: Re-read each generated text and confirm it genuinely
   expresses the labeled sentiment.  Remove any ambiguous cases.
2. **Diversity check**: Ensure no two generated texts are near-paraphrases
   of each other.  If they are, regenerate with lower prob_threshold.
3. **Financial authenticity**: Each text should sound like it could appear
   in a real financial document, not like a textbook example.
4. **Length distribution**: Match the length distribution of the original
   training data (typically 10-40 words for FPB-style data).

## Reference Files

Read these for additional context:
- `references/prompt_templates.md` — The exact VS prompt structures from
  the paper, adapted for financial sentiment
- `references/paper_summary.md` — Key empirical findings and recommended
  configurations from Zhang et al. (2025)
