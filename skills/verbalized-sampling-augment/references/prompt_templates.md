# VS Prompt Templates — Internal Reasoning Structure

These templates describe how Claude should internally structure its generation
when acting as the VS generator.  They are NOT API prompts — they are the
reasoning patterns Claude follows when producing augmented data.

## VS-CoT Internal Structure (Recommended)

For each misclassified sample, follow this structure:

### Phase 1: Chain-of-Thought Analysis

Think through these questions before generating:

1. **Confusion analysis**: Why would a transformer classifier confuse
   `{pred_label}` with `{true_label}` for this specific text?  What
   surface-level cues (keywords, patterns) might mislead it?

2. **Distinguishing cues**: What makes text genuinely `{true_label}` in
   financial contexts?  What are the subtle signals?
   - POSITIVE: strong directional language ("surged", "exceeded", "record"),
     forward-looking optimism, upgraded guidance, analyst upgrades
   - NEGATIVE: loss language ("declined", "fell short", "downgraded"),
     risk warnings, write-downs, missed targets
   - NEUTRAL: factual reporting, hedged language, descriptive without
     evaluation, procedural/administrative, merely stating numbers

3. **Sub-domain planning**: Which financial text types should the k
   candidates span?  Pick from:
   - Earnings call transcripts (management commentary)
   - Analyst research notes / equity research
   - News headlines and financial journalism
   - SEC filings (10-K, 10-Q, 8-K language)
   - Press releases (corporate communications)
   - Social media / retail investor commentary
   - Credit rating agency commentary
   - M&A and deal announcements
   - Central bank / monetary policy statements
   - Trading desk / market color commentary

### Phase 2: Generate k Candidates

For each candidate, produce:
- **text**: A realistic financial sentence (10-40 words)
- **probability**: How likely this phrasing is among ALL possible ways to
  express `{true_label}` sentiment in financial text (should sum to ~1.0
  across all k candidates)
- **sub_domain**: Which financial text type this represents

Ordering: Start with the most common/typical phrasing (highest probability),
then progress toward more unusual ones (lower probability).

### Phase 3: Tail Sampling (if prob_threshold is set)

If the user set a probability threshold (e.g., 0.10), ONLY generate
candidates whose probability is below that threshold.  This means:
- Skip the obvious/common phrasings entirely
- Focus on unusual registers, rare sub-domains, and atypical sentence
  structures that still clearly express the target sentiment
- Think about: what would a financial NLP model trained on news headlines
  NEVER see?  Generate that.

## VS-Standard Internal Structure

Same as VS-CoT but skip Phase 1 (the chain-of-thought analysis).
Go directly to generating k diverse candidates with probabilities.
Use when batch processing many samples and the confusion patterns are
straightforward.

## VS-Multi Internal Structure

### Turn 1: Generate first k candidates
Same as VS-Standard — k candidates with probabilities.

### Turn 2+: Generate k MORE candidates
Look at what you already generated and deliberately fill gaps:
- Different sub-domains than Turn 1
- Different sentence structures (declarative vs. quoted vs. reported)
- Different registers (formal SEC language vs. casual social media)
- Different complexity levels (short headline vs. multi-clause)

Repeat until `total_per_sample` is reached.

## Financial Sentiment Decision Boundaries

When generating near-boundary examples (the most valuable for training),
keep these distinctions sharp:

### NEUTRAL vs. POSITIVE boundary
- NEUTRAL: "Revenue increased 3% year-over-year" (factual statement)
- POSITIVE: "Revenue surged an impressive 3% year-over-year" (evaluative)
- Key signals: hedging words, absence of evaluative adjectives,
  "in line with expectations" framing

### NEUTRAL vs. NEGATIVE boundary
- NEUTRAL: "The company announced a restructuring plan" (factual)
- NEGATIVE: "The company was forced to announce a painful restructuring" (evaluative)
- Key signals: causal framing, consequence language, magnitude qualifiers

### POSITIVE vs. NEGATIVE (rare confusion, but important)
- POSITIVE: "Despite challenges, the company exceeded all targets"
- NEGATIVE: "Despite exceeding targets, significant challenges remain"
- Key signals: which clause gets emphasis, what's the takeaway

## Example Full Generation

**Input**: "Nokia's revenue grew modestly" — true: NEUTRAL, predicted: POSITIVE

**Phase 1 (CoT)**:
The classifier likely keyed on "revenue grew" — growth language typically
signals POSITIVE.  But "modestly" is a hedging qualifier that neutralizes
the sentiment.  NEUTRAL financial text often reports factual changes
(even growth) without evaluative framing.  I need texts that contain
superficially positive words but are clearly just reporting facts.

I'll span: earnings report language, analyst notes, trading commentary,
SEC filing language, and press release style.

**Phase 2 (candidates)**:

| # | text | prob | sub_domain |
|---|------|------|------------|
| 1 | "Quarterly earnings came in at $1.05 per share, slightly above the prior year's $1.02." | 0.22 | earnings_report |
| 2 | "The firm's operating margin remained broadly stable at 12.3% during the reporting period." | 0.20 | analyst_note |
| 3 | "Management noted incremental progress in the cloud segment without providing updated guidance." | 0.18 | earnings_call |
| 4 | "Trading activity in the name picked up modestly following the index rebalance announcement." | 0.08 | trading_commentary |
| 5 | "According to the 10-Q, accounts receivable increased by 4% quarter-over-quarter." | 0.06 | sec_filing |

All clearly NEUTRAL despite containing growth-adjacent language.  Spans
5 different sub-domains.  Probabilities reflect that earnings report style
is more common than SEC filing style for expressing neutral growth.
