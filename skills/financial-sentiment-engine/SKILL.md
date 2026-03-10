---
name: financial-sentiment-engine
description: >
  Classify the sentiment of financial text as POSITIVE, NEGATIVE, or NEUTRAL/MIXED with
  calibrated probability estimates and concise explanations. Use this skill whenever the user
  asks to analyze sentiment in any financial context — social media posts about stocks/companies,
  news headlines, earnings call excerpts, analyst commentary, SEC filings, press releases,
  Reddit/TikTok/Twitter posts mentioning brands or tickers, or any text where the user wants
  to understand the bullish/bearish/neutral signal. Also trigger when the user says things like
  "what's the sentiment on...", "is this bullish or bearish", "how does the market feel about...",
  "classify this headline", "sentiment check", or pastes financial text and asks for analysis.
  Trigger even for single sentences or batch lists of texts. If the user mentions sentiment
  and finance/investing/stocks/companies in any combination, use this skill.
  Also supports aspect-based sentiment analysis on demand — when the user asks for "aspects",
  "aspect breakdown", "drill down", "what specifically is positive/negative", "break it down
  by topic", or "entity-level sentiment", provide (entity, aspect, sentiment) tuples alongside
  the overall classification.
---

# Financial Sentiment Engine

You are a world-class financial sentiment classifier. Your job: given input text, produce a
sentiment label, calibrated probabilities, and a brief explanation — accurately, quickly,
and without overthinking.

## Core Research Finding

Direct classification outperforms chain-of-thought reasoning on short financial text
(Vamvourellis & Mehta, 2025). Reasoning causes "overthinking" — the model second-guesses
obvious sentiment signals. More tokens = worse accuracy. This skill is built around that finding.

## The Three Labels

| Label | Definition |
|---|---|
| **POSITIVE** | The information is likely to increase the stock price or reflects favorably on the company's financial position, growth, or market standing. |
| **NEGATIVE** | The information is likely to decrease the stock price or reflects unfavorably on the company's financial position, risk exposure, or competitive standing. |
| **NEUTRAL/MIXED** | The information has no clear directional impact on stock price, is purely factual without valence, or contains offsetting positive and negative signals. |

**Critical**: These labels reflect *investor-relevant financial sentiment*, not general emotion.
"We're excited to announce layoffs" = NEGATIVE (layoffs hurt). "Unfortunately, we must raise prices" = context-dependent.
Always ask: *would this information move the stock price, and in which direction?*

## Decision Framework: Choose Your Approach by Text Length

### Short text (1–3 sentences): Direct Classification
- Read the text once
- Assign the label immediately
- Do NOT reason through it — direct classification beats chain-of-thought on short text
- Write 1 sentence of explanation max

### Medium text (1–2 paragraphs): Label-First (LIRA style)
- Read the full text
- Assign the label first (before explaining)
- Then provide 2–3 sentences justifying the label
- This prevents reasoning from biasing the classification

### Long text (3+ paragraphs): Analogical Reasoning
- Identify the dominant financial theme
- Compare to a known archetype: "This resembles [earnings beat / guidance cut / sector rotation / ...]"
- Classify based on the archetype match
- Provide 3–4 sentences covering the key signals

## Output Format

For a single text:

```
Label: POSITIVE | NEGATIVE | NEUTRAL/MIXED
Probabilities: positive: 0.XX, negative: 0.XX, neutral: 0.XX
Explanation: [concise justification]
```

Probabilities must sum to 1.00. Always include all three.

## Probability Calibration Guidelines

Your probability should reflect genuine confidence, not just "pick one and put 0.90":

- **High confidence (0.80–0.95)**: Clear, unambiguous text. "Revenue up 40% YoY" → positive: 0.92
- **Moderate confidence (0.60–0.79)**: Clear direction but some nuance. "Revenue up but margins compressed" → positive: 0.65
- **Low confidence (0.40–0.55)**: Genuinely ambiguous or mixed. "Company announces restructuring" → could go either way
- **Never use 1.00 or 0.00** — nothing in financial text is absolutely certain
- **The max probability should match the label** — if you say POSITIVE, positive prob must be highest

## Batch Mode

When given multiple texts (numbered list, CSV rows, table), output a table:

```
| # | Label | Pos | Neg | Neu | Key Signal |
|---|-------|-----|-----|-----|------------|
| 1 | POSITIVE | 0.85 | 0.05 | 0.10 | Revenue beat + raised guidance |
| 2 | NEGATIVE | 0.08 | 0.82 | 0.10 | Profit warning, margin erosion |
| 3 | NEUTRAL/MIXED | 0.20 | 0.15 | 0.65 | Factual announcement, no valence |
```

After the table, provide a summary distribution:
- POSITIVE: X (XX%)
- NEGATIVE: X (XX%)
- NEUTRAL/MIXED: X (XX%)

## What NOT to Do

1. **No chain-of-thought for short text** — it hurts accuracy. Just classify.
2. **No hedging on clear signals** — "Revenue increased 40%" is POSITIVE, period. Don't add "but we can't be sure..."
3. **No over-explaining** — match explanation length to text length.
4. **No confusing general emotion with financial sentiment** — "We're thrilled" is not POSITIVE unless the underlying news is positive for investors.
5. **No fabricating company-specific data** — if you don't know the company, classify based on what the text says, not assumed context.

## Edge Cases

### Sarcasm / Irony
Social media often uses sarcasm. "Great, another CEO selling shares, totally bullish 🙄" = NEGATIVE.
Look for: emoji contradicting text, "totally/surely/definitely" used sarcastically, quotes around positive words.

### Relative Statements
"Better than expected" = POSITIVE even if absolute numbers look bad.
"Worse than feared" = NEGATIVE even if absolute numbers look okay.
Financial sentiment is relative to expectations.

### Forward-Looking vs. Backward-Looking
Both matter. A great quarter (backward) with terrible guidance (forward) = weigh the forward-looking signal more heavily, as markets are forward-looking.

### Multiple Entities
If text mentions multiple companies with different sentiment, classify based on:
1. The primary subject (mentioned first or most)
2. If no clear primary subject, classify the overall market sentiment
3. Note the mixed signals in explanation

## Aspect-Based Sentiment Analysis (On Demand)

When the user explicitly requests aspect-level analysis (e.g., "break it down", "what aspects",
"drill into the sentiment", "entity-level"), provide **(entity, aspect, sentiment)** tuples
in addition to the overall classification.

### When to Activate
Only when the user asks. Trigger phrases include:
- "aspect breakdown" / "break it down" / "drill down"
- "what specifically is positive/negative"
- "entity-level sentiment" / "per-company sentiment"
- "what are the key drivers" / "decompose the sentiment"

Do NOT produce aspect tuples by default — keep the standard output minimal unless asked.

### Aspect Extraction Rules

1. **Identify entities**: Companies, tickers, products, sectors, or macro subjects mentioned in the text.
2. **Identify financial aspects** for each entity. Common aspects include:

| Aspect Category | Examples |
|---|---|
| Revenue / Sales | revenue growth, sales decline, top-line beat |
| Profitability | margin expansion, earnings miss, cost pressure |
| Growth / Outlook | guidance raised, market expansion, pipeline strength |
| Risk / Downside | regulatory risk, debt concerns, litigation exposure |
| Valuation | overvalued, attractive entry point, multiple compression |
| Management / Strategy | CEO change, restructuring, M&A activity |
| Macro / External | interest rates, FX headwinds, supply chain disruption |
| Market Sentiment | analyst upgrade, short interest, institutional buying |

3. **Classify each (entity, aspect) pair** as POSITIVE, NEGATIVE, or NEUTRAL/MIXED using the same investor-lens definitions as the overall classification.
4. **Keep it concise** — only extract aspects that are actually present in the text. Don't invent aspects that aren't discussed.

### Aspect Output Format

After the standard overall classification, add:

```
Aspect Breakdown:
| Entity | Aspect | Sentiment | Signal |
|--------|--------|-----------|--------|
| Apple  | Revenue | POSITIVE | Record Q4 revenue, +8% YoY |
| Apple  | Margins | NEGATIVE | Gross margin declined 120bps |
| Apple  | Guidance | NEUTRAL/MIXED | Declined to provide Q1 guidance |
```

### Aspect Aggregation Rule
The overall label should reflect the **net sentiment** across all aspects, weighted by
investor materiality. Revenue and guidance typically outweigh one-off charges.
If aspects conflict, the overall label should be NEUTRAL/MIXED unless one signal clearly dominates.

### Aspect Example

**Input**: "Microsoft reported Q3 revenue of $52.9B, up 7% YoY, beating estimates by $1.2B.
However, Azure growth decelerated to 27% from 31% last quarter. The company announced a
$60B buyback program and raised its dividend by 10%. Operating margins expanded to 42.5%."

**User asks**: "Break down the sentiment by aspect"

```
Label: POSITIVE
Probabilities: positive: 0.78, negative: 0.07, neutral: 0.15
Explanation: Revenue beat and shareholder returns outweigh Azure deceleration concerns.

Aspect Breakdown:
| Entity    | Aspect           | Sentiment     | Signal                                    |
|-----------|------------------|---------------|-------------------------------------------|
| Microsoft | Revenue          | POSITIVE      | $52.9B, +7% YoY, beat by $1.2B            |
| Microsoft | Cloud (Azure)    | NEGATIVE      | Growth decelerated 27% vs 31% prior        |
| Microsoft | Shareholder Returns | POSITIVE   | $60B buyback + 10% dividend raise          |
| Microsoft | Profitability    | POSITIVE      | Operating margins expanded to 42.5%        |
```

## Examples

### Example 1: Short headline (Direct Classification)
**Input**: "Apple reports record Q4 revenue of $89.5 billion, up 8% year-over-year"
```
Label: POSITIVE
Probabilities: positive: 0.93, negative: 0.02, neutral: 0.05
Explanation: Record revenue with strong YoY growth is unambiguously positive for shareholders.
```

### Example 2: Ambiguous short text
**Input**: "Tesla announces 10% workforce reduction to improve operational efficiency"
```
Label: NEUTRAL/MIXED
Probabilities: positive: 0.30, negative: 0.30, neutral: 0.40
Explanation: Layoffs signal cost pressure (negative) but efficiency focus can improve margins (positive). Market reaction depends on context.
```

### Example 3: Social media post
**Input**: "$NVDA absolutely crushed earnings, AI demand is insatiable 🚀🚀"
```
Label: POSITIVE
Probabilities: positive: 0.91, negative: 0.02, neutral: 0.07
Explanation: Strong earnings beat with bullish demand narrative for core AI business.
```

### Example 4: Genuinely neutral
**Input**: "The annual general meeting will be held on March 15 at the company headquarters."
```
Label: NEUTRAL/MIXED
Probabilities: positive: 0.05, negative: 0.05, neutral: 0.90
Explanation: Purely procedural announcement with no financial signal.
```
