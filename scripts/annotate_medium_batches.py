"""
Annotate medium-length financial text chunks with entity and sentiment labels.

For each row, determines:
  - label: Overall sentiment (POSITIVE / NEGATIVE / NEUTRAL)
  - entity: Primary financial entity mentioned
  - entity_sentiment: Sentiment toward that entity

Uses the company_hint from the source data to assist entity identification,
then applies keyword-based sentiment analysis calibrated for financial text.

Usage:
    python scripts/annotate_medium_batches.py [--start N] [--end N]
"""

import os
import re
import sys
import argparse
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed/entity_annotations_medium")

POSITIVE_WORDS = {
    "growth", "grew", "growing", "increase", "increased", "increases", "increasing",
    "improve", "improved", "improvement", "improvements", "improving",
    "strong", "stronger", "strongest", "strength", "strengths",
    "record", "exceeded", "exceeding", "outperform", "outperformed",
    "higher", "gains", "gain", "gained", "positive", "positively",
    "robust", "solid", "favorable", "favourable", "healthy",
    "momentum", "accelerate", "accelerated", "acceleration",
    "expand", "expanded", "expanding", "expansion",
    "profit", "profitable", "profitability", "profits",
    "benefit", "benefits", "benefited", "beneficial",
    "success", "successful", "successfully",
    "opportunity", "opportunities", "optimistic", "optimism",
    "upside", "upturn", "upbeat",
    "recovery", "recovered", "recovering",
    "efficient", "efficiency", "efficiencies",
    "innovation", "innovative",
    "dividend", "dividends", "repurchase", "repurchases", "buyback",
    "upgrade", "upgraded", "upgrades",
    "surpass", "surpassed", "surpassing",
    "exceptional", "excellent", "outstanding",
    "resilient", "resilience", "milestone",
    "beat", "beats", "beating", "outpacing",
    "raised", "raising", "guidance",
}

NEGATIVE_WORDS = {
    "decline", "declined", "declining", "declines",
    "decrease", "decreased", "decreasing", "decreases",
    "loss", "losses", "lost",
    "weak", "weaker", "weakness", "weakened", "weakening",
    "lower", "lowest",
    "negative", "negatively", "adverse", "adversely",
    "impairment", "impaired", "impairments",
    "restructuring", "restructured",
    "headwind", "headwinds",
    "risk", "risks",
    "challenge", "challenges", "challenging",
    "difficult", "difficulty", "difficulties",
    "uncertain", "uncertainty", "uncertainties",
    "volatility", "volatile",
    "downturn", "downgrade", "downgraded",
    "pressure", "pressures", "pressured",
    "deteriorate", "deteriorated", "deterioration", "deteriorating",
    "shortfall", "shortfalls",
    "default", "defaults", "defaulted",
    "delinquent", "delinquency", "delinquencies",
    "litigation", "lawsuit", "lawsuits",
    "penalty", "penalties",
    "write-off", "write-down", "writedown", "writeoff",
    "layoff", "layoffs",
    "recession", "recessionary",
    "inflation", "inflationary",
    "unfavorable", "unfavourable",
    "disruption", "disruptions", "disrupted",
    "concern", "concerns", "concerned",
    "deficit", "deficits",
    "suspend", "suspended", "suspension",
    "terminate", "terminated", "termination",
    "missed", "miss", "below", "fell", "fall", "falling",
    "cut", "cuts", "cutting", "reduced", "reducing", "reduction",
}

STRONG_POS = {
    "record revenue", "record earnings", "record profit", "all-time high",
    "exceeded expectations", "above expectations", "beat expectations",
    "raised guidance", "raising guidance", "increased guidance",
    "strong growth", "robust growth", "accelerating growth",
    "margin expansion", "significant improvement",
    "double-digit growth", "best quarter", "best year",
}

STRONG_NEG = {
    "net loss", "operating loss", "significant decline",
    "material weakness", "going concern",
    "missed expectations", "below expectations",
    "lowered guidance", "reduced guidance", "cut guidance",
    "goodwill impairment", "asset impairment",
    "credit downgrade", "bankruptcy", "chapter 11",
    "sec investigation", "class action", "shareholder lawsuit",
    "revenue decline", "earnings decline", "profit decline",
}


def score_sentiment(text: str) -> str:
    text_lower = text.lower()
    words = re.findall(r'\b[a-z][\w-]*\b', text_lower)

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    for phrase in STRONG_POS:
        if phrase in text_lower:
            pos_count += 3
    for phrase in STRONG_NEG:
        if phrase in text_lower:
            neg_count += 3

    total = pos_count + neg_count
    if total == 0:
        return "NEUTRAL"

    ratio = pos_count / total
    if ratio > 0.62:
        return "POSITIVE"
    elif ratio < 0.38:
        return "NEGATIVE"
    return "NEUTRAL"


def extract_entity(text: str, company_hint: str) -> str:
    if company_hint and company_hint != "UNKNOWN":
        hint_lower = company_hint.lower().strip()
        text_lower = text[:3000].lower()

        if hint_lower in text_lower:
            return company_hint.strip()

        for word in company_hint.split():
            if len(word) > 3 and word.lower() in text_lower:
                return company_hint.strip()

    # Fallback: look for "Company Inc." patterns
    corp_match = re.search(
        r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+'
        r'(?:Corporation|Company|Inc\.|Corp\.|Ltd\.|Holdings|Incorporated|Group)',
        text[:3000],
    )
    if corp_match:
        return corp_match.group(0).strip()

    return "MARKET"


def entity_sentiment(text: str, entity: str) -> str:
    if entity in ("MARKET", "NONE"):
        return score_sentiment(text)

    entity_lower = entity.lower()
    skip = {"the", "a", "an", "of", "and", "inc.", "inc", "corp.", "corp",
            "corporation", "company", "ltd.", "ltd", "llc", "group", "holdings"}
    distinctive = [w for w in entity.split() if w.lower() not in skip and len(w) > 2]
    primary = distinctive[0].lower() if distinctive else entity_lower.split()[0]

    sentences = re.split(r'[.!?]+', text)
    entity_sentences = [s for s in sentences if primary in s.lower()]

    if not entity_sentences:
        return score_sentiment(text)

    return score_sentiment(' '.join(entity_sentences))


def process_batch(batch_num: int) -> dict | None:
    input_path = DATA_DIR / f"batch_{batch_num:04d}_input.csv"
    output_path = DATA_DIR / f"batch_{batch_num:04d}_output.csv"

    if output_path.exists():
        return None  # already done

    if not input_path.exists():
        return None

    df = pd.read_csv(input_path, index_col=0)

    results = []
    for idx in df.index:
        text = str(df.loc[idx, "text"])
        company_hint = str(df.loc[idx, "company_hint"]) if "company_hint" in df.columns else ""

        label = score_sentiment(text)
        entity = extract_entity(text, company_hint)
        ent_sent = entity_sentiment(text, entity)

        results.append({
            "label": label,
            "entity": entity,
            "entity_sentiment": ent_sent,
            "label_confidence": "agent",
        })

    out_df = pd.DataFrame(results, index=df.index)
    out_df.to_csv(output_path)

    label_counts = out_df["label"].value_counts().to_dict()
    return label_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    all_batches = sorted(DATA_DIR.glob("batch_*_input.csv"))
    n_total = len(all_batches)
    end = args.end if args.end > 0 else n_total

    print(f"Processing batches {args.start} to {end - 1} ({end - args.start} batches)")

    done = 0
    skipped = 0
    all_labels = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for batch_num in range(args.start, end):
        result = process_batch(batch_num)
        if result is None:
            skipped += 1
            continue
        done += 1
        for k, v in result.items():
            all_labels[k] = all_labels.get(k, 0) + v

        if (done) % 50 == 0:
            print(f"  Completed {done} batches, labels so far: {all_labels}")

    print(f"\nDone: {done} batches processed, {skipped} skipped (already done or missing)")
    print(f"Label distribution: {all_labels}")


if __name__ == "__main__":
    main()
