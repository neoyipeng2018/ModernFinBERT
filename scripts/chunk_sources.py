"""
Chunk earnings transcripts and SEC 10-K MD&A sections into medium-length
passages (500-3072 tokens) suitable for ModernFinBERT v2 training.

Usage:
    python scripts/chunk_sources.py

Outputs:
    data/processed/medium_chunks_raw.parquet
"""

import re
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 42
MIN_TOKENS = 500
MAX_TOKENS = 3072
# chars/token ratios from exploration: earnings ~4.7, 10-K ~5.1
# Use conservative 4.5 to avoid under-chunking
CHARS_PER_TOKEN = 4.5
MIN_CHARS = int(MIN_TOKENS * CHARS_PER_TOKEN)  # ~2250
MAX_CHARS = int(MAX_TOKENS * CHARS_PER_TOKEN)  # ~13824
OUTPUT_DIR = Path("data/processed")


def chunk_transcript(text: str) -> list[str]:
    """Split an earnings call transcript on speaker turns.

    Transcripts use "Name : text" format. We split on speaker boundaries
    and group consecutive turns into chunks of MIN_CHARS..MAX_CHARS.
    """
    # Split on speaker turn pattern: "SpeakerName : "
    # The pattern captures the speaker name; re.split returns interleaved
    # [pre, name1, text1, name2, text2, ...] so we reconstruct turns.
    parts = re.split(r'(?:^|\s)([A-Z][a-zA-Z\.\s\-\']{1,50}?)\s*:\s', text)

    # Reconstruct turns as "Name: text" blocks
    turns = []
    if parts[0].strip():
        turns.append(parts[0].strip())
    for i in range(1, len(parts) - 1, 2):
        speaker = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        turns.append(f"{speaker}: {body}")

    if len(turns) <= 1:
        turns = [p.strip() for p in text.split("\n\n") if p.strip()]

    return _group_into_chunks(turns, sep="\n\n")


def chunk_mda(text: str) -> list[str]:
    """Split an MD&A section on structural boundaries.

    MD&A has ALL CAPS HEADERS, numbered subsections, and paragraph breaks.
    """
    section_pattern = re.compile(
        r'\n(?=[A-Z][A-Z\s,&]{5,}\n)'  # ALL CAPS HEADERS
        r'|\n(?=\d+\.\s+[A-Z])'         # "1. Section Name"
        r'|\n(?=Item\s+\d)',             # "Item 7A"
        re.MULTILINE
    )
    sections = section_pattern.split(text)

    if len(sections) <= 1:
        sections = [p.strip() for p in text.split("\n\n") if p.strip()]

    return _group_into_chunks(sections, sep="\n\n")


def _group_into_chunks(segments: list[str], sep: str = "\n\n") -> list[str]:
    """Group text segments into chunks of MIN_CHARS..MAX_CHARS."""
    chunks = []
    current = ""
    for seg in segments:
        candidate = current + sep + seg if current else seg
        if len(candidate) > MAX_CHARS and len(current) >= MIN_CHARS:
            chunks.append(current.strip())
            current = seg
        else:
            current = candidate

    if len(current) >= MIN_CHARS:
        chunks.append(current.strip())
    elif current.strip() and chunks:
        if len(chunks[-1]) + len(current) <= MAX_CHARS * 1.2:
            chunks[-1] += sep + current.strip()

    return chunks


def process_earnings(tokenizer) -> list[dict]:
    print("Loading glopardo/sp500-earnings-transcripts...")
    ds = load_dataset("glopardo/sp500-earnings-transcripts")["train"]
    print(f"  {len(ds)} transcripts loaded")

    all_chunks = []
    empty_count = 0
    for i, row in enumerate(ds):
        text = row["transcript"]
        if not text or len(text) < MIN_CHARS:
            empty_count += 1
            continue

        chunks = chunk_transcript(text)
        company = row.get("company", "UNKNOWN")
        ticker = row.get("ticker", "")

        for chunk in chunks:
            toks = tokenizer(chunk, truncation=False)["input_ids"]
            n_tokens = len(toks)
            if MIN_TOKENS <= n_tokens <= MAX_TOKENS:
                all_chunks.append({
                    "text": chunk,
                    "source": "earnings_transcripts",
                    "source_domain": "earnings_calls",
                    "company_hint": company,
                    "ticker": ticker,
                    "n_tokens": n_tokens,
                })

        if (i + 1) % 2000 == 0:
            print(f"  processed {i+1}/{len(ds)} transcripts, {len(all_chunks)} chunks so far")

    print(f"  Done: {len(all_chunks)} chunks from {len(ds) - empty_count} transcripts ({empty_count} empty)")
    return all_chunks


def process_10k(tokenizer) -> list[dict]:
    print("Loading jlohding/sp500-edgar-10k...")
    ds = load_dataset("jlohding/sp500-edgar-10k")["train"]
    print(f"  {len(ds)} filings loaded")

    all_chunks = []
    empty_count = 0
    for i, row in enumerate(ds):
        text = row["item_7"]
        if not text or len(text) < MIN_CHARS:
            empty_count += 1
            continue

        chunks = chunk_mda(text)
        company = row.get("company", "UNKNOWN")

        for chunk in chunks:
            toks = tokenizer(chunk, truncation=False)["input_ids"]
            n_tokens = len(toks)
            if MIN_TOKENS <= n_tokens <= MAX_TOKENS:
                all_chunks.append({
                    "text": chunk,
                    "source": "sec_10k_mda",
                    "source_domain": "sec_filings",
                    "company_hint": company,
                    "ticker": "",
                    "n_tokens": n_tokens,
                })

        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(ds)} filings, {len(all_chunks)} chunks so far")

    print(f"  Done: {len(all_chunks)} chunks from {len(ds) - empty_count} filings ({empty_count} empty)")
    return all_chunks


def deduplicate(chunks: list[dict]) -> list[dict]:
    print(f"\nDeduplicating {len(chunks)} chunks...")
    seen = set()
    unique = []
    # Exact dedup on normalized text
    for c in chunks:
        key = re.sub(r'\s+', ' ', c["text"].lower().strip())[:500]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    print(f"  Exact dedup: {len(chunks)} -> {len(unique)} (removed {len(chunks) - len(unique)})")

    # Near-duplicate: use first-500-chars fingerprint (fast approximation)
    # Full MinHash is slow for 100K+ rows; this catches most overlaps
    seen_fp = set()
    final = []
    for c in unique:
        fp = re.sub(r'[^a-z0-9]', '', c["text"][:300].lower())
        if fp not in seen_fp:
            seen_fp.add(fp)
            final.append(c)

    print(f"  Near-dedup: {len(unique)} -> {len(final)} (removed {len(unique) - len(final)})")
    return final


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    earnings_chunks = process_earnings(tokenizer)
    mda_chunks = process_10k(tokenizer)

    all_chunks = earnings_chunks + mda_chunks
    print(f"\nTotal raw chunks: {len(all_chunks)}")
    print(f"  Earnings: {len(earnings_chunks)}")
    print(f"  10-K MD&A: {len(mda_chunks)}")

    all_chunks = deduplicate(all_chunks)

    # Token length stats
    token_lens = [c["n_tokens"] for c in all_chunks]
    print(f"\nToken length stats (after dedup):")
    print(f"  n={len(token_lens)}")
    print(f"  min={min(token_lens)}, p25={int(np.percentile(token_lens, 25))}, "
          f"median={int(np.median(token_lens))}, p75={int(np.percentile(token_lens, 75))}, "
          f"max={max(token_lens)}")
    print(f"  mean={np.mean(token_lens):.0f}")

    # Per-source stats
    for src in ["earnings_transcripts", "sec_10k_mda"]:
        src_lens = [c["n_tokens"] for c in all_chunks if c["source"] == src]
        print(f"  {src}: n={len(src_lens)}, median={int(np.median(src_lens))}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_chunks)
    out_path = OUTPUT_DIR / "medium_chunks_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} chunks to {out_path}")


if __name__ == "__main__":
    main()
