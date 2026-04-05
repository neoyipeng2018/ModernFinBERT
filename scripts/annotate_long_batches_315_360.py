"""
Annotate long financial texts (batches 0315-0360) for:
  - label: overall sentiment (POSITIVE / NEGATIVE / NEUTRAL)
  - entity: primary financial entity
  - entity_sentiment: sentiment toward that entity
"""

import pandas as pd
import re
import os

DATA_DIR = "/Users/boo/Documents/ModernFinBERT/data/processed/entity_annotations_long"

# ── Sentiment keyword lists ──────────────────────────────────────────────
POS_WORDS = {
    # Growth & performance
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
    "resilient", "resilience",
    "milestone",
}

NEG_WORDS = {
    # Decline & weakness
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
    "penalty", "penalties", "fine", "fines",
    "write-off", "write-down", "writedown", "writeoff",
    "charge", "charges",
    "layoff", "layoffs",
    "recession", "recessionary",
    "inflation", "inflationary",
    "unfavorable", "unfavourable",
    "disruption", "disruptions", "disrupted",
    "concern", "concerns", "concerned",
    "deficit", "deficits",
    "suspend", "suspended", "suspension",
    "terminate", "terminated", "termination",
}

# Words that are common in financial texts but don't carry sentiment
NEUTRAL_DAMPENERS = {
    "risk", "risks",  # very common in boilerplate risk factor language
    "charge", "charges",  # accounting term, not always negative
}

# ── Company name patterns ────────────────────────────────────────────────
# Common suffixes to strip for canonical names
COMPANY_SUFFIXES = r"(?:\s*(?:Inc\.?|Corp(?:oration)?\.?|Co\.?|Ltd\.?|LLC|L\.?P\.?|plc|PLC|N\.?V\.?|S\.?A\.?|AG|SE|Group|Holdings?|International|Enterprises?))*"

# Patterns to find company names in opening text
COMPANY_PATTERNS = [
    # Earnings call: "welcome to X's Q1 2022 Earnings"
    re.compile(r"welcome to (?:the )?(.+?)(?:'s|'s)?\s+(?:Q[1-4]|first|second|third|fourth|FY|fiscal|full[- ]year|annual|quarterly|\d{4})", re.IGNORECASE),
    # "welcome to X earnings"
    re.compile(r"welcome to (?:the )?(.+?)\s+(?:earnings|conference|investor|quarterly|annual)", re.IGNORECASE),
    # "This is the X conference call"
    re.compile(r"(?:this is|joining us from)\s+(?:the\s+)?(.+?)\s+(?:conference|earnings|call|investor)", re.IGNORECASE),
    # 10-K/10-Q style: company name at start or in header
    re.compile(r"(?:^|\n)\s*(?:PART\s+[IV]+[.\s]+)?(?:Item\s+\d+[A-Za-z]?\.?\s+)?(?:Management.s Discussion.+?\n\s*)(.+?)(?:\s+(?:and|&)\s+(?:its\s+)?(?:subsidiaries|affiliates))?\s*\n", re.IGNORECASE),
    # "X reported" or "X announced"
    re.compile(r"(?:^|\.\s+)([A-Z][A-Za-z&\s,.']+?)\s+(?:reported|announced|delivered|posted|achieved|recorded|generated)", re.IGNORECASE),
    # Operator intro: "host, Mr/Ms. Name, ... of CompanyName"
    re.compile(r"(?:of|from|for)\s+([A-Z][A-Za-z&\s,.']+?)\s*\.\s+(?:Thank|Please|Go ahead)", re.IGNORECASE),
    # "On behalf of X"
    re.compile(r"[Oo]n behalf of\s+(.+?)(?:,|\s+and\s+|\s+I\s+)", re.IGNORECASE),
    # Ford, Valero style: "CompanyName 10-K" or just big company name mentions
    re.compile(r"(?:^|\n)\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:Motor|Energy|Financial|Capital|Health|Pharma|Tech)", re.IGNORECASE),
]

# Known company name mappings for canonical forms
CANONICAL_NAMES = {
    "apple": "Apple Inc.",
    "microsoft": "Microsoft Corporation",
    "google": "Alphabet Inc.",
    "alphabet": "Alphabet Inc.",
    "amazon": "Amazon.com Inc.",
    "meta": "Meta Platforms Inc.",
    "facebook": "Meta Platforms Inc.",
    "tesla": "Tesla Inc.",
    "nvidia": "NVIDIA Corporation",
    "ford": "Ford Motor Company",
    "ford motor": "Ford Motor Company",
    "general motors": "General Motors Company",
    "gm": "General Motors Company",
    "jpmorgan": "JPMorgan Chase & Co.",
    "jp morgan": "JPMorgan Chase & Co.",
    "goldman sachs": "Goldman Sachs Group Inc.",
    "bank of america": "Bank of America Corporation",
    "wells fargo": "Wells Fargo & Company",
    "citigroup": "Citigroup Inc.",
    "citi": "Citigroup Inc.",
    "morgan stanley": "Morgan Stanley",
    "johnson & johnson": "Johnson & Johnson",
    "j&j": "Johnson & Johnson",
    "pfizer": "Pfizer Inc.",
    "unitedhealth": "UnitedHealth Group Inc.",
    "walmart": "Walmart Inc.",
    "wal-mart": "Walmart Inc.",
    "home depot": "The Home Depot Inc.",
    "chevron": "Chevron Corporation",
    "exxon": "Exxon Mobil Corporation",
    "exxonmobil": "Exxon Mobil Corporation",
    "exxon mobil": "Exxon Mobil Corporation",
    "valero": "Valero Energy Corporation",
    "conocophillips": "ConocoPhillips",
    "conoco": "ConocoPhillips",
    "procter & gamble": "Procter & Gamble Company",
    "p&g": "Procter & Gamble Company",
    "coca-cola": "The Coca-Cola Company",
    "coca cola": "The Coca-Cola Company",
    "pepsico": "PepsiCo Inc.",
    "pepsi": "PepsiCo Inc.",
    "intel": "Intel Corporation",
    "ibm": "IBM Corporation",
    "cisco": "Cisco Systems Inc.",
    "oracle": "Oracle Corporation",
    "salesforce": "Salesforce Inc.",
    "adobe": "Adobe Inc.",
    "netflix": "Netflix Inc.",
    "disney": "The Walt Disney Company",
    "walt disney": "The Walt Disney Company",
    "comcast": "Comcast Corporation",
    "at&t": "AT&T Inc.",
    "verizon": "Verizon Communications Inc.",
    "t-mobile": "T-Mobile US Inc.",
    "boeing": "The Boeing Company",
    "lockheed martin": "Lockheed Martin Corporation",
    "raytheon": "Raytheon Technologies Corporation",
    "caterpillar": "Caterpillar Inc.",
    "deere": "Deere & Company",
    "john deere": "Deere & Company",
    "3m": "3M Company",
    "honeywell": "Honeywell International Inc.",
    "ge": "General Electric Company",
    "general electric": "General Electric Company",
    "ups": "United Parcel Service Inc.",
    "fedex": "FedEx Corporation",
    "berkshire": "Berkshire Hathaway Inc.",
    "berkshire hathaway": "Berkshire Hathaway Inc.",
    "visa": "Visa Inc.",
    "mastercard": "Mastercard Incorporated",
    "paypal": "PayPal Holdings Inc.",
    "square": "Block Inc.",
    "block": "Block Inc.",
    "merck": "Merck & Co. Inc.",
    "abbvie": "AbbVie Inc.",
    "amgen": "Amgen Inc.",
    "gilead": "Gilead Sciences Inc.",
    "eli lilly": "Eli Lilly and Company",
    "lilly": "Eli Lilly and Company",
    "bristol-myers": "Bristol-Myers Squibb Company",
    "bristol myers": "Bristol-Myers Squibb Company",
    "starbucks": "Starbucks Corporation",
    "mcdonald": "McDonald's Corporation",
    "mcdonalds": "McDonald's Corporation",
    "nike": "Nike Inc.",
    "costco": "Costco Wholesale Corporation",
    "target": "Target Corporation",
    "lowes": "Lowe's Companies Inc.",
    "lowe's": "Lowe's Companies Inc.",
    "qualcomm": "Qualcomm Incorporated",
    "broadcom": "Broadcom Inc.",
    "texas instruments": "Texas Instruments Incorporated",
    "amd": "Advanced Micro Devices Inc.",
    "advanced micro": "Advanced Micro Devices Inc.",
    "applied materials": "Applied Materials Inc.",
    "micron": "Micron Technology Inc.",
    "snap": "Snap Inc.",
    "twitter": "Twitter Inc.",
    "uber": "Uber Technologies Inc.",
    "lyft": "Lyft Inc.",
    "airbnb": "Airbnb Inc.",
    "doordash": "DoorDash Inc.",
    "shopify": "Shopify Inc.",
    "zoom": "Zoom Video Communications Inc.",
    "snowflake": "Snowflake Inc.",
    "palantir": "Palantir Technologies Inc.",
    "crowdstrike": "CrowdStrike Holdings Inc.",
    "datadog": "Datadog Inc.",
    "twilio": "Twilio Inc.",
    "okta": "Okta Inc.",
    "zscaler": "Zscaler Inc.",
    "palo alto": "Palo Alto Networks Inc.",
    "fortinet": "Fortinet Inc.",
    "willis towers watson": "Willis Towers Watson",
    "wtw": "Willis Towers Watson",
    "aon": "Aon plc",
    "marsh": "Marsh & McLennan Companies Inc.",
    "chubb": "Chubb Limited",
    "progressive": "Progressive Corporation",
    "allstate": "The Allstate Corporation",
    "travelers": "The Travelers Companies Inc.",
    "metlife": "MetLife Inc.",
    "prudential": "Prudential Financial Inc.",
    "aflac": "Aflac Incorporated",
    "charles schwab": "The Charles Schwab Corporation",
    "schwab": "The Charles Schwab Corporation",
    "blackrock": "BlackRock Inc.",
    "state street": "State Street Corporation",
    "northern trust": "Northern Trust Corporation",
    "bny mellon": "The Bank of New York Mellon Corporation",
    "capital one": "Capital One Financial Corporation",
    "american express": "American Express Company",
    "amex": "American Express Company",
    "discover": "Discover Financial Services",
    "synchrony": "Synchrony Financial",
    "ally": "Ally Financial Inc.",
    "regions": "Regions Financial Corporation",
    "keycorp": "KeyCorp",
    "key": "KeyCorp",
    "fifth third": "Fifth Third Bancorp",
    "huntington": "Huntington Bancshares Incorporated",
    "m&t": "M&T Bank Corporation",
    "zions": "Zions Bancorporation",
    "comerica": "Comerica Incorporated",
    "svb": "SVB Financial Group",
    "silicon valley bank": "SVB Financial Group",
    "first republic": "First Republic Bank",
    "pnc": "PNC Financial Services Group Inc.",
    "truist": "Truist Financial Corporation",
    "u.s. bancorp": "U.S. Bancorp",
    "us bancorp": "U.S. Bancorp",
}


def count_sentiment_words(text_lower):
    """Count positive and negative sentiment words in text."""
    words = set(re.findall(r'\b[a-z][\w-]*\b', text_lower))

    pos_count = len(words & POS_WORDS)
    neg_count = len(words & NEG_WORDS)

    # Also count frequency-weighted matches for stronger signals
    all_words = re.findall(r'\b[a-z][\w-]*\b', text_lower)
    pos_freq = sum(1 for w in all_words if w in POS_WORDS)
    neg_freq = sum(1 for w in all_words if w in NEG_WORDS)

    return pos_count, neg_count, pos_freq, neg_freq


def classify_sentiment(text):
    """Classify overall text sentiment using keyword analysis."""
    text_lower = text.lower()

    pos_count, neg_count, pos_freq, neg_freq = count_sentiment_words(text_lower)

    # Use frequency-weighted scoring
    total_freq = pos_freq + neg_freq
    if total_freq == 0:
        return "NEUTRAL"

    pos_ratio = pos_freq / total_freq

    # Strong positive or negative signals
    if pos_ratio > 0.62:
        return "POSITIVE"
    elif pos_ratio < 0.38:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def extract_entity_from_text(text, first_500):
    """Extract the primary financial entity from the text."""

    # Try regex patterns on the first 500 chars
    for pattern in COMPANY_PATTERNS:
        match = pattern.search(first_500)
        if match:
            name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            name = name.strip(' ,.')
            # Skip if too short or too long or generic
            if 2 < len(name) < 80 and name.lower() not in {'the', 'our', 'this', 'we', 'a', 'an', 'item'}:
                return canonicalize_name(name, text)

    # Fallback: look for known company names in the first 1000 chars
    first_1000_lower = text[:1000].lower()
    best_match = None
    best_pos = len(first_1000_lower)  # Position in text (earlier = better)

    for key, canonical in CANONICAL_NAMES.items():
        pos = first_1000_lower.find(key.lower())
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_match = canonical

    if best_match:
        return best_match

    # Extended fallback: look for "Company" or "Corporation" patterns in first 2000 chars
    first_2000 = text[:2000]
    corp_pattern = re.compile(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+(?:Corporation|Company|Inc\.|Corp\.|Ltd\.)', re.IGNORECASE)
    match = corp_pattern.search(first_2000)
    if match:
        name = match.group(0).strip()
        if len(name) > 3:
            return canonicalize_name(name, text)

    # Check for "we" or "our" which indicates self-reference in 10-K
    if re.search(r'\b(?:we|our|the company)\b', first_500, re.IGNORECASE):
        # Try to find any company name in a wider search
        for key, canonical in CANONICAL_NAMES.items():
            if key.lower() in text[:5000].lower():
                return canonical

    return "MARKET"


def canonicalize_name(name, full_text):
    """Convert extracted name to canonical form."""
    name_clean = name.strip(' ,.')
    name_lower = name_clean.lower()

    # Remove common prefixes
    for prefix in ['the ', 'a ']:
        if name_lower.startswith(prefix):
            name_clean = name_clean[len(prefix):]
            name_lower = name_clean.lower()

    # Check canonical mapping
    for key, canonical in CANONICAL_NAMES.items():
        if key in name_lower or name_lower in key:
            # Verify it appears in the text
            if key in full_text[:5000].lower():
                return canonical

    # Clean up suffix for better presentation
    name_clean = re.sub(r'\s*,\s*$', '', name_clean)

    # If name is just a person's name (common in earnings calls), look deeper
    if len(name_clean.split()) <= 2 and not any(s in name_clean for s in ['Inc', 'Corp', 'Co.', 'Ltd', 'LLC']):
        # Could be a person's name from earnings call intro; try harder
        for key, canonical in CANONICAL_NAMES.items():
            if key in full_text[:5000].lower():
                return canonical

    if len(name_clean) < 3:
        return "MARKET"

    return name_clean


def entity_sentiment(text, entity):
    """Determine sentiment specifically toward the identified entity."""
    if entity == "MARKET" or entity == "NONE":
        return classify_sentiment(text)

    # For a specific entity, look at sentences mentioning it
    text_lower = text.lower()
    entity_lower = entity.lower()

    # Get first word of entity for broader matching
    entity_words = entity_lower.split()
    primary_word = entity_words[0] if entity_words else entity_lower

    # Skip very common words
    if primary_word in {'the', 'a', 'an', 'of'}:
        primary_word = entity_words[1] if len(entity_words) > 1 else entity_lower

    # Find sentences containing the entity or its primary word
    sentences = re.split(r'[.!?]+', text)
    entity_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if primary_word in s_lower or entity_lower in s_lower:
            entity_sentences.append(s)

    if not entity_sentences:
        # Entity not found in sentence-split text; use overall sentiment
        return classify_sentiment(text)

    # Analyze sentiment of entity-related sentences
    entity_text = ' '.join(entity_sentences)
    return classify_sentiment(entity_text)


def verify_entity_in_text(entity, text):
    """Verify that the entity name actually appears in the text."""
    if entity in ("MARKET", "NONE"):
        return True

    text_lower = text.lower()
    entity_lower = entity.lower()

    # Direct match
    if entity_lower in text_lower:
        return True

    # Check first word
    first_word = entity_lower.split()[0]
    if first_word in text_lower and len(first_word) > 2:
        return True

    # Check canonical name parts
    for part in entity.split():
        if len(part) > 3 and part.lower() in text_lower:
            return True

    return False


def process_batch(batch_num):
    """Process a single batch file."""
    input_path = os.path.join(DATA_DIR, f"batch_{batch_num:04d}_input.csv")
    output_path = os.path.join(DATA_DIR, f"batch_{batch_num:04d}_output.csv")

    if not os.path.exists(input_path):
        print(f"  SKIP: {input_path} not found")
        return

    df = pd.read_csv(input_path, index_col=0)

    results = []
    for idx in df.index:
        text = str(df.loc[idx, 'text'])
        first_500 = text[:500]

        # 1. Overall sentiment
        label = classify_sentiment(text)

        # 2. Entity extraction
        entity = extract_entity_from_text(text, first_500)

        # 3. Verify entity appears in text
        if not verify_entity_in_text(entity, text):
            entity = "MARKET"

        # 4. Entity sentiment
        ent_sent = entity_sentiment(text, entity)

        results.append({
            'index': idx,
            'label': label,
            'entity': entity,
            'entity_sentiment': ent_sent,
        })

    out_df = pd.DataFrame(results).set_index('index')
    out_df.to_csv(output_path)

    # Summary stats
    label_counts = out_df['label'].value_counts().to_dict()
    entity_counts = out_df['entity'].value_counts()
    top_entities = entity_counts.head(3).to_dict()

    return label_counts, top_entities


def main():
    print("=" * 70)
    print("Processing batches 0315-0360 (46 batches, ~4525 rows)")
    print("=" * 70)

    all_labels = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    all_entities = {}

    for batch_num in range(315, 361):
        print(f"\nBatch {batch_num:04d}...", end=" ")
        result = process_batch(batch_num)
        if result:
            label_counts, top_entities = result
            print(f"Labels: {label_counts} | Top entities: {list(top_entities.keys())[:3]}")
            for k, v in label_counts.items():
                all_labels[k] = all_labels.get(k, 0) + v
            for k, v in top_entities.items():
                all_entities[k] = all_entities.get(k, 0) + v

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total label distribution: {all_labels}")
    total = sum(all_labels.values())
    if total > 0:
        for k, v in all_labels.items():
            print(f"    {k}: {v} ({100*v/total:.1f}%)")

    print(f"\n  Top 15 entities:")
    sorted_entities = sorted(all_entities.items(), key=lambda x: -x[1])
    for name, count in sorted_entities[:15]:
        print(f"    {name}: {count}")

    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
