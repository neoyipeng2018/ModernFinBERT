"""
Annotate long financial texts for sentiment and entity extraction.
Processes batch files 0000-0044 in entity_annotations_long/.

For each row, determines:
  - label: Overall sentiment (POSITIVE / NEGATIVE / NEUTRAL)
  - entity: Primary financial entity mentioned
  - entity_sentiment: Sentiment toward that entity
"""

import os
import re
import pandas as pd

DATA_DIR = "/Users/boo/Documents/ModernFinBERT/data/processed/entity_annotations_long"

# ── Canonical name mapping (key -> canonical) ────────────────────────────────
# Keys MUST be matched as whole words via regex to avoid substring collisions.
# Use _match_canonical() instead of raw "in" checks.
CANONICAL_NAMES = {
    "apple": "Apple Inc.",
    "microsoft": "Microsoft Corp.",
    "google": "Alphabet Inc.",
    "alphabet": "Alphabet Inc.",
    "amazon": "Amazon.com Inc.",
    "meta platforms": "Meta Platforms Inc.",
    "facebook": "Meta Platforms Inc.",
    "tesla": "Tesla Inc.",
    "nvidia": "NVIDIA Corp.",
    "jpmorgan": "JPMorgan Chase & Co.",
    "jp morgan": "JPMorgan Chase & Co.",
    "j.p. morgan": "JPMorgan Chase & Co.",
    "goldman sachs": "Goldman Sachs Group Inc.",
    "bank of america": "Bank of America Corp.",
    "wells fargo": "Wells Fargo & Co.",
    "citigroup": "Citigroup Inc.",
    "citibank": "Citigroup Inc.",
    "berkshire hathaway": "Berkshire Hathaway Inc.",
    "johnson & johnson": "Johnson & Johnson",
    "procter & gamble": "Procter & Gamble Co.",
    "coca-cola": "The Coca-Cola Co.",
    "coca cola": "The Coca-Cola Co.",
    "pepsico": "PepsiCo Inc.",
    "walmart": "Walmart Inc.",
    "wal-mart": "Walmart Inc.",
    "walt disney": "The Walt Disney Co.",
    "disney": "The Walt Disney Co.",
    "netflix": "Netflix Inc.",
    "intel": "Intel Corp.",
    "ibm": "IBM Corp.",
    "oracle": "Oracle Corp.",
    "salesforce": "Salesforce Inc.",
    "adobe": "Adobe Inc.",
    "paypal": "PayPal Holdings Inc.",
    "visa": "Visa Inc.",
    "mastercard": "Mastercard Inc.",
    "chevron": "Chevron Corp.",
    "exxon mobil": "Exxon Mobil Corp.",
    "exxonmobil": "Exxon Mobil Corp.",
    "pfizer": "Pfizer Inc.",
    "merck": "Merck & Co.",
    "unitedhealth": "UnitedHealth Group Inc.",
    "at&t": "AT&T Inc.",
    "verizon": "Verizon Communications Inc.",
    "comcast": "Comcast Corp.",
    "home depot": "The Home Depot Inc.",
    "boeing": "The Boeing Co.",
    "lockheed martin": "Lockheed Martin Corp.",
    "caterpillar": "Caterpillar Inc.",
    "3m": "3M Co.",
    "general electric": "General Electric Co.",
    "ford motor": "Ford Motor Co.",
    "general motors": "General Motors Co.",
    "uber": "Uber Technologies Inc.",
    "lyft": "Lyft Inc.",
    "airbnb": "Airbnb Inc.",
    "snap inc": "Snap Inc.",
    "snapchat": "Snap Inc.",
    "twitter": "Twitter Inc.",
    "spotify": "Spotify Technology S.A.",
    "zoom video": "Zoom Video Communications Inc.",
    "shopify": "Shopify Inc.",
    "block inc": "Block Inc.",
    "coinbase": "Coinbase Global Inc.",
    "robinhood": "Robinhood Markets Inc.",
    "moderna": "Moderna Inc.",
    "biontech": "BioNTech SE",
    "abbvie": "AbbVie Inc.",
    "amgen": "Amgen Inc.",
    "gilead sciences": "Gilead Sciences Inc.",
    "eli lilly": "Eli Lilly and Co.",
    "bristol-myers": "Bristol-Myers Squibb Co.",
    "starbucks": "Starbucks Corp.",
    "mcdonald's": "McDonald's Corp.",
    "mcdonalds": "McDonald's Corp.",
    "nike": "Nike Inc.",
    "costco": "Costco Wholesale Corp.",
    "target": "Target Corp.",
    "lowe's": "Lowe's Companies Inc.",
    "kroger": "The Kroger Co.",
    "walgreens": "Walgreens Boots Alliance Inc.",
    "cvs health": "CVS Health Corp.",
    "united parcel": "United Parcel Service Inc.",
    "fedex": "FedEx Corp.",
    "delta air": "Delta Air Lines Inc.",
    "united airlines": "United Airlines Holdings Inc.",
    "american airlines": "American Airlines Group Inc.",
    "southwest airlines": "Southwest Airlines Co.",
    "t-mobile": "T-Mobile US Inc.",
    "marsh & mclennan": "Marsh & McLennan Companies Inc.",
    "marsh mclennan": "Marsh & McLennan Companies Inc.",
    "nextera energy": "NextEra Energy Inc.",
    "southern company": "Southern Company",
    "dominion energy": "Dominion Energy Inc.",
    "devon energy": "Devon Energy Corp.",
    "marathon petroleum": "Marathon Petroleum Corp.",
    "freeport-mcmoran": "Freeport-McMoRan Inc.",
    "freeport mcmoran": "Freeport-McMoRan Inc.",
    "d.r. horton": "D.R. Horton Inc.",
    "dr horton": "D.R. Horton Inc.",
    "lamb weston": "Lamb Weston Holdings Inc.",
    "fiserv": "Fiserv Inc.",
    "allegion": "Allegion PLC",
    "baxter international": "Baxter International Inc.",
    "conagra": "Conagra Brands Inc.",
    "generac": "Generac Holdings Inc.",
    "fastenal": "Fastenal Company",
    "autozone": "AutoZone Inc.",
    "aon": "Aon PLC",
    "blackstone": "Blackstone Inc.",
    "booking holdings": "Booking Holdings Inc.",
    "c.h. robinson": "C.H. Robinson Worldwide Inc.",
    "charter communications": "Charter Communications Inc.",
    "eastman chemical": "Eastman Chemical Co.",
    "henry schein": "Henry Schein Inc.",
    "hershey": "The Hershey Co.",
    "hormel foods": "Hormel Foods Corp.",
    "iff": "International Flavors & Fragrances Inc.",
    "insulet": "Insulet Corp.",
    "interpublic": "The Interpublic Group of Companies Inc.",
    "intuit": "Intuit Inc.",
    "itw": "Illinois Tool Works Inc.",
    "illinois tool works": "Illinois Tool Works Inc.",
    "jabil": "Jabil Inc.",
    "adp": "Automatic Data Processing Inc.",
    "kimberly-clark": "Kimberly-Clark Corp.",
    "kimberly clark": "Kimberly-Clark Corp.",
    "lam research": "Lam Research Corp.",
    "lululemon": "Lululemon Athletica Inc.",
    "lyondellbasell": "LyondellBasell Industries N.V.",
    "medtronic": "Medtronic PLC",
    "news corp": "News Corp.",
    "nisource": "NiSource Inc.",
    "nordson": "Nordson Corp.",
    "packaging corporation": "Packaging Corporation of America",
    "pool corporation": "Pool Corp.",
    "progressive": "The Progressive Corp.",
    "ptc": "PTC Inc.",
    "quanta services": "Quanta Services Inc.",
    "quest diagnostics": "Quest Diagnostics Inc.",
    "synchrony": "Synchrony Financial",
    "tapestry": "Tapestry Inc.",
    "thermo fisher": "Thermo Fisher Scientific Inc.",
    "tyler technologies": "Tyler Technologies Inc.",
    "union pacific": "Union Pacific Corp.",
    "united rentals": "United Rentals Inc.",
    "western digital": "Western Digital Corp.",
    "ebay": "eBay Inc.",
    "erie indemnity": "Erie Indemnity Co.",
    "fmc corporation": "FMC Corp.",
    "amphenol": "Amphenol Corp.",
    "laboratory corporation": "Laboratory Corp. of America Holdings",
    "labcorp": "Laboratory Corp. of America Holdings",
    "anthem": "Elevance Health Inc.",
    "elevance": "Elevance Health Inc.",
    "atmos energy": "Atmos Energy Corp.",
    "cme group": "CME Group Inc.",
    "unum": "Unum Group",
    "super micro": "Super Micro Computer Inc.",
    "supermicro": "Super Micro Computer Inc.",
    "state street": "State Street Corp.",
    "exelon": "Exelon Corp.",
    "firstenergy": "FirstEnergy Corp.",
    "first energy": "FirstEnergy Corp.",
    "lennar": "Lennar Corp.",
    "masco": "Masco Corp.",
    "molina": "Molina Healthcare Inc.",
    "paychex": "Paychex Inc.",
    "ralph lauren": "Ralph Lauren Corp.",
    "expeditors": "Expeditors International of Washington Inc.",
    "w.r. berkley": "W.R. Berkley Corp.",
    "w. r. berkley": "W.R. Berkley Corp.",
    "domino's": "Domino's Pizza Inc.",
    "dominos": "Domino's Pizza Inc.",
    "hewlett packard enterprise": "Hewlett Packard Enterprise Co.",
    "norfolk southern": "Norfolk Southern Corp.",
    "servicenow": "ServiceNow Inc.",
    "ross stores": "Ross Stores Inc.",
    "universal health": "Universal Health Services Inc.",
    "caesars entertainment": "Caesars Entertainment Inc.",
    "mettler-toledo": "Mettler-Toledo International Inc.",
    "mettler toledo": "Mettler-Toledo International Inc.",
    "pnc financial": "PNC Financial Services Group Inc.",
    "chipotle": "Chipotle Mexican Grill Inc.",
    "american water": "American Water Works Co. Inc.",
    "international paper": "International Paper Co.",
    "akamai": "Akamai Technologies Inc.",
    "parker-hannifin": "Parker-Hannifin Corp.",
    "parker hannifin": "Parker-Hannifin Corp.",
    "diamondback energy": "Diamondback Energy Inc.",
    "forest laboratories": "Forest Laboratories Inc.",
    "legg mason": "Legg Mason Inc.",
    "fossil group": "Fossil Group Inc.",
    "marketaxess": "MarketAxess Holdings Inc.",
    "metlife": "MetLife Inc.",
    "philip morris": "Philip Morris International Inc.",
    "humana": "Humana Inc.",
    "globe life": "Globe Life Inc.",
    "republic services": "Republic Services Inc.",
    "textron": "Textron Inc.",
    "hilton": "Hilton Worldwide Holdings Inc.",
    "samsung": "Samsung Electronics Co.",
    "volvo": "Volvo Group",
    "marathon oil": "Marathon Oil Corp.",
    "conocophillips": "ConocoPhillips",
    "suncoke": "SunCoke Energy Inc.",
    "ppl corp": "PPL Corp.",
    "ppl": "PPL Corp.",
    "westar energy": "Evergy Inc.",
    "constellation": "Constellation Energy Corp.",
}


def _match_canonical(text_lower: str) -> str | None:
    """Check text against canonical names using word-boundary matching."""
    # Sort by key length descending so longer matches win
    for key in sorted(CANONICAL_NAMES.keys(), key=len, reverse=True):
        # Build a regex with word boundaries
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, text_lower):
            return CANONICAL_NAMES[key]
    return None


# ── Positive / Negative word lists (financial domain) ───────────────────────
POSITIVE_WORDS = {
    "growth", "grew", "growing", "increase", "increased", "increasing",
    "strong", "strength", "strengthen", "strengthened", "solid",
    "improve", "improved", "improvement", "improving",
    "record", "exceeded", "exceeding", "exceed", "beat",
    "profit", "profitable", "profitability",
    "gain", "gains", "gained",
    "positive", "positively",
    "expansion", "expanded", "expanding", "expand",
    "outperform", "outperformed", "outperforming",
    "momentum", "accelerating", "accelerated",
    "robust", "resilient", "resilience",
    "favorable", "favourable",
    "upside", "uptick", "upturn",
    "success", "successful", "successfully",
    "optimistic", "optimism", "confident", "confidence",
    "higher", "highest",
    "benefit", "benefits", "benefited", "benefiting",
    "opportunity", "opportunities",
    "innovation", "innovative",
    "efficient", "efficiency", "efficiencies",
    "dividend", "dividends",
    "delivered", "delivering", "deliver",
    "upgrade", "upgraded",
    "synergies", "synergy",
    "transformative",
}

NEGATIVE_WORDS = {
    "decline", "declined", "declining",
    "decrease", "decreased", "decreasing",
    "loss", "losses", "lost",
    "weak", "weakness", "weakened", "weakening",
    "risk", "risks", "risky",
    "challenge", "challenges", "challenging",
    "headwind", "headwinds",
    "pressure", "pressures", "pressured",
    "downturn", "downside",
    "negative", "negatively",
    "deteriorating", "deteriorated", "deterioration",
    "impaired", "impairment",
    "restructuring",
    "litigation", "lawsuit", "lawsuits",
    "deficit", "deficits",
    "concern", "concerns", "concerned",
    "volatility", "volatile",
    "uncertainty", "uncertain",
    "lower", "lowest",
    "underperform", "underperformed",
    "missed",
    "layoff", "layoffs",
    "closure", "closures",
    "downgrade", "downgraded",
    "unfavorable", "unfavourable",
    "inflation", "inflationary",
    "recession", "recessionary",
    "disruption", "disruptions",
    "shortage", "shortages",
    "adverse", "adversely",
    "penalty", "penalties",
    "reduced", "reducing", "reduction",
    "contraction", "contracted",
}

# Strongly positive / negative phrases carry extra weight
STRONG_POS = {
    "record revenue", "record earnings", "record profit", "all-time high",
    "exceeded expectations", "above expectations", "beat expectations",
    "raised guidance", "raising guidance", "increased guidance",
    "strong growth", "robust growth", "accelerating growth",
    "margin expansion", "significant improvement", "outstanding results",
    "double-digit growth", "triple-digit growth",
    "best quarter", "best year", "best performance",
}

STRONG_NEG = {
    "net loss", "operating loss", "significant decline",
    "material weakness", "going concern",
    "missed expectations", "below expectations",
    "lowered guidance", "reduced guidance", "cut guidance",
    "significant headwinds", "severe challenges",
    "substantial risk", "material risk",
    "goodwill impairment", "asset impairment",
    "credit downgrade", "debt downgrade",
    "bankruptcy", "chapter 11", "insolvency",
    "fraud", "sec investigation",
    "class action", "shareholder lawsuit",
    "revenue decline", "earnings decline", "profit decline",
}

# Words that should NOT be treated as company entities
ENTITY_BLACKLIST = {
    "company", "corporation", "group", "financial", "consolidated financial",
    "management", "discussion and analysis", "notes to consolidated financial",
    "notes to the consolidated financial", "liquidity and capital resources",
    "non-gaap financial", "management's discussion",
    "chief financial", "investor relations", "our", "second", "first",
    "third", "fourth", "the", "income and other", "results of operations",
    "consolidated", "operating", "total", "net", "overview",
    "s discussion and analysis", "executive", "item",
}


def _clean_entity_name(name: str) -> str:
    """Clean and title-case an entity name. Return empty string if invalid."""
    name = name.strip().rstrip(",.:;'\"")
    name = re.sub(r"^(?:the\s+)", "", name, flags=re.IGNORECASE)
    name = name.strip()

    # Reject if too short or too long
    if len(name) < 3 or len(name) > 45:
        return ""

    # Reject if it's a blacklisted generic term
    name_lower = name.lower()
    if name_lower.rstrip("s") in ENTITY_BLACKLIST or name_lower in ENTITY_BLACKLIST:
        return ""

    # Reject phrases that start with common non-entity words
    bad_starts = [
        "discussion", "management", "notes to", "consolidated",
        "non-gaap", "liquidity", "results of", "income",
        "chief", "investor", "executive", "overview",
        "item ", "report", "second quarter", "third quarter",
        "first quarter", "fourth quarter", "today's", "today",
        "joining", "participating", "our ", "we ", "a ",
        "in addition", "the portion", "sure.", "well,",
        "okay", "thanks", "yes", "no,", "so,", "and ",
    ]
    for bs in bad_starts:
        if name_lower.startswith(bs):
            return ""

    # Reject if it starts with a lowercase word (likely a sentence fragment)
    if name[0].islower():
        return ""

    # Reject if it contains too many words (likely a sentence fragment)
    word_count = len(name.split())
    if word_count > 6:
        return ""

    # Reject if it looks like a person's name (analyst) rather than company
    # Pattern: "FirstName LastName" with no corporate suffix
    if (word_count == 2
        and not re.search(r"(?:Inc|Corp|Co|Ltd|LLC|PLC|Group|Holdings|Bank|Energy|Capital|Financial|Services|Systems|Technologies|Partners|Trust|Fund|Insurance|Brands|Foods|Industries|Resources|Realty|Properties|Communications|Networks|Solutions|Enterprises|International|Therapeutics|Pharmaceuticals|Biosciences)", name)
        and re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name)):
        return ""

    return name


def extract_entity(text: str) -> str:
    """Extract the primary financial entity from the text."""
    snippet_lower = text[:1500].lower()

    # ── Step 1: Try canonical name match first (most reliable) ──────────
    # Check in first 1500 chars
    canonical = _match_canonical(snippet_lower)
    if canonical:
        return canonical

    # ── Step 2: Earnings call patterns ──────────────────────────────────

    # Pattern: "Welcome to [the] [Company] ... Earnings/Conference Call"
    ec_match = re.search(
        r"welcome to (?:the\s+)?(.+?)\s+(?:\d{4}\s+)?(?:first|second|third|fourth|q[1-4]|full[- ]?year|annual|fiscal)?\s*(?:quarter|year)?\s*\d{0,4}\s*(?:earnings|quarterly|financial|results|conference|investor)",
        snippet_lower,
    )
    if ec_match:
        raw = ec_match.group(1).strip()
        # Try canonical on the extracted name
        c = _match_canonical(raw)
        if c:
            return c
        # Title case it
        cleaned = _clean_entity_name(raw.title())
        if cleaned:
            return _finalize_entity(cleaned, text)

    # Pattern: "turn the conference/call over to ... for/of/at [Company]"
    for_match = re.search(
        r"(?:turn the (?:conference|call) over to .+?(?:for|of|at)\s+)([A-Z][A-Za-z\s&\.\-\']+?)[\.\,\n]",
        text[:1500],
    )
    if for_match:
        raw = for_match.group(1).strip()
        c = _match_canonical(raw.lower())
        if c:
            return c
        cleaned = _clean_entity_name(raw)
        if cleaned:
            return _finalize_entity(cleaned, text)

    # Pattern: "joining [us for/on] [Company]'s ... call"
    join_match = re.search(
        r"(?:joining\s+(?:us\s+)?(?:for|on)\s+)([A-Za-z][A-Za-z\s&\.\-\']+?)(?:'s|'s)\s+",
        text[:1500],
        re.IGNORECASE,
    )
    if join_match:
        raw = join_match.group(1).strip()
        c = _match_canonical(raw.lower())
        if c:
            return c
        cleaned = _clean_entity_name(raw.title() if raw[0].islower() else raw)
        if cleaned:
            return _finalize_entity(cleaned, text)

    # Pattern: "welcome to [Company]'s" (possessive form)
    poss_match = re.search(
        r"welcome to (?:the\s+)?([A-Za-z][A-Za-z\s&\.\-\']+?)(?:'s|'s)\s+",
        text[:1500],
        re.IGNORECASE,
    )
    if poss_match:
        raw = poss_match.group(1).strip()
        c = _match_canonical(raw.lower())
        if c:
            return c
        cleaned = _clean_entity_name(raw.title() if raw[0].islower() else raw)
        if cleaned:
            return _finalize_entity(cleaned, text)

    # Pattern: "today's [Company] conference call"
    todays_match = re.search(
        r"today'?s\s+([A-Za-z][A-Za-z\s&\.\-\']+?)\s+(?:conference|earnings|quarterly|investor|call)",
        text[:1500],
        re.IGNORECASE,
    )
    if todays_match:
        raw = todays_match.group(1).strip()
        c = _match_canonical(raw.lower())
        if c:
            return c
        cleaned = _clean_entity_name(raw.title() if raw[0].islower() else raw)
        if cleaned:
            return _finalize_entity(cleaned, text)

    # ── Step 3: SEC / 10-K filing patterns ──────────────────────────────

    # Pattern: "[COMPANY NAME], Inc." or similar with legal suffix
    corp_match = re.search(
        r"([A-Z][A-Za-z\s&\.\-\',]{2,40}?)\s*(?:,\s*)?(?:Inc\.|Corp\.|Co\.|Ltd\.|LLC|L\.P\.|plc|PLC|N\.V\.)",
        text[:3000],
    )
    if corp_match:
        raw = corp_match.group(0).strip()
        c = _match_canonical(raw.lower())
        if c:
            return c
        cleaned = _clean_entity_name(raw)
        if cleaned:
            return cleaned

    # ── Step 4: "[Company] reported / announced" ────────────────────────
    report_match = re.search(
        r"([A-Z][A-Za-z\s&\.\-\']{2,40}?)\s+(?:reported|announced|delivered|posted|generated|recorded)\s",
        text[:2000],
    )
    if report_match:
        raw = report_match.group(1).strip()
        c = _match_canonical(raw.lower())
        if c:
            return c
        cleaned = _clean_entity_name(raw)
        if cleaned:
            return _finalize_entity(cleaned, text)

    # ── Step 5: Broader canonical check in wider text ───────────────────
    canonical_wide = _match_canonical(text[:3000].lower())
    if canonical_wide:
        return canonical_wide

    # ── Step 6: Fallback - generic market discussion ────────────────────
    market_words = ["market", "economy", "sector", "industry", "index",
                    "s&p", "dow jones", "nasdaq", "federal reserve"]
    for w in market_words:
        if w in snippet_lower:
            return "MARKET"

    return "NONE"


def _finalize_entity(name: str, full_text: str) -> str:
    """Try to find the fullest legal version of a name in the text."""
    # Try to find "Name, Inc." or "Name Corp." in the text
    escaped = re.escape(name.rstrip(",. "))
    fuller = re.search(
        rf"({escaped}[\w\s\-&,]*?\s*(?:Inc\.|Corp\.|Co\.|Ltd\.|LLC|Company|Corporation|Holdings|L\.P\.|PLC|plc|N\.V\.))",
        full_text[:5000],
        re.IGNORECASE,
    )
    if fuller:
        result = fuller.group(1).strip().rstrip(",")
        if len(result) < 60:
            return result

    # Return title-cased name
    return name


def score_sentiment(text: str) -> str:
    """Score overall sentiment of a financial text passage."""
    text_lower = text.lower()
    # Sample beginning and end (guidance/conclusions often at end)
    sample = text_lower[:3000] + " " + text_lower[-1500:]

    pos_count = 0
    neg_count = 0

    # Count strong phrases (worth 3 points each)
    for phrase in STRONG_POS:
        if phrase in sample:
            pos_count += 3

    for phrase in STRONG_NEG:
        if phrase in sample:
            neg_count += 3

    # Count individual words
    words = re.findall(r'\b\w+\b', sample)
    for word in words:
        if word in POSITIVE_WORDS:
            pos_count += 1
        elif word in NEGATIVE_WORDS:
            neg_count += 1

    # Decision thresholds
    total = pos_count + neg_count
    if total == 0:
        return "NEUTRAL"

    ratio = pos_count / total

    if ratio > 0.62:
        return "POSITIVE"
    elif ratio < 0.38:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def process_row(text: str) -> dict:
    """Process a single text row and return annotations."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"label": "NEUTRAL", "entity": "NONE", "entity_sentiment": "NEUTRAL"}

    entity = extract_entity(text)
    overall_sentiment = score_sentiment(text)

    # Entity sentiment matches overall for earnings calls / 10-K (entity = subject)
    entity_sentiment = overall_sentiment

    return {
        "label": overall_sentiment,
        "entity": entity,
        "entity_sentiment": entity_sentiment,
    }


def process_batch(batch_num: int) -> None:
    """Process a single batch file."""
    input_path = os.path.join(DATA_DIR, f"batch_{batch_num:04d}_input.csv")
    output_path = os.path.join(DATA_DIR, f"batch_{batch_num:04d}_output.csv")

    if not os.path.exists(input_path):
        print(f"  WARNING: {input_path} not found, skipping.")
        return

    df = pd.read_csv(input_path)

    # Get the index column
    if "Unnamed: 0" in df.columns:
        idx_col = df["Unnamed: 0"]
    else:
        idx_col = df.index

    results = []
    for i, row in df.iterrows():
        text = row["text"]
        annotations = process_row(text)
        results.append(annotations)

    out_df = pd.DataFrame(results)
    out_df.insert(0, "", idx_col.values)

    out_df.to_csv(output_path, index=False)


def main():
    print(f"Processing 45 batch files in {DATA_DIR}")
    print("=" * 60)

    for batch_num in range(45):
        print(f"Processing batch_{batch_num:04d}...", end=" ")
        process_batch(batch_num)
        # Verify output
        output_path = os.path.join(DATA_DIR, f"batch_{batch_num:04d}_output.csv")
        if os.path.exists(output_path):
            out_df = pd.read_csv(output_path)
            label_counts = out_df["label"].value_counts().to_dict()
            entity_sample = out_df["entity"].iloc[0] if len(out_df) > 0 else "N/A"
            print(f"OK ({len(out_df)} rows) | {label_counts} | e.g. entity: {entity_sample}")
        else:
            print("FAILED")

    print("=" * 60)
    print("Done. All 45 batches processed.")


if __name__ == "__main__":
    main()
