"""
Annotate long financial texts (batches 0135-0179) for:
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
}

# ── Company name patterns ────────────────────────────────────────────────
# Patterns to find company names in opening text
COMPANY_PATTERNS = [
    # Earnings call: "welcome to X's Q1 2022 Earnings"
    re.compile(r"welcome to (?:the )?(.+?)(?:'s|'s)?\s+(?:Q[1-4]|first|second|third|fourth|FY|fiscal|full[- ]year|annual|quarterly|\d{4})", re.IGNORECASE),
    # "welcome to X earnings"
    re.compile(r"welcome to (?:the )?(.+?)\s+(?:earnings|conference|investor|quarterly|annual)", re.IGNORECASE),
    # "This is the X conference call"
    re.compile(r"(?:this is|joining us from)\s+(?:the\s+)?(.+?)\s+(?:conference|earnings|call|investor)", re.IGNORECASE),
    # "X reported" or "X announced" (require Corp/Inc-like suffix or multi-word cap name)
    re.compile(r"(?:^|\.\s+)([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\s+(?:reported|announced|delivered|posted|achieved|recorded|generated)\b", re.IGNORECASE),
    # Operator intro: "host, Mr/Ms. Name, ... of CompanyName"
    re.compile(r"(?:of|from|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\s*\.\s+(?:Thank|Please|Go ahead)", re.IGNORECASE),
    # "On behalf of X"
    re.compile(r"[Oo]n behalf of\s+(.+?)(?:,|\s+and\s+|\s+I\s+)", re.IGNORECASE),
]

# Known company name mappings for canonical forms
# IMPORTANT: keys that are short/ambiguous MUST use word-boundary matching.
# We split into two groups: short keys needing \b matching, and longer safe keys.
CANONICAL_NAMES_LONG = {
    "apple": "Apple Inc.",
    "microsoft": "Microsoft Corporation",
    "google": "Alphabet Inc.",
    "alphabet": "Alphabet Inc.",
    "amazon": "Amazon.com Inc.",
    "meta platforms": "Meta Platforms Inc.",
    "facebook": "Meta Platforms Inc.",
    "tesla": "Tesla Inc.",
    "nvidia": "NVIDIA Corporation",
    "ford motor": "Ford Motor Company",
    "general motors": "General Motors Company",
    "jpmorgan": "JPMorgan Chase & Co.",
    "jp morgan": "JPMorgan Chase & Co.",
    "goldman sachs": "Goldman Sachs Group Inc.",
    "bank of america": "Bank of America Corporation",
    "wells fargo": "Wells Fargo & Company",
    "citigroup": "Citigroup Inc.",
    "morgan stanley": "Morgan Stanley",
    "johnson & johnson": "Johnson & Johnson",
    "pfizer": "Pfizer Inc.",
    "unitedhealth": "UnitedHealth Group Inc.",
    "walmart": "Walmart Inc.",
    "wal-mart": "Walmart Inc.",
    "home depot": "The Home Depot Inc.",
    "chevron": "Chevron Corporation",
    "exxon mobil": "Exxon Mobil Corporation",
    "exxonmobil": "Exxon Mobil Corporation",
    "valero": "Valero Energy Corporation",
    "conocophillips": "ConocoPhillips",
    "procter & gamble": "Procter & Gamble Company",
    "coca-cola": "The Coca-Cola Company",
    "coca cola": "The Coca-Cola Company",
    "pepsico": "PepsiCo Inc.",
    "intel": "Intel Corporation",
    "cisco": "Cisco Systems Inc.",
    "oracle": "Oracle Corporation",
    "salesforce": "Salesforce Inc.",
    "adobe": "Adobe Inc.",
    "netflix": "Netflix Inc.",
    "disney": "The Walt Disney Company",
    "walt disney": "The Walt Disney Company",
    "comcast": "Comcast Corporation",
    "verizon": "Verizon Communications Inc.",
    "t-mobile": "T-Mobile US Inc.",
    "boeing": "The Boeing Company",
    "lockheed martin": "Lockheed Martin Corporation",
    "raytheon": "Raytheon Technologies Corporation",
    "caterpillar": "Caterpillar Inc.",
    "deere": "Deere & Company",
    "john deere": "Deere & Company",
    "honeywell": "Honeywell International Inc.",
    "general electric": "General Electric Company",
    "fedex": "FedEx Corporation",
    "berkshire hathaway": "Berkshire Hathaway Inc.",
    "mastercard": "Mastercard Incorporated",
    "paypal": "PayPal Holdings Inc.",
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
    "costco": "Costco Wholesale Corporation",
    "target": "Target Corporation",
    "qualcomm": "Qualcomm Incorporated",
    "broadcom": "Broadcom Inc.",
    "texas instruments": "Texas Instruments Incorporated",
    "advanced micro": "Advanced Micro Devices Inc.",
    "applied materials": "Applied Materials Inc.",
    "micron": "Micron Technology Inc.",
    "shopify": "Shopify Inc.",
    "snowflake": "Snowflake Inc.",
    "palantir": "Palantir Technologies Inc.",
    "crowdstrike": "CrowdStrike Holdings Inc.",
    "datadog": "Datadog Inc.",
    "twilio": "Twilio Inc.",
    "willis towers watson": "Willis Towers Watson",
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
    "capital one": "Capital One Financial Corporation",
    "american express": "American Express Company",
    "discover financial": "Discover Financial Services",
    "synchrony": "Synchrony Financial",
    "ally financial": "Ally Financial Inc.",
    "regions financial": "Regions Financial Corporation",
    "keycorp": "KeyCorp",
    "fifth third": "Fifth Third Bancorp",
    "huntington bancshares": "Huntington Bancshares Incorporated",
    "comerica": "Comerica Incorporated",
    "silicon valley bank": "SVB Financial Group",
    "first republic": "First Republic Bank",
    "truist": "Truist Financial Corporation",
    "u.s. bancorp": "U.S. Bancorp",
    "us bancorp": "U.S. Bancorp",
    "pnc financial": "PNC Financial Services Group Inc.",
    "at&t": "AT&T Inc.",
    "uber": "Uber Technologies Inc.",
    "lyft": "Lyft Inc.",
    "airbnb": "Airbnb Inc.",
    "doordash": "DoorDash Inc.",
    "snap inc": "Snap Inc.",
    "twitter": "Twitter Inc.",
    "zoom video": "Zoom Video Communications Inc.",
    "palo alto networks": "Palo Alto Networks Inc.",
    "fortinet": "Fortinet Inc.",
    "zscaler": "Zscaler Inc.",
    "okta": "Okta Inc.",
    "visa": "Visa Inc.",
    "block inc": "Block Inc.",
    "square inc": "Block Inc.",
    "berkshire": "Berkshire Hathaway Inc.",
    "moody's": "Moody's Corporation",
    "moodys": "Moody's Corporation",
    "moody": "Moody's Corporation",
    "cigna": "Cigna Corporation",
    "anthem": "Anthem Inc.",
    "aetna": "Aetna Inc.",
    "humana": "Humana Inc.",
    "centene": "Centene Corporation",
    "molina": "Molina Healthcare Inc.",
    "cvs health": "CVS Health Corporation",
    "walgreens": "Walgreens Boots Alliance Inc.",
    "kroger": "The Kroger Company",
    "albertsons": "Albertsons Companies Inc.",
    "sysco": "Sysco Corporation",
    "tyson": "Tyson Foods Inc.",
    "mondelez": "Mondelez International Inc.",
    "kraft heinz": "The Kraft Heinz Company",
    "general mills": "General Mills Inc.",
    "kellogg": "Kellogg Company",
    "colgate": "Colgate-Palmolive Company",
    "kimberly-clark": "Kimberly-Clark Corporation",
    "church & dwight": "Church & Dwight Co. Inc.",
    "clorox": "The Clorox Company",
    "estee lauder": "The Estee Lauder Companies Inc.",
    "ralph lauren": "Ralph Lauren Corporation",
    "tapestry": "Tapestry Inc.",
    "hasbro": "Hasbro Inc.",
    "mattel": "Mattel Inc.",
    "activision": "Activision Blizzard Inc.",
    "electronic arts": "Electronic Arts Inc.",
    "take-two": "Take-Two Interactive Software Inc.",
    "roblox": "Roblox Corporation",
    "spotify": "Spotify Technology S.A.",
    "roku": "Roku Inc.",
    "paramount": "Paramount Global",
    "warner bros": "Warner Bros. Discovery Inc.",
    "fox corporation": "Fox Corporation",
    "nextera": "NextEra Energy Inc.",
    "duke energy": "Duke Energy Corporation",
    "southern company": "The Southern Company",
    "dominion": "Dominion Energy Inc.",
    "exelon": "Exelon Corporation",
    "entergy": "Entergy Corporation",
    "firstenergy": "FirstEnergy Corp.",
    "consolidated edison": "Consolidated Edison Inc.",
    "american electric": "American Electric Power Company Inc.",
    "xcel energy": "Xcel Energy Inc.",
    "sempra": "Sempra Energy",
    "centerpoint": "CenterPoint Energy Inc.",
    "cms energy": "CMS Energy Corporation",
    "dte energy": "DTE Energy Company",
    "atmos energy": "Atmos Energy Corporation",
    "nisource": "NiSource Inc.",
    "baker hughes": "Baker Hughes Company",
    "schlumberger": "Schlumberger Limited",
    "halliburton": "Halliburton Company",
    "marathon": "Marathon Petroleum Corporation",
    "phillips 66": "Phillips 66",
    "hess": "Hess Corporation",
    "devon energy": "Devon Energy Corporation",
    "diamondback": "Diamondback Energy Inc.",
    "pioneer natural": "Pioneer Natural Resources Company",
    "coterra": "Coterra Energy Inc.",
    "eog resources": "EOG Resources Inc.",
    "occidental": "Occidental Petroleum Corporation",
    "williams companies": "The Williams Companies Inc.",
    "kinder morgan": "Kinder Morgan Inc.",
    "oneok": "ONEOK Inc.",
    "enterprise products": "Enterprise Products Partners L.P.",
    "simon property": "Simon Property Group Inc.",
    "prologis": "Prologis Inc.",
    "american tower": "American Tower Corporation",
    "crown castle": "Crown Castle International Corp.",
    "equinix": "Equinix Inc.",
    "digital realty": "Digital Realty Trust Inc.",
    "public storage": "Public Storage",
    "realty income": "Realty Income Corporation",
    "weyerhaeuser": "Weyerhaeuser Company",
    "boston properties": "Boston Properties Inc.",
    "vornado": "Vornado Realty Trust",
    "macy": "Macy's Inc.",
    "nordstrom": "Nordstrom Inc.",
    "ross stores": "Ross Stores Inc.",
    "dollar general": "Dollar General Corporation",
    "dollar tree": "Dollar Tree Inc.",
    "autozone": "AutoZone Inc.",
    "o'reilly": "O'Reilly Automotive Inc.",
    "advance auto": "Advance Auto Parts Inc.",
    "tractor supply": "Tractor Supply Company",
    "bath & body": "Bath & Body Works Inc.",
    "gap inc": "Gap Inc.",
    "lululemon": "Lululemon Athletica Inc.",
    "under armour": "Under Armour Inc.",
    "foot locker": "Foot Locker Inc.",
    "dick's sporting": "Dick's Sporting Goods Inc.",
    "best buy": "Best Buy Co. Inc.",
    "gamestop": "GameStop Corp.",
    "wayfair": "Wayfair Inc.",
    "chewy": "Chewy Inc.",
    "etsy": "Etsy Inc.",
    "ebay": "eBay Inc.",
    "carvana": "Carvana Co.",
    "carmax": "CarMax Inc.",
    "autonation": "AutoNation Inc.",
    "lithia motors": "Lithia Motors Inc.",
    "emerson electric": "Emerson Electric Co.",
    "illinois tool": "Illinois Tool Works Inc.",
    "parker hannifin": "Parker-Hannifin Corporation",
    "eaton": "Eaton Corporation",
    "dover": "Dover Corporation",
    "roper technologies": "Roper Technologies Inc.",
    "rockwell": "Rockwell Automation Inc.",
    "nordson": "Nordson Corporation",
    "xylem": "Xylem Inc.",
    "waste management": "Waste Management Inc.",
    "republic services": "Republic Services Inc.",
    "cintas": "Cintas Corporation",
    "copart": "Copart Inc.",
    "fastenal": "Fastenal Company",
    "verisk": "Verisk Analytics Inc.",
    "factset": "FactSet Research Systems Inc.",
    "msci": "MSCI Inc.",
    "s&p global": "S&P Global Inc.",
    "intercontinental exchange": "Intercontinental Exchange Inc.",
    "cme group": "CME Group Inc.",
    "cboe": "Cboe Global Markets Inc.",
    "nasdaq": "Nasdaq Inc.",
    "fidelity national": "Fidelity National Information Services Inc.",
    "fis": "Fidelity National Information Services Inc.",
    "fiserv": "Fiserv Inc.",
    "jack henry": "Jack Henry & Associates Inc.",
    "global payments": "Global Payments Inc.",
    "fleetcor": "FLEETCOR Technologies Inc.",
    "wex inc": "WEX Inc.",
    "accenture": "Accenture plc",
    "cognizant": "Cognizant Technology Solutions Corporation",
    "infosys": "Infosys Limited",
    "wipro": "Wipro Limited",
    "automatic data": "Automatic Data Processing Inc.",
    "paychex": "Paychex Inc.",
    "paycom": "Paycom Software Inc.",
    "paylocity": "Paylocity Holding Corporation",
    "workday": "Workday Inc.",
    "servicenow": "ServiceNow Inc.",
    "splunk": "Splunk Inc.",
    "dynatrace": "Dynatrace Inc.",
    "elastic": "Elastic N.V.",
    "mongodb": "MongoDB Inc.",
    "confluent": "Confluent Inc.",
    "hashicorp": "HashiCorp Inc.",
    "cloudflare": "Cloudflare Inc.",
    "fastly": "Fastly Inc.",
    "akamai": "Akamai Technologies Inc.",
    "arista": "Arista Networks Inc.",
    "juniper": "Juniper Networks Inc.",
    "motorola": "Motorola Solutions Inc.",
    "zebra technologies": "Zebra Technologies Corporation",
    "trimble": "Trimble Inc.",
    "garmin": "Garmin Ltd.",
    "keysight": "Keysight Technologies Inc.",
    "teledyne": "Teledyne Technologies Incorporated",
    "ametek": "AMETEK Inc.",
    "mettler-toledo": "Mettler-Toledo International Inc.",
    "waters corporation": "Waters Corporation",
    "agilent": "Agilent Technologies Inc.",
    "danaher": "Danaher Corporation",
    "thermo fisher": "Thermo Fisher Scientific Inc.",
    "illumina": "Illumina Inc.",
    "regeneron": "Regeneron Pharmaceuticals Inc.",
    "vertex": "Vertex Pharmaceuticals Incorporated",
    "biogen": "Biogen Inc.",
    "moderna": "Moderna Inc.",
    "biontech": "BioNTech SE",
    "intuitive surgical": "Intuitive Surgical Inc.",
    "stryker": "Stryker Corporation",
    "medtronic": "Medtronic plc",
    "boston scientific": "Boston Scientific Corporation",
    "edwards lifesciences": "Edwards Lifesciences Corporation",
    "becton dickinson": "Becton Dickinson and Company",
    "baxter": "Baxter International Inc.",
    "zimmer biomet": "Zimmer Biomet Holdings Inc.",
    "hologic": "Hologic Inc.",
    "dexcom": "DexCom Inc.",
    "insulet": "Insulet Corporation",
    "resmed": "ResMed Inc.",
    "align technology": "Align Technology Inc.",
    "cooper companies": "The Cooper Companies Inc.",
    "teleflex": "Teleflex Incorporated",
    "globus medical": "Globus Medical Inc.",
    "nuvasive": "NuVasive Inc.",
    "iqvia": "IQVIA Holdings Inc.",
    "charles river": "Charles River Laboratories International Inc.",
    "west pharmaceutical": "West Pharmaceutical Services Inc.",
    "bio-rad": "Bio-Rad Laboratories Inc.",
    "repligen": "Repligen Corporation",
    "azenta": "Azenta Inc.",
    "veeva systems": "Veeva Systems Inc.",
    "certara": "Certara Inc.",
    "hca healthcare": "HCA Healthcare Inc.",
    "tenet healthcare": "Tenet Healthcare Corporation",
    "universal health": "Universal Health Services Inc.",
    "community health": "Community Health Systems Inc.",
    "encompass health": "Encompass Health Corporation",
    "acadia healthcare": "Acadia Healthcare Company Inc.",
    "labcorp": "Laboratory Corporation of America Holdings",
    "quest diagnostics": "Quest Diagnostics Incorporated",
    "amedisys": "Amedisys Inc.",
    "addus homecare": "Addus HomeCare Corporation",
    "lhc group": "LHC Group Inc.",
    "chemed": "Chemed Corporation",
    "brookdale": "Brookdale Senior Living Inc.",
}

# Short/ambiguous keys that need word-boundary matching
CANONICAL_NAMES_SHORT = {
    "ford": "Ford Motor Company",
    "gm": "General Motors Company",
    "citi": "Citigroup Inc.",
    "exxon": "Exxon Mobil Corporation",
    "pepsi": "PepsiCo Inc.",
    "ibm": "IBM Corporation",
    "nike": "Nike Inc.",
    "amd": "Advanced Micro Devices Inc.",
    "ups": "United Parcel Service Inc.",
    "3m": "3M Company",
    "ge": "General Electric Company",
    "ally": "Ally Financial Inc.",
    "meta": "Meta Platforms Inc.",
    "conoco": "ConocoPhillips",
    "p&g": "Procter & Gamble Company",
    "j&j": "Johnson & Johnson",
    "amex": "American Express Company",
    "svb": "SVB Financial Group",
    "pnc": "PNC Financial Services Group Inc.",
    "wtw": "Willis Towers Watson",
    "aon": "Aon plc",
    "key": "KeyCorp",
    "snap": "Snap Inc.",
    "hess": "Hess Corporation",
    "fis": "Fidelity National Information Services Inc.",
}


def find_canonical_in_text(text_lower, search_range=1000):
    """Search for known company names in text with proper word boundaries."""
    snippet = text_lower[:search_range]
    best_match = None
    best_pos = len(snippet)

    # Long keys: safe to use substring matching (they're unambiguous)
    for key, canonical in CANONICAL_NAMES_LONG.items():
        pos = snippet.find(key)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_match = canonical

    # Short keys: require word boundary matching
    for key, canonical in CANONICAL_NAMES_SHORT.items():
        pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
        m = pattern.search(snippet)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best_match = canonical

    return best_match


def count_sentiment_words(text_lower):
    """Count positive and negative sentiment words in text."""
    all_words = re.findall(r'\b[a-z][\w-]*\b', text_lower)
    pos_freq = sum(1 for w in all_words if w in POS_WORDS)
    neg_freq = sum(1 for w in all_words if w in NEG_WORDS)
    return pos_freq, neg_freq


def classify_sentiment(text):
    """Classify overall text sentiment using keyword analysis."""
    text_lower = text.lower()
    pos_freq, neg_freq = count_sentiment_words(text_lower)

    total_freq = pos_freq + neg_freq
    if total_freq == 0:
        return "NEUTRAL"

    pos_ratio = pos_freq / total_freq

    if pos_ratio > 0.62:
        return "POSITIVE"
    elif pos_ratio < 0.38:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def extract_entity_from_text(text, first_500):
    """Extract the primary financial entity from the text."""
    text_lower = text.lower()

    # 1) Try regex patterns on the first 500 chars (earnings call intros, etc.)
    for pattern in COMPANY_PATTERNS:
        match = pattern.search(first_500)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s+', ' ', name)
            name = name.strip(' ,.')
            # Skip generic / too-short / section-header-like matches
            if len(name) < 3 or len(name) > 80:
                continue
            if name.lower() in {'the', 'our', 'this', 'we', 'a', 'an', 'item', 'mr', 'ms', 'mrs', 'dr'}:
                continue
            # Skip all-caps section headers (LIQUIDITY AND CAPITAL RESOURCES, etc.)
            if re.match(r'^[A-Z\s&,]+$', name) and len(name) > 10:
                continue
            return canonicalize_name(name, text_lower)

    # 2) Look for known company names in first 1500 chars with proper matching
    found = find_canonical_in_text(text_lower, search_range=1500)
    if found:
        return found

    # 3) Look for "Corporation", "Company", "Inc." patterns in first 2000 chars
    corp_pattern = re.compile(
        r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+'
        r'(?:Corporation|Company|Inc\.|Corp\.|Ltd\.|Holdings|Incorporated|Group)',
    )
    match = corp_pattern.search(text[:2000])
    if match:
        name = match.group(0).strip()
        if len(name) > 3:
            return canonicalize_name(name, text_lower)

    # 4) Wider search for known names (first 5000 chars)
    found = find_canonical_in_text(text_lower, search_range=5000)
    if found:
        return found

    return "MARKET"


def canonicalize_name(name, text_lower):
    """Convert extracted name to canonical form."""
    name_clean = name.strip(' ,.')
    name_lower = name_clean.lower()

    # Remove common prefixes
    for prefix in ['the ', 'a ']:
        if name_lower.startswith(prefix):
            name_clean = name_clean[len(prefix):]
            name_lower = name_clean.lower()

    # Check long canonical mapping (substring is fine here since we're matching the extracted name)
    for key, canonical in CANONICAL_NAMES_LONG.items():
        if key in name_lower or name_lower in key:
            return canonical

    # Check short canonical mapping with exact match
    for key, canonical in CANONICAL_NAMES_SHORT.items():
        if key == name_lower:
            return canonical

    # If name is a person's name (1-2 words, no corporate suffix), try to find company elsewhere
    if len(name_clean.split()) <= 2 and not any(s in name_clean for s in ['Inc', 'Corp', 'Co.', 'Ltd', 'LLC', 'Company', 'Group']):
        # Search wider text for a known company
        found = find_canonical_in_text(text_lower, search_range=3000)
        if found:
            return found

    if len(name_clean) < 3:
        return "MARKET"

    return name_clean


def entity_sentiment(text, entity):
    """Determine sentiment specifically toward the identified entity."""
    if entity in ("MARKET", "NONE"):
        return classify_sentiment(text)

    # For a specific entity, look at sentences mentioning it
    entity_lower = entity.lower()

    # Get a distinctive word from the entity name for matching
    entity_words = entity_lower.split()
    # Pick the most distinctive word (skip common words)
    skip = {'the', 'a', 'an', 'of', 'and', 'inc.', 'inc', 'corp.', 'corp',
            'corporation', 'company', 'ltd.', 'ltd', 'llc', 'group', 'holdings',
            'co.', 'co', 'incorporated', 'limited', 'plc', 'technologies'}
    distinctive_words = [w for w in entity_words if w not in skip]
    primary_word = distinctive_words[0] if distinctive_words else entity_words[0]

    # Find sentences containing the entity
    sentences = re.split(r'[.!?]+', text)
    entity_sentences = [s for s in sentences if primary_word in s.lower()]

    if not entity_sentences:
        return classify_sentiment(text)

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

    # Check distinctive words from the entity
    skip = {'the', 'a', 'an', 'of', 'and', 'inc.', 'inc', 'corp.', 'corp',
            'corporation', 'company', 'ltd.', 'ltd', 'llc', 'group', 'holdings',
            'co.', 'co', 'incorporated', 'limited', 'plc', 'technologies'}
    for part in entity.split():
        if len(part) > 2 and part.lower() not in skip and part.lower() in text_lower:
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

    label_counts = out_df['label'].value_counts().to_dict()
    entity_counts = out_df['entity'].value_counts()
    top_entities = entity_counts.head(5).to_dict()
    unique_entities = len(entity_counts)

    return label_counts, top_entities, unique_entities


def main():
    print("=" * 70)
    print("Processing batches 0135-0179 (45 batches, ~4500 rows)")
    print("=" * 70)

    all_labels = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    all_entities = {}

    for batch_num in range(135, 180):
        print(f"\nBatch {batch_num:04d}...", end=" ")
        result = process_batch(batch_num)
        if result:
            label_counts, top_entities, unique = result
            print(f"Labels: {label_counts} | {unique} unique entities | Top: {list(top_entities.keys())[:3]}")
            for k, v in label_counts.items():
                all_labels[k] = all_labels.get(k, 0) + v
            for k, v in top_entities.items():
                all_entities[k] = all_entities.get(k, 0) + v

    print("\n" + "=" * 70)
    print("SUMMARY")
    total = sum(all_labels.values())
    print(f"  Total rows: {total}")
    print(f"  Label distribution:")
    for k, v in sorted(all_labels.items()):
        print(f"    {k}: {v} ({100*v/total:.1f}%)")

    print(f"\n  Top 20 entities:")
    sorted_entities = sorted(all_entities.items(), key=lambda x: -x[1])
    for name, count in sorted_entities[:20]:
        print(f"    {name}: {count}")

    print(f"\n  Total unique entities: {len(all_entities)}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
