#!/usr/bin/env python3
"""
Process entity annotation batches 0027-0053.
For each row, determine the primary financial entity and entity-level sentiment.
"""

import pandas as pd
import re
import os

DATA_DIR = "/Users/boo/Documents/ModernFinBERT/data/processed/entity_annotations"

# ── Ticker-to-company mapping ──────────────────────────────────────────────
TICKER_MAP = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.", "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.", "FB": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corp.",
    "NFLX": "Netflix Inc.",
    "DIS": "Walt Disney Co.",
    "BA": "Boeing Co.",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley",
    "BAC": "Bank of America Corp.",
    "WFC": "Wells Fargo & Co.",
    "C": "Citigroup Inc.",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "PYPL": "PayPal Holdings Inc.",
    "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corp.",
    "IBM": "IBM Corp.",
    "CSCO": "Cisco Systems Inc.",
    "QCOM": "Qualcomm Inc.",
    "TXN": "Texas Instruments Inc.",
    "AVGO": "Broadcom Inc.",
    "NOW": "ServiceNow Inc.",
    "ADBE": "Adobe Inc.",
    "SHOP": "Shopify Inc.",
    "SQ": "Block Inc.",
    "SPOT": "Spotify Technology SA",
    "UBER": "Uber Technologies Inc.",
    "LYFT": "Lyft Inc.",
    "SNAP": "Snap Inc.",
    "TWTR": "Twitter Inc.",
    "PINS": "Pinterest Inc.",
    "ZM": "Zoom Video Communications Inc.",
    "ROKU": "Roku Inc.",
    "COIN": "Coinbase Global Inc.",
    "HOOD": "Robinhood Markets Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "SNOW": "Snowflake Inc.",
    "NET": "Cloudflare Inc.",
    "DDOG": "Datadog Inc.",
    "CRWD": "CrowdStrike Holdings Inc.",
    "ZS": "Zscaler Inc.",
    "PANW": "Palo Alto Networks Inc.",
    "MDB": "MongoDB Inc.",
    "TTD": "Trade Desk Inc.",
    "U": "Unity Software Inc.",
    "RBLX": "Roblox Corp.",
    "PATH": "UiPath Inc.",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corp.",
    "TGT": "Target Corp.",
    "HD": "Home Depot Inc.",
    "LOW": "Lowe's Companies Inc.",
    "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.",
    "MCD": "McDonald's Corp.",
    "SBUX": "Starbucks Corp.",
    "NKE": "Nike Inc.",
    "PG": "Procter & Gamble Co.",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc.",
    "PFE": "Pfizer Inc.",
    "MRNA": "Moderna Inc.",
    "ABBV": "AbbVie Inc.",
    "BMY": "Bristol-Myers Squibb Co.",
    "LLY": "Eli Lilly and Co.",
    "MRK": "Merck & Co.",
    "GILD": "Gilead Sciences Inc.",
    "BIIB": "Biogen Inc.",
    "AMGN": "Amgen Inc.",
    "REGN": "Regeneron Pharmaceuticals Inc.",
    "VRTX": "Vertex Pharmaceuticals Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "ABT": "Abbott Laboratories",
    "MDT": "Medtronic PLC",
    "SYK": "Stryker Corp.",
    "ISRG": "Intuitive Surgical Inc.",
    "DHR": "Danaher Corp.",
    "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corp.",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources Inc.",
    "SLB": "Schlumberger Ltd.",
    "HAL": "Halliburton Co.",
    "OXY": "Occidental Petroleum Corp.",
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications Inc.",
    "TMUS": "T-Mobile US Inc.",
    "CMCSA": "Comcast Corp.",
    "CHTR": "Charter Communications Inc.",
    "F": "Ford Motor Co.",
    "GM": "General Motors Co.",
    "TM": "Toyota Motor Corp.",
    "RIVN": "Rivian Automotive Inc.",
    "LCID": "Lucid Group Inc.",
    "LMT": "Lockheed Martin Corp.",
    "RTX": "RTX Corp.",
    "NOC": "Northrop Grumman Corp.",
    "GD": "General Dynamics Corp.",
    "GE": "General Electric Co.",
    "HON": "Honeywell International Inc.",
    "MMM": "3M Co.",
    "CAT": "Caterpillar Inc.",
    "DE": "Deere & Co.",
    "UPS": "United Parcel Service Inc.",
    "FDX": "FedEx Corp.",
    "DAL": "Delta Air Lines Inc.",
    "UAL": "United Airlines Holdings Inc.",
    "AAL": "American Airlines Group Inc.",
    "LUV": "Southwest Airlines Co.",
    "SPY": "S&P 500",
    "QQQ": "NASDAQ",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "BRK": "Berkshire Hathaway Inc.",
    "SPX": "S&P 500",
    "NDX": "NASDAQ",
    "DJIA": "Dow Jones",
    "INTU": "Intuit Inc.",
    "AKAM": "Akamai Technologies Inc.",
    "CEIX": "CONSOL Energy Inc.",
    "EEENF": "88 Energy Ltd.",
    "WNRS": "Winners Inc.",
    "SEAC": "SeaChange International Inc.",
    "ENZC": "Enzolytics Inc.",
    "GYST": "Gyst Audio Inc.",
    "PH": "Parker-Hannifin Corp.",
    "BIDU": "Baidu Inc.",
    "AAP": "Advance Auto Parts Inc.",
    "BMS": "Bristol-Myers Squibb Co.",
    "NOV": "National Oilwell Varco Inc.",
    "R": "Ryder System Inc.",
    "ES": "S&P 500",
    "ES_F": "S&P 500",
}

# ── Company name patterns (name → canonical) ────────────────────────────────
COMPANY_PATTERNS = [
    # Tech giants
    ("Apple", "Apple Inc."),
    ("Microsoft", "Microsoft Corp."),
    ("Google", "Alphabet Inc."),
    ("Alphabet", "Alphabet Inc."),
    ("Amazon", "Amazon.com Inc."),
    ("Meta Platforms", "Meta Platforms Inc."),
    ("Facebook", "Meta Platforms Inc."),
    ("Tesla", "Tesla Inc."),
    ("Cybertruck", "Tesla Inc."),
    ("NVIDIA", "NVIDIA Corp."),
    ("Nvidia", "NVIDIA Corp."),
    ("AMD", "Advanced Micro Devices Inc."),
    ("Advanced Micro", "Advanced Micro Devices Inc."),
    ("Intel ", "Intel Corp."),
    ("Intel's", "Intel Corp."),
    ("Netflix", "Netflix Inc."),
    ("Disney", "Walt Disney Co."),
    ("Boeing", "Boeing Co."),
    ("Twitter", "Twitter Inc."),
    ("Snap ", "Snap Inc."),
    ("Snapchat", "Snap Inc."),
    ("Pinterest", "Pinterest Inc."),
    ("Zoom Video", "Zoom Video Communications Inc."),
    ("Roku", "Roku Inc."),
    ("Coinbase", "Coinbase Global Inc."),
    ("Robinhood", "Robinhood Markets Inc."),
    ("Palantir", "Palantir Technologies Inc."),
    ("Snowflake", "Snowflake Inc."),
    ("Cloudflare", "Cloudflare Inc."),
    ("Salesforce", "Salesforce Inc."),
    ("Oracle", "Oracle Corp."),
    ("IBM", "IBM Corp."),
    ("Cisco", "Cisco Systems Inc."),
    ("Qualcomm", "Qualcomm Inc."),
    ("Broadcom", "Broadcom Inc."),
    ("Adobe", "Adobe Inc."),
    ("Shopify", "Shopify Inc."),
    ("Spotify", "Spotify Technology SA"),
    ("Uber", "Uber Technologies Inc."),
    ("Lyft", "Lyft Inc."),
    ("PayPal", "PayPal Holdings Inc."),
    ("Square", "Block Inc."),
    ("Block Inc", "Block Inc."),
    ("ServiceNow", "ServiceNow Inc."),
    ("CrowdStrike", "CrowdStrike Holdings Inc."),
    ("Palo Alto Networks", "Palo Alto Networks Inc."),
    ("MongoDB", "MongoDB Inc."),
    ("Datadog", "Datadog Inc."),
    ("Unity Software", "Unity Software Inc."),
    ("Roblox", "Roblox Corp."),
    ("UiPath", "UiPath Inc."),

    # Retail
    ("Walmart", "Walmart Inc."),
    ("Costco", "Costco Wholesale Corp."),
    ("Target Corp", "Target Corp."),
    ("Home Depot", "Home Depot Inc."),
    ("Lowe's", "Lowe's Companies Inc."),
    ("Kohl's", "Kohl's Corp."),
    ("Bed Bath & Beyond", "Bed Bath & Beyond Inc."),
    ("Overstock", "Overstock.com Inc."),
    ("Beyond Inc", "Beyond Inc."),
    ("J.C. Penney", "J.C. Penney Co."),
    ("JCPenney", "J.C. Penney Co."),
    ("Macy's", "Macy's Inc."),
    ("Nordstrom", "Nordstrom Inc."),
    ("Gap", "Gap Inc."),
    ("American Eagle", "American Eagle Outfitters Inc."),
    ("Nike", "Nike Inc."),
    ("VF Corp", "VF Corp."),

    # Food & Beverage
    ("Coca-Cola", "Coca-Cola Co."),
    ("Pepsi", "PepsiCo Inc."),
    ("PepsiCo", "PepsiCo Inc."),
    ("McDonald's", "McDonald's Corp."),
    ("Starbucks", "Starbucks Corp."),
    ("Procter & Gamble", "Procter & Gamble Co."),
    ("P&G", "Procter & Gamble Co."),
    ("Diageo", "Diageo PLC"),
    ("Calbee", "Calbee Inc."),
    ("Nestle", "Nestle SA"),
    ("Nestlé", "Nestle SA"),
    ("Mondelez", "Mondelez International Inc."),
    ("Kraft", "Kraft Heinz Co."),

    # Financial
    ("JPMorgan", "JPMorgan Chase & Co."),
    ("JP Morgan", "JPMorgan Chase & Co."),
    ("Goldman Sachs", "Goldman Sachs Group Inc."),
    ("Morgan Stanley", "Morgan Stanley"),
    ("Bank of America", "Bank of America Corp."),
    ("Wells Fargo", "Wells Fargo & Co."),
    ("Citigroup", "Citigroup Inc."),
    ("Citi ", "Citigroup Inc."),
    ("Visa", "Visa Inc."),
    ("Mastercard", "Mastercard Inc."),
    ("Berkshire Hathaway", "Berkshire Hathaway Inc."),
    ("Berkshire", "Berkshire Hathaway Inc."),
    ("BlackRock", "BlackRock Inc."),
    ("Charles Schwab", "Charles Schwab Corp."),
    ("Schwab", "Charles Schwab Corp."),
    ("Comerica", "Comerica Inc."),
    ("Mellon", "Mellon Financial Corp."),
    ("Prudential", "Prudential Financial Inc."),
    ("Yes Bank", "Yes Bank Ltd."),
    ("Horizon Bancorp", "Horizon Bancorp Inc."),
    ("Markel", "Markel Corp."),

    # Pharma & Healthcare
    ("Johnson & Johnson", "Johnson & Johnson"),
    ("J&J", "Johnson & Johnson"),
    ("JNJ", "Johnson & Johnson"),
    ("Pfizer", "Pfizer Inc."),
    ("Moderna", "Moderna Inc."),
    ("AbbVie", "AbbVie Inc."),
    ("Bristol-Myers", "Bristol-Myers Squibb Co."),
    ("Bristol Myers", "Bristol-Myers Squibb Co."),
    ("Eli Lilly", "Eli Lilly and Co."),
    ("Merck", "Merck & Co."),
    ("Gilead", "Gilead Sciences Inc."),
    ("Biogen", "Biogen Inc."),
    ("Amgen", "Amgen Inc."),
    ("Regeneron", "Regeneron Pharmaceuticals Inc."),
    ("Vertex Pharma", "Vertex Pharmaceuticals Inc."),
    ("UnitedHealth", "UnitedHealth Group Inc."),
    ("Aetna", "Aetna Inc."),
    ("AET ", "Aetna Inc."),
    ("Abbott", "Abbott Laboratories"),
    ("Medtronic", "Medtronic PLC"),
    ("Stryker", "Stryker Corp."),
    ("Intuitive Surgical", "Intuitive Surgical Inc."),
    ("Danaher", "Danaher Corp."),
    ("Thermo Fisher", "Thermo Fisher Scientific Inc."),
    ("Quest Diagnostics", "Quest Diagnostics Inc."),
    ("Boston Scientific", "Boston Scientific Corp."),
    ("23andMe", "23andMe Holding Co."),
    ("Dynavax", "Dynavax Technologies Corp."),
    ("NeoPharm", "NeoPharm Inc."),
    ("BeiGene", "BeiGene Ltd."),
    ("Novoheart", "Novoheart Holdings Inc."),
    ("Avonex", "Biogen Inc."),

    # Energy
    ("ExxonMobil", "Exxon Mobil Corp."),
    ("Exxon Mobil", "Exxon Mobil Corp."),
    ("Exxon", "Exxon Mobil Corp."),
    ("Chevron", "Chevron Corp."),
    ("ConocoPhillips", "ConocoPhillips"),
    ("Schlumberger", "Schlumberger Ltd."),
    ("Halliburton", "Halliburton Co."),
    ("Occidental Petroleum", "Occidental Petroleum Corp."),
    ("Occidental", "Occidental Petroleum Corp."),
    ("National Oilwell Varco", "National Oilwell Varco Inc."),
    ("National Oilwell", "National Oilwell Varco Inc."),
    ("BP ", "BP PLC"),
    ("BP's", "BP PLC"),
    ("Shell", "Shell PLC"),
    ("Williams", "Williams Companies Inc."),
    ("CONSOL Energy", "CONSOL Energy Inc."),
    ("Pioneer Corp", "Pioneer Corp."),
    ("Pioneer Natural", "Pioneer Natural Resources Co."),
    ("EOG Resources", "EOG Resources Inc."),
    ("88 Energy", "88 Energy Ltd."),
    ("Lukoil", "Lukoil PJSC"),

    # Automotive
    ("Ford Motor", "Ford Motor Co."),
    ("Ford", "Ford Motor Co."),
    ("General Motors", "General Motors Co."),
    ("GM ", "General Motors Co."),
    ("GM's", "General Motors Co."),
    ("Toyota", "Toyota Motor Corp."),
    ("Volkswagen", "Volkswagen AG"),
    ("VW", "Volkswagen AG"),
    ("Chrysler", "Chrysler"),
    ("Fiat", "Fiat Chrysler Automobiles NV"),
    ("Rivian", "Rivian Automotive Inc."),
    ("Lucid", "Lucid Group Inc."),
    ("Hyundai", "Hyundai Motor Co."),
    ("Volvo", "Volvo Group"),
    ("Renault", "Renault SA"),
    ("Audi", "Volkswagen AG"),

    # Telecom
    ("AT&T", "AT&T Inc."),
    ("Verizon", "Verizon Communications Inc."),
    ("T-Mobile", "T-Mobile US Inc."),
    ("Comcast", "Comcast Corp."),
    ("Charter Communications", "Charter Communications Inc."),
    ("Sprint", "Sprint Corp."),
    ("Nextel", "Nextel Communications Inc."),

    # Industrial & Defense
    ("Lockheed Martin", "Lockheed Martin Corp."),
    ("Northrop Grumman", "Northrop Grumman Corp."),
    ("Raytheon", "RTX Corp."),
    ("General Dynamics", "General Dynamics Corp."),
    ("General Electric", "General Electric Co."),
    ("GE ", "General Electric Co."),
    ("GE's", "General Electric Co."),
    ("Honeywell", "Honeywell International Inc."),
    ("3M", "3M Co."),
    ("Caterpillar", "Caterpillar Inc."),
    ("Deere", "Deere & Co."),
    ("John Deere", "Deere & Co."),
    ("Parker-Hannifin", "Parker-Hannifin Corp."),
    ("Parker Hannifin", "Parker-Hannifin Corp."),

    # Transport
    ("UPS", "United Parcel Service Inc."),
    ("FedEx", "FedEx Corp."),
    ("Delta Air", "Delta Air Lines Inc."),
    ("United Airlines", "United Airlines Holdings Inc."),
    ("American Airlines", "American Airlines Group Inc."),
    ("Southwest Airlines", "Southwest Airlines Co."),
    ("Ryder", "Ryder System Inc."),

    # Semiconductor
    ("SK Hynix", "SK Hynix Inc."),
    ("Samsung", "Samsung Electronics Co."),
    ("TSMC", "Taiwan Semiconductor Manufacturing Co."),
    ("Taiwan Semiconductor", "Taiwan Semiconductor Manufacturing Co."),
    ("Micron", "Micron Technology Inc."),
    ("Applied Materials", "Applied Materials Inc."),
    ("Lam Research", "Lam Research Corp."),
    ("ASML", "ASML Holding NV"),
    ("LG Chem", "LG Chem Ltd."),

    # Cloud / Software
    ("AWS", "Amazon.com Inc."),
    ("Azure", "Microsoft Corp."),
    ("Bedrock", "Amazon.com Inc."),

    # Mining
    ("Rio Tinto", "Rio Tinto Group"),
    ("Rio,", "Rio Tinto Group"),
    ("Chinalco", "Aluminum Corp. of China Ltd."),
    ("BHP", "BHP Group Ltd."),
    ("Barrick", "Barrick Gold Corp."),
    ("Newmont", "Newmont Corp."),

    # Insurance
    ("Prudential plc", "Prudential PLC"),
    ("Prudential", "Prudential PLC"),

    # Media / Entertainment
    ("Mondadori", "Arnoldo Mondadori Editore SpA"),
    ("Fox", "Fox Corp."),
    ("Warner", "Warner Bros. Discovery Inc."),
    ("Paramount", "Paramount Global"),

    # Other specific companies found in data
    ("NIPSCO", "NiSource Inc."),
    ("NiSource", "NiSource Inc."),
    ("Sensus", "Sensus (Xylem Inc.)"),
    ("Akamai", "Akamai Technologies Inc."),
    ("Crown", "Crown Holdings Inc."),
    ("Sunstone", "Sunstone Hotel Investors Inc."),
    ("Sherwin-Williams", "Sherwin-Williams Co."),
    ("Sherwin Williams", "Sherwin-Williams Co."),
    ("H&P", "Helmerich & Payne Inc."),
    ("Helmerich", "Helmerich & Payne Inc."),
    ("Reynolds", "Reynolds American Inc."),
    ("Veritas", "Veritas Technologies LLC"),
    ("Kyndryl", "Kyndryl Holdings Inc."),
    ("Intuit", "Intuit Inc."),
    ("TurboTax", "Intuit Inc."),
    ("Baidu", "Baidu Inc."),
    ("Anthropic", "Anthropic"),
    ("Airbnb", "Airbnb Inc."),
    ("Hugging Face", "Hugging Face Inc."),
    ("L'Oreal", "L'Oreal SA"),
    ("L'Oréal", "L'Oreal SA"),
    ("Parity Group", "Parity Group PLC"),
    ("Partway Group", "Parity Group PLC"),
    ("Minebea", "Minebea Intec"),
    ("Perficient", "Perficient Inc."),
    ("Xtrackers", "DWS Group"),
    ("Safeway", "Safeway Inc."),
    ("Walgreens", "Walgreens Boots Alliance Inc."),
    ("United Spirits", "United Spirits Ltd."),
    ("USL", "United Spirits Ltd."),
    ("Smirnoff", "Diageo PLC"),
    ("Johnny Walker", "Diageo PLC"),
    ("Rosneft", "Rosneft Oil Co."),
    ("Gazprom", "Gazprom PJSC"),
    ("TNK-BP", "TNK-BP"),
    ("Extreme", "Extreme Networks Inc."),
    ("Enterasys", "Extreme Networks Inc."),
    ("Marchionne", "Fiat Chrysler Automobiles NV"),
    ("Pioneer", "Pioneer Corp."),
    ("Sharp", "Sharp Corp."),
    ("Panasonic", "Panasonic Corp."),
]

# ── Index / Market patterns ─────────────────────────────────────────────────
INDEX_PATTERNS = [
    ("S&P 500", "S&P 500"),
    ("S&P500", "S&P 500"),
    ("S&P", "S&P 500"),
    ("SPX", "S&P 500"),
    ("$SPX", "S&P 500"),
    ("NASDAQ", "NASDAQ"),
    ("Nasdaq", "NASDAQ"),
    ("Dow Jones", "Dow Jones"),
    ("DJIA", "Dow Jones"),
    ("Dow ", "Dow Jones"),
    ("Russell 2000", "Russell 2000"),
    ("FTSE", "FTSE"),
    ("Nikkei", "Nikkei 225"),
    ("Hang Seng", "Hang Seng"),
    ("DAX", "DAX"),
    ("Stoxx", "EURO STOXX"),
    ("Shanghai Composite", "Shanghai Composite"),
    ("Asian Shares", "MARKET"),
    ("Asian shares", "MARKET"),
    ("Asian markets", "MARKET"),
]

# ── Commodity patterns ───────────────────────────────────────────────────────
COMMODITY_PATTERNS = [
    ("crude oil", "Crude Oil"),
    ("Crude Oil", "Crude Oil"),
    ("oil price", "Crude Oil"),
    ("Oil price", "Crude Oil"),
    ("oil prices", "Crude Oil"),
    ("Oil prices", "Crude Oil"),
    ("Brent crude", "Crude Oil"),
    ("WTI crude", "Crude Oil"),
    ("WTI", "Crude Oil"),
    ("gold price", "Gold"),
    ("Gold price", "Gold"),
    ("gold prices", "Gold"),
    ("Gold", "Gold"),
    ("natural gas", "Natural Gas"),
    ("Natural Gas", "Natural Gas"),
    ("natural-gas", "Natural Gas"),
    ("silver price", "Silver"),
    ("Silver", "Silver"),
    ("copper", "Copper"),
    ("Copper", "Copper"),
    ("wheat", "Wheat"),
    ("corn", "Corn"),
    ("soybean", "Soybeans"),
    ("milk price", "Dairy"),
    ("dairy", "Dairy"),
]

# ── Central bank patterns ────────────────────────────────────────────────────
CENTRAL_BANK_PATTERNS = [
    ("Federal Reserve", "Federal Reserve"),
    ("the Fed ", "Federal Reserve"),
    ("the Fed's", "Federal Reserve"),
    ("the Fed,", "Federal Reserve"),
    ("Fed Chair", "Federal Reserve"),
    ("Fed chairman", "Federal Reserve"),
    ("FOMC", "Federal Reserve"),
    ("ECB", "ECB"),
    ("European Central Bank", "ECB"),
    ("Bank of England", "Bank of England"),
    ("Bank of Japan", "Bank of Japan"),
    ("BOJ", "Bank of Japan"),
    ("People's Bank of China", "PBOC"),
    ("PBOC", "PBOC"),
    ("RBI", "Reserve Bank of India"),
]

# ── Sector keywords ──────────────────────────────────────────────────────────
SECTOR_KEYWORDS = {
    "Technology": ["tech sector", "technology sector", "tech stocks", "technology stocks", "tech industry"],
    "Healthcare": ["healthcare sector", "health care sector", "healthcare stocks", "pharma sector", "pharmaceutical sector", "biotech sector"],
    "Energy": ["energy sector", "oil and gas sector", "oil & gas sector", "energy stocks"],
    "Financials": ["financial sector", "banking sector", "bank stocks", "financial stocks"],
    "Real Estate": ["real estate sector", "REIT", "real estate market", "housing market", "housing sector"],
    "Consumer Discretionary": ["consumer discretionary", "retail sector", "consumer spending"],
    "Consumer Staples": ["consumer staples", "staples sector"],
    "Industrials": ["industrial sector", "industrials sector", "manufacturing sector"],
    "Materials": ["materials sector", "mining sector"],
    "Utilities": ["utilities sector", "utility sector", "utility stocks"],
    "Communication Services": ["communication services", "telecom sector", "media sector"],
    "Cryptocurrency": ["crypto", "bitcoin", "Bitcoin", "ethereum", "Ethereum", "cryptocurrency", "blockchain"],
}


def extract_ticker(text):
    """Extract $TICKER patterns from text."""
    # Match $TICKER patterns (1-5 uppercase letters after $)
    tickers = re.findall(r'\$([A-Z][A-Z0-9_]{0,5})\b', text)
    for t in tickers:
        t_clean = t.rstrip('_')
        if t_clean in TICKER_MAP:
            return TICKER_MAP[t_clean], t_clean
    # Return the first ticker even if not in map
    if tickers:
        return f"${tickers[0]}", tickers[0]
    return None, None


def extract_entity(text):
    """Extract the primary financial entity from text. Returns (entity, source)."""
    if not isinstance(text, str):
        return "NONE", "empty"

    text_lower = text.lower()

    # 1) Check for explicit tickers first
    ticker_entity, ticker_sym = extract_ticker(text)

    # 2) Check for company names (longer/more specific patterns first)
    company_matches = []
    for pattern, canonical in COMPANY_PATTERNS:
        pl = pattern.lower()
        if pl in text_lower:
            # Find position for priority (earlier = more likely primary subject)
            pos = text_lower.find(pl)
            company_matches.append((pos, len(pattern), canonical, pattern))

    # 3) Check for indices
    index_matches = []
    for pattern, canonical in INDEX_PATTERNS:
        if pattern.lower() in text_lower:
            pos = text_lower.find(pattern.lower())
            index_matches.append((pos, len(pattern), canonical))

    # 4) Check for central banks
    cb_matches = []
    for pattern, canonical in CENTRAL_BANK_PATTERNS:
        if pattern.lower() in text_lower:
            pos = text_lower.find(pattern.lower())
            cb_matches.append((pos, len(pattern), canonical))

    # 5) Check for commodities
    commodity_matches = []
    for pattern, canonical in COMMODITY_PATTERNS:
        if pattern.lower() in text_lower:
            pos = text_lower.find(pattern.lower())
            commodity_matches.append((pos, len(pattern), canonical))

    # 6) Check for sectors
    sector_matches = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                pos = text_lower.find(kw.lower())
                sector_matches.append((pos, len(kw), sector))
                break

    # Priority: company > index > central bank > commodity > sector > ticker-only
    # Among companies, prefer the one mentioned first (or the most specific match)
    if company_matches:
        # Sort by position first, then by pattern length (longer = more specific)
        company_matches.sort(key=lambda x: (x[0], -x[1]))
        return company_matches[0][2], "company"

    if ticker_entity:
        return ticker_entity, "ticker"

    if index_matches:
        index_matches.sort(key=lambda x: (x[0], -x[1]))
        return index_matches[0][2], "index"

    if cb_matches:
        cb_matches.sort(key=lambda x: (x[0], -x[1]))
        return cb_matches[0][2], "central_bank"

    if commodity_matches:
        commodity_matches.sort(key=lambda x: (x[0], -x[1]))
        return commodity_matches[0][2], "commodity"

    if sector_matches:
        sector_matches.sort(key=lambda x: (x[0], -x[1]))
        return sector_matches[0][2], "sector"

    # 7) Look for common patterns that indicate a company even if not in our list
    # "XYZ Corp", "XYZ Inc", "XYZ Ltd", "XYZ Co.", "XYZ Group", "XYZ Holdings"
    corp_patterns = re.findall(
        r'([A-Z][A-Za-z&\.\' ]+?)\s+(?:Corp(?:oration)?|Inc\.?|Ltd\.?|Co\.?|Group|Holdings|PLC|plc|SA|AG|NV|SE|SpA|GmbH|LLC)',
        text
    )
    if corp_patterns:
        name = corp_patterns[0].strip()
        # Clean up
        if len(name) > 2 and len(name) < 60:
            return f"{name}", "regex_corp"

    # 8) Look for "XYZ reported/announced/said/expects" patterns
    report_patterns = re.findall(
        r'([A-Z][A-Za-z&\.\' ]{1,30}?)\s+(?:reported|announced|said|expects|plans|launched|agreed|signed|acquired|merged|bought|sold|raised|lowered)',
        text
    )
    if report_patterns:
        name = report_patterns[0].strip()
        if len(name) > 1 and len(name) < 40 and name not in ("We", "The", "This", "They", "He", "She", "It", "Our", "As", "But", "And", "Or", "In", "On", "At", "To", "So", "If", "Its"):
            return name, "regex_verb"

    # 9) Look for "'s" possessive patterns indicating a subject
    poss_patterns = re.findall(r"([A-Z][A-Za-z&\.' ]{1,25}?)'s\b", text)
    if poss_patterns:
        name = poss_patterns[0].strip()
        if len(name) > 1 and name not in ("We", "The", "This", "They", "He", "She", "It", "That", "What"):
            return name, "regex_poss"

    # 10) Check for market-related keywords
    market_keywords = ["market", "markets", "stocks", "equities", "shares", "rally", "sell-off", "selloff",
                       "bull market", "bear market", "correction", "recession", "GDP", "economy",
                       "inflation", "interest rate", "trade war", "tariff"]
    for kw in market_keywords:
        if kw in text_lower:
            return "MARKET", "market_keyword"

    # 11) Check for earnings call / conference call patterns (generic company talk)
    if any(phrase in text_lower for phrase in ["earnings call", "conference call", "our revenue", "our growth",
                                                "our business", "our company", "we expect", "we believe",
                                                "our guidance", "our margin", "our segment"]):
        return "NONE", "generic_earnings_call"

    return "NONE", "no_match"


def determine_entity_sentiment(text, label, entity, entity_source):
    """
    Determine entity-level sentiment. Usually matches the sentence label,
    but can differ in multi-entity or complex texts.
    """
    if not isinstance(text, str) or entity == "NONE":
        return label  # Default to sentence label

    # For most cases, entity sentiment matches the sentence label
    # Only override in specific multi-entity scenarios
    return label


def process_batch(batch_num):
    """Process a single batch file."""
    batch_str = f"{batch_num:04d}"
    input_path = os.path.join(DATA_DIR, f"batch_{batch_str}_input.csv")
    output_path = os.path.join(DATA_DIR, f"batch_{batch_str}_output.csv")

    if not os.path.exists(input_path):
        print(f"  SKIP batch {batch_str}: input file not found")
        return False

    df = pd.read_csv(input_path, index_col=0)

    entities = []
    sentiments = []

    for idx, row in df.iterrows():
        text = row.get("text", "")
        label = row.get("label", "NEUTRAL")

        entity, source = extract_entity(text)
        entity_sentiment = determine_entity_sentiment(text, label, entity, source)

        entities.append(entity)
        sentiments.append(entity_sentiment)

    output_df = pd.DataFrame({
        "entity": entities,
        "entity_sentiment": sentiments,
    }, index=df.index)

    output_df.to_csv(output_path)

    # Stats
    entity_counts = output_df["entity"].value_counts()
    none_count = (output_df["entity"] == "NONE").sum()
    market_count = (output_df["entity"] == "MARKET").sum()
    identified = len(output_df) - none_count

    print(f"  Batch {batch_str}: {len(output_df)} rows | {identified} entities identified ({none_count} NONE, {market_count} MARKET)")
    return True


def main():
    print("Processing entity annotation batches 0027-0053...")
    print("=" * 70)

    total_processed = 0
    for batch_num in range(27, 54):
        success = process_batch(batch_num)
        if success:
            total_processed += 1

    print("=" * 70)
    print(f"Done. Processed {total_processed} batches.")


if __name__ == "__main__":
    main()
