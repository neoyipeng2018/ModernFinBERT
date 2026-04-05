import pandas as pd
import re
import os
import json

OUTPUT_DIR = "data/processed"
ANNOTATION_DIR = f"{OUTPUT_DIR}/entity_annotations"

TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")

MACRO_ENTITIES = [
    (re.compile(r"\b(federal reserve|the fed(?:eral)?|fomc|powell|yellen|bernanke)\b", re.I), "Federal Reserve"),
    (re.compile(r"\b(european central bank|ecb|lagarde|draghi)\b", re.I), "ECB"),
    (re.compile(r"\b(bank of england|boe)\b", re.I), "Bank of England"),
    (re.compile(r"\b(bank of japan|boj)\b", re.I), "Bank of Japan"),
    (re.compile(r"\b(people'?s bank of china|pboc)\b", re.I), "People's Bank of China"),
    (re.compile(r"\b(s&p\s*500|s&p500|spy)\b", re.I), "S&P 500"),
    (re.compile(r"\b(nasdaq|qqq)\b", re.I), "NASDAQ"),
    (re.compile(r"\b(dow jones|djia|the dow)\b", re.I), "Dow Jones"),
    (re.compile(r"\b(russell\s*2000|iwm)\b", re.I), "Russell 2000"),
    (re.compile(r"\b(bitcoin|btc)\b", re.I), "Bitcoin"),
    (re.compile(r"\b(ethereum|eth(?:er)?)\b", re.I), "Ethereum"),
    (re.compile(r"\b(crude oil|wti|brent)\b", re.I), "Crude Oil"),
    (re.compile(r"\bgold\b", re.I), "Gold"),
    (re.compile(r"\b(natural gas|natgas)\b", re.I), "Natural Gas"),
    (re.compile(r"\b(treasury|treasuries|10[\s-]?year)\b", re.I), "U.S. Treasuries"),
    (re.compile(r"\b(u\.?s\.?\s*dollar|usd|greenback)\b", re.I), "U.S. Dollar"),
]

COMPANY_PATTERNS = [
    (re.compile(r"\bapple\b", re.I), "Apple Inc."),
    (re.compile(r"\bmicrosoft\b", re.I), "Microsoft Corp."),
    (re.compile(r"\bgoogle|alphabet\b", re.I), "Alphabet Inc."),
    (re.compile(r"\bamazon\b", re.I), "Amazon.com Inc."),
    (re.compile(r"\bmeta\s+platforms|facebook\b", re.I), "Meta Platforms Inc."),
    (re.compile(r"\btesla\b", re.I), "Tesla Inc."),
    (re.compile(r"\bnvidia\b", re.I), "NVIDIA Corp."),
    (re.compile(r"\bjpmorgan|jp\s*morgan\b", re.I), "JPMorgan Chase"),
    (re.compile(r"\bgoldman\s*sachs\b", re.I), "Goldman Sachs"),
    (re.compile(r"\bmorgan\s*stanley\b", re.I), "Morgan Stanley"),
    (re.compile(r"\bbank of america|bofa\b", re.I), "Bank of America"),
    (re.compile(r"\bcitigroup|citi\b", re.I), "Citigroup"),
    (re.compile(r"\bwells\s*fargo\b", re.I), "Wells Fargo"),
    (re.compile(r"\bberkshire\s*hathaway\b", re.I), "Berkshire Hathaway"),
    (re.compile(r"\bwalmart|wal[\s-]mart\b", re.I), "Walmart Inc."),
    (re.compile(r"\bnetflix\b", re.I), "Netflix Inc."),
    (re.compile(r"\bdisney\b", re.I), "Walt Disney Co."),
    (re.compile(r"\bboeing\b", re.I), "Boeing Co."),
    (re.compile(r"\bpfizer\b", re.I), "Pfizer Inc."),
    (re.compile(r"\bjohnson\s*&?\s*johnson\b", re.I), "Johnson & Johnson"),
    (re.compile(r"\bprocter\s*&?\s*gamble\b", re.I), "Procter & Gamble"),
    (re.compile(r"\bvisa\b", re.I), "Visa Inc."),
    (re.compile(r"\bmastercard\b", re.I), "Mastercard Inc."),
    (re.compile(r"\bpaypal\b", re.I), "PayPal Holdings"),
    (re.compile(r"\bsalesforce\b", re.I), "Salesforce Inc."),
    (re.compile(r"\bintuit\b", re.I), "Intuit Inc."),
    (re.compile(r"\boracle\b", re.I), "Oracle Corp."),
    (re.compile(r"\bibm\b", re.I), "IBM"),
    (re.compile(r"\bintel\b", re.I), "Intel Corp."),
    (re.compile(r"\bamd\b", re.I), "AMD"),
    (re.compile(r"\bqualcomm\b", re.I), "Qualcomm Inc."),
    (re.compile(r"\bcisco\b", re.I), "Cisco Systems"),
    (re.compile(r"\bsnapchat|snap inc\b", re.I), "Snap Inc."),
    (re.compile(r"\buber\b", re.I), "Uber Technologies"),
    (re.compile(r"\blyft\b", re.I), "Lyft Inc."),
    (re.compile(r"\bairbnb\b", re.I), "Airbnb Inc."),
    (re.compile(r"\bshopify\b", re.I), "Shopify Inc."),
    (re.compile(r"\btwitter\b", re.I), "Twitter/X"),
    (re.compile(r"\bchevron\b", re.I), "Chevron Corp."),
    (re.compile(r"\bexxon\s*mobil|exxon\b", re.I), "ExxonMobil"),
    (re.compile(r"\bshell\b", re.I), "Shell plc"),
    (re.compile(r"\bbp\b"), "BP plc"),
    (re.compile(r"\btotalenergies|total\s+sa\b", re.I), "TotalEnergies"),
    (re.compile(r"\bcocacola|coca[\s-]cola\b", re.I), "Coca-Cola Co."),
    (re.compile(r"\bpepsi(?:co)?\b", re.I), "PepsiCo Inc."),
    (re.compile(r"\bstarbucks\b", re.I), "Starbucks Corp."),
    (re.compile(r"\bmcdonald'?s\b", re.I), "McDonald's Corp."),
    (re.compile(r"\bnike\b", re.I), "Nike Inc."),
    (re.compile(r"\bford\b", re.I), "Ford Motor Co."),
    (re.compile(r"\bgeneral motors|gm\b", re.I), "General Motors"),
    (re.compile(r"\btoyota\b", re.I), "Toyota Motor Corp."),
    (re.compile(r"\bsony\b", re.I), "Sony Group"),
    (re.compile(r"\bsamsung\b", re.I), "Samsung Electronics"),
]

TICKER_TO_COMPANY = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOG": "Alphabet Inc.",
    "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com Inc.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.", "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "BAC": "Bank of America",
    "C": "Citigroup", "WFC": "Wells Fargo", "BRK": "Berkshire Hathaway",
    "WMT": "Walmart Inc.", "NFLX": "Netflix Inc.", "DIS": "Walt Disney Co.",
    "BA": "Boeing Co.", "PFE": "Pfizer Inc.", "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble", "V": "Visa Inc.", "MA": "Mastercard Inc.",
    "PYPL": "PayPal Holdings", "CRM": "Salesforce Inc.", "ORCL": "Oracle Corp.",
    "IBM": "IBM", "INTC": "Intel Corp.", "AMD": "AMD", "QCOM": "Qualcomm Inc.",
    "CSCO": "Cisco Systems", "SNAP": "Snap Inc.", "UBER": "Uber Technologies",
    "LYFT": "Lyft Inc.", "ABNB": "Airbnb Inc.", "SHOP": "Shopify Inc.",
    "CVX": "Chevron Corp.", "XOM": "ExxonMobil", "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.", "SBUX": "Starbucks Corp.", "MCD": "McDonald's Corp.",
    "NKE": "Nike Inc.", "F": "Ford Motor Co.", "GM": "General Motors",
    "SPY": "S&P 500", "QQQ": "NASDAQ", "DIA": "Dow Jones", "IWM": "Russell 2000",
    "BTC": "Bitcoin", "ETH": "Ethereum",
    "COIN": "Coinbase", "SQ": "Block Inc.", "PLTR": "Palantir Technologies",
    "RIVN": "Rivian Automotive", "LCID": "Lucid Group", "NIO": "NIO Inc.",
    "BABA": "Alibaba Group", "JD": "JD.com", "TSM": "TSMC",
    "ROKU": "Roku Inc.", "DKNG": "DraftKings", "HOOD": "Robinhood Markets",
    "GME": "GameStop Corp.", "AMC": "AMC Entertainment", "BB": "BlackBerry",
    "NOK": "Nokia Corp.", "SOFI": "SoFi Technologies", "SPCE": "Virgin Galactic",
}

MARKET_RE = re.compile(
    r"\b(market(?:s)?|stock(?:s)?|equit(?:y|ies)|rall(?:y|ied|ies)|sell[\s-]?off|"
    r"bull(?:ish)?|bear(?:ish)?|wall\s*street|investor(?:s)?|trader(?:s)?|"
    r"trading|IPO|earnings?\s+season)\b", re.I
)

SECTOR_PATTERNS = [
    (re.compile(r"\b(tech(?:nology)?|software|semiconductor|AI|artificial intelligence)\b", re.I), "Technology"),
    (re.compile(r"\b(healthcare|pharma|biotech|medical|drug)\b", re.I), "Healthcare"),
    (re.compile(r"\b(energy|oil\s+&?\s*gas|renewable|solar|wind\s+energy)\b", re.I), "Energy"),
    (re.compile(r"\b(financ(?:e|ial)|bank(?:ing)?|insurance)\b", re.I), "Financials"),
    (re.compile(r"\b(real\s+estate|housing|mortgage|REIT)\b", re.I), "Real Estate"),
    (re.compile(r"\b(consumer\s+(?:discretionary|staples)|retail)\b", re.I), "Consumer"),
    (re.compile(r"\b(industrial|manufacturing|aerospace|defense)\b", re.I), "Industrials"),
    (re.compile(r"\b(utilit(?:y|ies)|electric(?:ity)?|power\s+grid)\b", re.I), "Utilities"),
    (re.compile(r"\b(telecom|communications|5G)\b", re.I), "Communications"),
    (re.compile(r"\b(material(?:s)?|mining|steel|copper|lithium)\b", re.I), "Materials"),
]

NEGATION_RE = re.compile(r"\b(not|no|never|neither|nor|hardly|barely|n't|don't|doesn't|didn't|won't|wouldn't|can't|cannot|isn't|aren't|wasn't|weren't)\b", re.I)
NEGATIVE_WORDS = re.compile(r"\b(loss|losses|decline|declined|declining|decrease|decreased|drop|dropped|fell|fall|falling|weak|weaker|weakness|miss|missed|below|downgrade|downgraded|cut|cuts|risk|risks|concern|concerned|warning|warned|deficit|layoff|layoffs|restructuring|impairment|writedown|lawsuit|penalty|fine|fraud|bankruptcy|default|struggle|struggling|slump|slumped|plunge|plunged|crash|crashed|recession|downturn|bearish|sell[\s-]?off|underperform|disappoint|disappointed|disappointing)\b", re.I)
POSITIVE_WORDS = re.compile(r"\b(gain|gains|growth|grew|increase|increased|rising|rose|beat|beats|exceeded|exceeding|above|upgrade|upgraded|strong|stronger|strength|record|high|profit|profitable|surge|surged|rally|rallied|boost|boosted|outperform|dividend|buyback|expansion|innovation|breakthrough|approval|recover|recovered|recovery|bullish|optimistic|upbeat|soar|soared)\b", re.I)


def extract_entity(text):
    tickers = TICKER_RE.findall(text)
    if tickers:
        ticker = tickers[0]
        return TICKER_TO_COMPANY.get(ticker, f"${ticker}")

    for pattern, entity_name in MACRO_ENTITIES:
        if pattern.search(text):
            return entity_name

    for pattern, company_name in COMPANY_PATTERNS:
        if pattern.search(text):
            return company_name

    for pattern, sector_name in SECTOR_PATTERNS:
        if pattern.search(text):
            return sector_name

    if MARKET_RE.search(text):
        return "MARKET"

    return "NONE"


def assign_entity_sentiment(text, sentence_label, entity):
    if entity in ("NONE", "MARKET"):
        return sentence_label

    entity_lower = entity.lower().replace("$", "").split()[0]
    text_lower = text.lower()
    pos = text_lower.find(entity_lower)
    if pos == -1:
        return sentence_label

    window_start = max(0, pos - 100)
    window_end = min(len(text_lower), pos + len(entity_lower) + 100)
    window = text_lower[window_start:window_end]

    neg_hits = len(NEGATIVE_WORDS.findall(window))
    pos_hits = len(POSITIVE_WORDS.findall(window))
    has_negation = bool(NEGATION_RE.search(window))

    if neg_hits > 0 and pos_hits == 0:
        return "POSITIVE" if has_negation else "NEGATIVE"
    elif pos_hits > 0 and neg_hits == 0:
        return "NEGATIVE" if has_negation else "POSITIVE"
    else:
        return sentence_label


def annotate_dataframe(df):
    entities = []
    entity_sentiments = []
    for _, row in df.iterrows():
        entity = extract_entity(row["text"])
        entity_sent = assign_entity_sentiment(row["text"], row["label"], entity)
        entities.append(entity)
        entity_sentiments.append(entity_sent)
    df["entity"] = entities
    df["entity_sentiment"] = entity_sentiments
    return df


if __name__ == "__main__":
    df = pd.read_parquet(f"{OUTPUT_DIR}/all_deduped.parquet")
    print(f"Loaded {len(df)} rows for annotation")

    df = annotate_dataframe(df)

    assert df["entity"].notna().all()
    assert df["entity_sentiment"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"]).all()

    print(f"\nEntity coverage (non-NONE): {(df['entity'] != 'NONE').mean():.1%}")
    print(f"\nTop 30 entities:\n{df['entity'].value_counts().head(30)}")
    print(f"\nEntity sentiment distribution:\n{df['entity_sentiment'].value_counts()}")
    print(f"\nAgreement with sentence label: {(df['entity_sentiment'] == df['label']).mean():.1%}")

    df.to_parquet(f"{OUTPUT_DIR}/all_annotated.parquet", index=False)
    print(f"\nSaved to {OUTPUT_DIR}/all_annotated.parquet")
