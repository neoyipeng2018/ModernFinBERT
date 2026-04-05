#!/usr/bin/env python3
"""
Entity annotation script for batches 0135-0161.
Identifies financial entities and entity-level sentiment from text.
"""

import pandas as pd
import re
import os

BASE_DIR = "/Users/boo/Documents/ModernFinBERT/data/processed/entity_annotations"

# ─── Ticker-to-company mapping ───────────────────────────────────────────────
TICKER_MAP = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com Inc.",
    "GOOG": "Alphabet Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "FB": "Meta Platforms Inc.",
    "NFLX": "Netflix Inc.",
    "NVDA": "NVIDIA Corp.",
    "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corp.",
    "BA": "Boeing Co.",
    "DIS": "Walt Disney Co.",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley",
    "BAC": "Bank of America Corp.",
    "WFC": "Wells Fargo & Co.",
    "C": "Citigroup Inc.",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "PYPL": "PayPal Holdings Inc.",
    "SQ": "Block Inc.",
    "GM": "General Motors Co.",
    "F": "Ford Motor Co.",
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications Inc.",
    "CMCSA": "Comcast Corp.",
    "PFE": "Pfizer Inc.",
    "JNJ": "Johnson & Johnson",
    "MRK": "Merck & Co.",
    "ABBV": "AbbVie Inc.",
    "BMY": "Bristol-Myers Squibb Co.",
    "LLY": "Eli Lilly & Co.",
    "UNH": "UnitedHealth Group Inc.",
    "CVS": "CVS Health Corp.",
    "WMT": "Walmart Inc.",
    "TGT": "Target Corp.",
    "COST": "Costco Wholesale Corp.",
    "HD": "Home Depot Inc.",
    "LOW": "Lowe's Companies Inc.",
    "NKE": "Nike Inc.",
    "SBUX": "Starbucks Corp.",
    "MCD": "McDonald's Corp.",
    "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.",
    "PG": "Procter & Gamble Co.",
    "CL": "Colgate-Palmolive Co.",
    "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corp.",
    "COP": "ConocoPhillips",
    "OXY": "Occidental Petroleum Corp.",
    "SLB": "Schlumberger Ltd.",
    "HAL": "Halliburton Co.",
    "CAT": "Caterpillar Inc.",
    "DE": "Deere & Co.",
    "MMM": "3M Co.",
    "HON": "Honeywell International Inc.",
    "GE": "General Electric Co.",
    "RTX": "Raytheon Technologies Corp.",
    "LMT": "Lockheed Martin Corp.",
    "NOC": "Northrop Grumman Corp.",
    "GD": "General Dynamics Corp.",
    "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corp.",
    "IBM": "IBM Corp.",
    "CSCO": "Cisco Systems Inc.",
    "ADBE": "Adobe Inc.",
    "NOW": "ServiceNow Inc.",
    "SNOW": "Snowflake Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "ZM": "Zoom Video Communications Inc.",
    "UBER": "Uber Technologies Inc.",
    "LYFT": "Lyft Inc.",
    "ABNB": "Airbnb Inc.",
    "DASH": "DoorDash Inc.",
    "COIN": "Coinbase Global Inc.",
    "HOOD": "Robinhood Markets Inc.",
    "RIVN": "Rivian Automotive Inc.",
    "LCID": "Lucid Group Inc.",
    "NIO": "NIO Inc.",
    "BABA": "Alibaba Group Holdings Ltd.",
    "JD": "JD.com Inc.",
    "PDD": "PDD Holdings Inc.",
    "TSM": "Taiwan Semiconductor Manufacturing Co.",
    "SONY": "Sony Group Corp.",
    "TM": "Toyota Motor Corp.",
    "SPOT": "Spotify Technology SA",
    "TWTR": "Twitter Inc.",
    "SNAP": "Snap Inc.",
    "PINS": "Pinterest Inc.",
    "SHOP": "Shopify Inc.",
    "SE": "Sea Limited",
    "ROKU": "Roku Inc.",
    "SIX": "Six Flags Entertainment Corp.",
    "BKS": "Barnes & Noble Inc.",
    "SPY": "S&P 500",
    "QQQ": "NASDAQ",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000",
    "USO": "Crude Oil",
    "GLD": "Gold",
    "SLV": "Silver",
    "WBA": "Walgreens Boots Alliance Inc.",
    "AGN": "Allergan plc",
    "ENDP": "Endo International plc",
    "GEVO": "Gevo Inc.",
    "ATVI": "Activision Blizzard Inc.",
    "EA": "Electronic Arts Inc.",
    "ERTS": "Electronic Arts Inc.",
    "BEP": "Brookfield Renewable Partners LP",
    "SSL": "Sasol Ltd.",
    "HES": "Hess Corp.",
    "SD": "SandRidge Energy Inc.",
    "MER": "Merrill Lynch",
    "CMTL": "Comtech Telecommunications Corp.",
    "NREF": "NexPoint Real Estate Finance Inc.",
    "BLK": "BlackRock Inc.",
    "COF": "Capital One Financial Corp.",
    "ADP": "Automatic Data Processing Inc.",
    "AUVI": "Applied UV Inc.",
    "EEENF": "88 Energy Ltd.",
    "LTNC": "L-Town Cannabis Inc.",
    "CERPQ": "CereProcTech",
    "GYST": "Gyst Audio Inc.",
    "SEAC": "SeaChange International Inc.",
    "RBA": "RB Global Inc.",
    "CAT": "Caterpillar Inc.",
    "DAKT": "Daktronics Inc.",
    "AYI": "Acuity Brands Inc.",
    "QCOM": "Qualcomm Inc.",
    "AVGO": "Broadcom Inc.",
    "WRK": "WestRock Co.",
    "SLE": "Sara Lee Corp.",
    "PCL": "Plum Creek Timber Co.",
    "SBPH": "Spring Bioscience Corp.",
    "DNO": "DNO ASA",
    "TLRY": "Tilray Inc.",
    "COX": "COX",
    "AM": "Antero Midstream Corp.",
    "BRK": "Berkshire Hathaway Inc.",
    "AMGN": "Amgen Inc.",
    "LNC": "Lincoln National Corp.",
    "EFX": "Equifax Inc.",
    "MBI": "MBIA Inc.",
    "MBIA": "MBIA Inc.",
    "BAM": "Brookfield Asset Management Inc.",
    "WISH": "ContextLogic Inc.",
    "DVN": "Devon Energy Corp.",
    "TROW": "T. Rowe Price Group Inc.",
    "SJW": "SJW Group",
    "APD": "Air Products & Chemicals Inc.",
    "EMC": "EMC Corp.",
    "DG": "Dell Technologies Inc.",
    "LU": "Lucent Technologies Inc.",
    "JNPR": "Juniper Networks Inc.",
    "CSNA3": "Ciena Corp.",
    "CIEN": "Ciena Corp.",
    "MERL": "Meralco",
    "SAP": "SAP SE",
    "SGP": "Schering-Plough Corp.",
    "MSGE": "Mediaset Espana",
    "MFG": "MF Global",
    "JEF": "Jefferies Group Inc.",
    "PBI": "Pitney Bowes Inc.",
    "LVN": "Live Nation Entertainment Inc.",
    "LYV": "Live Nation Entertainment Inc.",
    "MO": "Altria Group Inc.",
    "GOL": "Gol Linhas Aereas Inteligentes SA",
    "RACE": "Ferrari N.V.",
    "PARA": "Paramount Global",
    "NWSA": "News Corp.",
    "NWS": "News Corp.",
    "FOX": "Fox Corp.",
    "FOXA": "Fox Corp.",
    "SHW": "Sherwin-Williams Co.",
}

# ─── Company name patterns (name → canonical) ────────────────────────────────
COMPANY_PATTERNS = [
    # Tech
    (r'\bapple\b', "Apple Inc."),
    (r'\bamazon\b', "Amazon.com Inc."),
    (r'\bgoogle\b', "Alphabet Inc."),
    (r'\balphabet\b', "Alphabet Inc."),
    (r'\bmicrosoft\b', "Microsoft Corp."),
    (r'\btesla\b', "Tesla Inc."),
    (r'\bfacebook\b', "Meta Platforms Inc."),
    (r'\bmeta platforms\b', "Meta Platforms Inc."),
    (r'\bnetflix\b', "Netflix Inc."),
    (r'\bnvidia\b', "NVIDIA Corp."),
    (r'\bamd\b', "Advanced Micro Devices Inc."),
    (r'\badvanced micro\b', "Advanced Micro Devices Inc."),
    (r'\bintel\b(?!\s*(?:ligent|lectual|rior|rest|rnation|grat|nsive|nded|ractive))', "Intel Corp."),
    (r'\bboeing\b', "Boeing Co."),
    (r'\bdisney\b', "Walt Disney Co."),
    (r'\bwalmart\b', "Walmart Inc."),
    (r'\bwal-mart\b', "Walmart Inc."),
    (r'\bcostco\b', "Costco Wholesale Corp."),
    (r'\bhome depot\b', "Home Depot Inc."),
    (r'\bnike\b', "Nike Inc."),
    (r'\bstarbucks\b', "Starbucks Corp."),
    (r'\bmcdonald\b', "McDonald's Corp."),
    (r'\bcoca-cola\b', "Coca-Cola Co."),
    (r'\bcoca cola\b', "Coca-Cola Co."),
    (r'\bpepsi\b', "PepsiCo Inc."),
    (r'\bpepsico\b', "PepsiCo Inc."),
    (r'\bprocter\s*[&and]*\s*gamble\b', "Procter & Gamble Co."),
    (r'\bsalesforce\b', "Salesforce Inc."),
    (r'\boracle\b', "Oracle Corp."),
    (r'\bibm\b', "IBM Corp."),
    (r'\bcisco\b', "Cisco Systems Inc."),
    (r'\badobe\b', "Adobe Inc."),
    (r'\bpalantir\b', "Palantir Technologies Inc."),
    (r'\bzoom\b', "Zoom Video Communications Inc."),
    (r'\buber\b', "Uber Technologies Inc."),
    (r'\blyft\b', "Lyft Inc."),
    (r'\bairbnb\b', "Airbnb Inc."),
    (r'\brobinhood\b', "Robinhood Markets Inc."),
    (r'\brivian\b', "Rivian Automotive Inc."),
    (r'\blucid\s*(?:group|motors)\b', "Lucid Group Inc."),
    (r'\bnio\b', "NIO Inc."),
    (r'\balibaba\b', "Alibaba Group Holdings Ltd."),
    (r'\bshopify\b', "Shopify Inc."),
    (r'\bspotify\b', "Spotify Technology SA"),
    (r'\btwitter\b', "Twitter Inc."),
    (r'\bsnap\s*(?:chat|inc)\b', "Snap Inc."),
    (r'\bpinterest\b', "Pinterest Inc."),
    (r'\broku\b', "Roku Inc."),
    (r'\bebay\b', "eBay Inc."),
    (r'\bpaypal\b', "PayPal Holdings Inc."),
    (r'\bcoinbase\b', "Coinbase Global Inc."),
    (r'\bsquare\b(?!\s*(?:feet|foot|mile|meter|inch|yard))', "Block Inc."),
    (r'\bblock\s+inc\b', "Block Inc."),

    # Finance / Banks
    (r'\bjpmorgan\b', "JPMorgan Chase & Co."),
    (r'\bjp\s*morgan\b', "JPMorgan Chase & Co."),
    (r'\bgoldman\s*sachs\b', "Goldman Sachs Group Inc."),
    (r'\bmorgan\s+stanley\b', "Morgan Stanley"),
    (r'\bbank\s+of\s+america\b', "Bank of America Corp."),
    (r'\bwells\s+fargo\b', "Wells Fargo & Co."),
    (r'\bcitigroup\b', "Citigroup Inc."),
    (r'\bcitibank\b', "Citigroup Inc."),
    (r'\bciti\b(?!\s*(?:es|zen|ng))', "Citigroup Inc."),
    (r'\bvisa\b(?!\s*(?:versa|vis))', "Visa Inc."),
    (r'\bmastercard\b', "Mastercard Inc."),
    (r'\bblackrock\b', "BlackRock Inc."),
    (r'\bberkshire\s*hathaway\b', "Berkshire Hathaway Inc."),
    (r'\bwarren\s+buffett\b', "Berkshire Hathaway Inc."),
    (r'\bcapital\s+one\b', "Capital One Financial Corp."),
    (r'\bcharles\s+schwab\b', "Charles Schwab Corp."),
    (r'\bmerrill\s+lynch\b', "Merrill Lynch"),
    (r'\bcredit\s+suisse\b', "Credit Suisse Group AG"),
    (r'\bubs\b(?!\s)', "UBS Group AG"),
    (r'\bdeutsche\s+bank\b', "Deutsche Bank AG"),
    (r'\bhsbc\b', "HSBC Holdings plc"),
    (r'\bbarclays\b', "Barclays plc"),
    (r'\bstate\s+street\b', "State Street Corp."),
    (r'\bt\.\s*rowe\s+price\b', "T. Rowe Price Group Inc."),
    (r'\bsuntrust\b', "SunTrust Banks Inc."),

    # Healthcare / Pharma
    (r'\bpfizer\b', "Pfizer Inc."),
    (r'\bjohnson\s+(?:&|and)\s+johnson\b', "Johnson & Johnson"),
    (r'\bj\s*&\s*j\b', "Johnson & Johnson"),
    (r'\bmerck\b', "Merck & Co."),
    (r'\babbvie\b', "AbbVie Inc."),
    (r'\bbristol[\s-]*myers\b', "Bristol-Myers Squibb Co."),
    (r'\beli\s+lilly\b', "Eli Lilly & Co."),
    (r'\blilly\b(?=.*(?:pharma|drug|insulin))', "Eli Lilly & Co."),
    (r'\bunitedhealth\b', "UnitedHealth Group Inc."),
    (r'\bcvs\s+health\b', "CVS Health Corp."),
    (r'\bamgen\b', "Amgen Inc."),
    (r'\bgilead\b', "Gilead Sciences Inc."),
    (r'\bmedimmune\b', "MedImmune Inc."),
    (r'\bmedtronic\b', "Medtronic plc"),
    (r'\bschering[\s-]*plough\b', "Schering-Plough Corp."),

    # Energy
    (r'\bexxon\s*mobil\b', "Exxon Mobil Corp."),
    (r'\bexxon\b', "Exxon Mobil Corp."),
    (r'\bchevron\b', "Chevron Corp."),
    (r'\bconocophillips\b', "ConocoPhillips"),
    (r'\boccidental\s+petroleum\b', "Occidental Petroleum Corp."),
    (r'\bschlumberger\b', "Schlumberger Ltd."),
    (r'\bhalliburton\b', "Halliburton Co."),
    (r'\bbp\b(?=\s|$|\')', "BP plc"),
    (r'\bshell\b(?=\s+(?:was|is|has|had|will|share|stock|oil|gas|petrol|energy))', "Shell plc"),
    (r'\broyal\s+dutch\s+shell\b', "Shell plc"),
    (r'\bdevon\s+energy\b', "Devon Energy Corp."),
    (r'\bsasol\b', "Sasol Ltd."),
    (r'\bxstrata\b', "Xstrata plc"),
    (r'\bglencore\b', "Glencore plc"),
    (r'\bdno\s+(?:asa|hike)\b', "DNO ASA"),
    (r'\btilray\b', "Tilray Inc."),
    (r'\bseanergy\b', "Seanergy Maritime Holdings Corp."),

    # Auto
    (r'\bgeneral\s+motors\b', "General Motors Co."),
    (r'\bford\s+motor\b', "Ford Motor Co."),
    (r'\btoyota\b', "Toyota Motor Corp."),
    (r'\bnissan\b', "Nissan Motor Co."),
    (r'\brenault\b', "Renault SA"),
    (r'\bvolkswagen\b', "Volkswagen AG"),
    (r'\baudi\b', "Audi AG"),
    (r'\bmaruti\s+suzuki\b', "Maruti Suzuki India Ltd."),

    # Industrials / Defense
    (r'\bcaterpillar\b', "Caterpillar Inc."),
    (r'\bdeere\b(?!\s+&\s+company)', "Deere & Co."),
    (r'\b3m\b', "3M Co."),
    (r'\bhoneywell\b', "Honeywell International Inc."),
    (r'\bgeneral\s+electric\b', "General Electric Co."),
    (r'\braytheon\b', "Raytheon Technologies Corp."),
    (r'\blockheed\s+martin\b', "Lockheed Martin Corp."),
    (r'\bnorthrop\s+grumman\b', "Northrop Grumman Corp."),
    (r'\bgeneral\s+dynamics\b', "General Dynamics Corp."),
    (r'\bqualcomm\b', "Qualcomm Inc."),
    (r'\bbroadcom\b', "Broadcom Inc."),

    # Telecom / Media
    (r'\bat&t\b', "AT&T Inc."),
    (r'\bverizon\b', "Verizon Communications Inc."),
    (r'\bcomcast\b', "Comcast Corp."),
    (r'\bactivision\s*blizzard\b', "Activision Blizzard Inc."),
    (r'\bactivision\b', "Activision Blizzard Inc."),
    (r'\belectronic\s+arts\b', "Electronic Arts Inc."),
    (r'\bnews\s+corp\b', "News Corp."),
    (r'\bfox\s+(?:corp|news|broadcasting)\b', "Fox Corp."),
    (r'\blive\s+nation\b', "Live Nation Entertainment Inc."),
    (r'\bticketmaster\b', "Live Nation Entertainment Inc."),
    (r'\bdish\s+network\b', "Dish Network Corp."),
    (r'\bcox\s+media\b', "Cox Media Group"),

    # Retail / Consumer
    (r'\bwalgreens\b', "Walgreens Boots Alliance Inc."),
    (r'\brite\s+aid\b', "Rite Aid Corp."),
    (r'\btarget\b(?=\s+(?:corp|stock|share|price|store|retail))', "Target Corp."),
    (r'\bgamestop\b', "GameStop Corp."),
    (r'\bgreene\s+king\b', "Greene King plc"),
    (r'\bsix\s+flags\b', "Six Flags Entertainment Corp."),
    (r'\bbarnes\s*(?:&|and)\s*noble\b', "Barnes & Noble Inc."),
    (r'\bvf\s+corp\b', "VF Corp."),
    (r'\bvans\s+parent\b', "VF Corp."),
    (r'\bcoles\s+group\b', "Coles Group Ltd."),
    (r'\bdreamworks\b', "DreamWorks"),
    (r'\bdassault\s+systemes\b', "Dassault Systemes SE"),
    (r'\bsteinhoff\b', "Steinhoff International Holdings"),
    (r'\bbrait\b', "Brait SE"),

    # Tech services / Software
    (r'\bsap\b(?=\s+(?:s/4|ariba|success|solution|partner))', "SAP SE"),
    (r'\batlassian\b', "Atlassian Corp."),
    (r'\bcyient\b', "Cyient Ltd."),
    (r'\bkpit\b', "KPIT Technologies Ltd."),
    (r'\bmindtree\b', "Mindtree Ltd."),
    (r'\bpersistent\s+systems\b', "Persistent Systems Ltd."),
    (r'\bcoforge\b', "Coforge Ltd."),
    (r'\bnavisite\b', "Navisite Inc."),
    (r'\bcomtech\b', "Comtech Telecommunications Corp."),
    (r'\bdaktronics\b', "Daktronics Inc."),
    (r'\bavaya\b', "Avaya Holdings Corp."),
    (r'\bemc\b(?=\s|\')', "EMC Corp."),
    (r'\bsitel\b', "Sitel Corp."),
    (r'\blightstream\b', "LightStream"),
    (r'\bepicor\b', "Epicor Software Corp."),

    # Insurance / Financial Services
    (r'\blincoln\b(?=\s+(?:national|financial|also|re\b))', "Lincoln National Corp."),
    (r'\bequifax\b', "Equifax Inc."),
    (r'\bmbia\b', "MBIA Inc."),
    (r'\bageas\b', "Ageas SA"),
    (r'\balaris\s+royalty\b', "Alaris Equity Partners Income Trust"),
    (r'\bsilvercorp\b', "Silvercorp Metals Inc."),
    (r'\bsilver\s+standard\b', "Silver Standard Resources Inc."),
    (r'\bbrookfield\s+asset\b', "Brookfield Asset Management Inc."),
    (r'\bbrookfield\s+renewable\b', "Brookfield Renewable Partners LP"),
    (r'\bbrookfield\b', "Brookfield Asset Management Inc."),
    (r'\belliott\s+management\b', "Elliott Management Corp."),
    (r'\bnexpoint\b', "NexPoint Real Estate Finance Inc."),
    (r'\baltice\b', "Altice USA Inc."),
    (r'\bjefferies\b', "Jefferies Group Inc."),
    (r'\bpitney\b', "Pitney Bowes Inc."),
    (r'\bpitney\s+bowes\b', "Pitney Bowes Inc."),
    (r'\bwestrock\b', "WestRock Co."),
    (r'\bpanoro\s+energy\b', "Panoro Energy ASA"),
    (r'\bmf\s+global\b', "MF Global Holdings Ltd."),

    # Airlines / Travel
    (r'\bgol\s+linhas\b', "Gol Linhas Aereas Inteligentes SA"),
    (r'\bhilton\b', "Hilton Worldwide Holdings Inc."),
    (r'\bmeralco\b', "Manila Electric Co."),
    (r'\bexpedia\b', "Expedia Group Inc."),
    (r'\bquebecor\b', "Quebecor Inc."),
    (r'\brogers\b(?=.*shaw)', "Rogers Communications Inc."),
    (r'\bshaw\b(?=.*rogers)', "Shaw Communications Inc."),

    # Misc
    (r'\baltria\b', "Altria Group Inc."),
    (r'\bnjoy\b', "NJOY Holdings Inc."),
    (r'\bjuul\b', "Juul Labs Inc."),
    (r'\blockheed\b', "Lockheed Martin Corp."),
    (r'\bair\s+products\b', "Air Products & Chemicals Inc."),
    (r'\bplum\s+creek\b', "Plum Creek Timber Co."),
    (r'\btrican\b', "Trican Well Service Ltd."),
    (r'\bmedias[e]t\s+espana\b', "Mediaset Espana Comunicacion SA"),
    (r'\bpacific\s+star\b', "Pacific Star Network Ltd."),
    (r'\bsix\s+flags\b', "Six Flags Entertainment Corp."),
    (r'\bbank\s+of\s+baroda\b', "Bank of Baroda"),
    (r'\bbank\s+of\s+maharashtra\b', "Bank of Maharashtra"),
    (r'\bindian\s+overseas\s+bank\b', "Indian Overseas Bank"),
    (r'\bpunjab\s+(?:&|and)\s+sind\b', "Punjab & Sind Bank"),
    (r'\bcentral\s+bank\s+of\s+india\b', "Central Bank of India"),
    (r'\buco\s+bank\b', "UCO Bank"),
    (r'\btpg[\s-]*axon\b', "TPG-Axon Capital"),
    (r'\bhmt\b(?=\'s)', "Hersha Hospitality Trust"),
    (r'\boculus\b', "Meta Platforms Inc."),
    (r'\bwish\s+faced\b', "ContextLogic Inc."),
    (r'\bwish\b(?=\s+(?:stock|share|app))', "ContextLogic Inc."),
    (r'\btejon\s+ranch\b', "Tejon Ranch Co."),
    (r'\bcarmike\b', "Carmike Cinemas Inc."),
    (r'\bdairy\s+queen\b', "Berkshire Hathaway Inc."),
    (r'\blyrica\b', "Pfizer Inc."),
    (r'\bvytorin\b', "Schering-Plough Corp."),
    (r'\bzocor\b', "Merck & Co."),
    (r'\binterscope\s+records\b', "Universal Music Group"),
    (r'\bumg\b', "Universal Music Group"),
    (r'\bgevo\b', "Gevo Inc."),
    (r'\biea\b', "International Energy Agency"),
    (r'\belon\s+musk\b', "Tesla Inc."),
    (r'\bmusk\b(?=.*(?:tesla|tweet|twitter|spacex|concern|announce))', "Tesla Inc."),
    (r'\bspielberg\b', "DreamWorks"),
    (r'\bsolomon\b(?=.*barney)', "Citigroup Inc."),
    (r'\bsalomon\s+smith\b', "Citigroup Inc."),
    (r'\bnelnet\b', "Nelnet Inc."),
    (r'\bds\b(?=\s*(?:report|earning))', "Dassault Systemes SE"),
]

# ─── Index / Market patterns ─────────────────────────────────────────────────
INDEX_PATTERNS = [
    (r'\bs\s*&\s*p\s*500\b', "S&P 500"),
    (r'\bsp500\b', "S&P 500"),
    (r'\bnasdaq\b', "NASDAQ"),
    (r'\bdow\s+jones\b', "Dow Jones"),
    (r'\bdjia\b', "Dow Jones"),
    (r'\brussell\s*2000\b', "Russell 2000"),
    (r'\bftse\b', "FTSE"),
    (r'\bnikkei\b', "Nikkei 225"),
    (r'\bhang\s+seng\b', "Hang Seng Index"),
    (r'\bdax\b', "DAX"),
]

# ─── Commodity patterns ──────────────────────────────────────────────────────
COMMODITY_PATTERNS = [
    (r'\bcrude\s+oil\b', "Crude Oil"),
    (r'\boil\s+(?:price|production|demand|supply|market|export|futures|slid|fell|drop|rose|surge|sank|rally)\b', "Crude Oil"),
    (r'\b(?:brent|wti)\s+(?:crude|oil)\b', "Crude Oil"),
    (r'\bgold\s+(?:price|futures|mine|mining|miner|bullion|bar|spot|rally)\b', "Gold"),
    (r'\bnatural\s+gas\b', "Natural Gas"),
    (r'\bcopper\s+(?:price|futures|mine|mining)\b', "Copper"),
    (r'\bsilver\s+(?:price|futures|bullion)\b', "Silver"),
]

# ─── Central bank patterns ───────────────────────────────────────────────────
CENTRAL_BANK_PATTERNS = [
    (r'\bfederal\s+reserve\b', "Federal Reserve"),
    (r'\bthe\s+fed\b', "Federal Reserve"),
    (r'\bfed\s+(?:report|rate|hike|cut|meeting|chair|decision|policy|announce|raise|lower)\b', "Federal Reserve"),
    (r'\b#FedReport\b', "Federal Reserve"),
    (r'\becb\b', "ECB"),
    (r'\beuropean\s+central\s+bank\b', "ECB"),
    (r'\bbank\s+of\s+(?:england|japan|canada)\b', lambda m: f"Bank of {m.group(0).split('of ')[1].title()}"),
    (r'\bpeople\'?s?\s+bank\s+of\s+china\b', "People's Bank of China"),
]

# ─── Sector patterns ─────────────────────────────────────────────────────────
SECTOR_PATTERNS = [
    (r'\btechnology\s+sector\b', "Technology"),
    (r'\btech\s+(?:sector|stocks|industry|companies)\b', "Technology"),
    (r'\bhealthcare\s+(?:sector|stocks|industry)\b', "Healthcare"),
    (r'\benergy\s+(?:sector|stocks|industry)\b', "Energy"),
    (r'\bfinancial\s+(?:sector|stocks|industry)\b', "Financials"),
    (r'\bbank(?:ing)?\s+(?:sector|stocks|industry)\b', "Financials"),
    (r'\breal\s+estate\s+(?:sector|stocks|industry)\b', "Real Estate"),
    (r'\boil\s+and\s+gas\s+(?:sector|industry|space)\b', "Energy"),
    (r'\bcable\s+(?:company|industry|sector)\b', "Communications"),
    (r'\binsurance\s+(?:sector|industry|companies)\b', "Insurance"),
    (r'\bretail\s+(?:sector|industry|space)\b', "Consumer Discretionary"),
]


def extract_ticker(text):
    """Extract $TICKER patterns from text."""
    tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    # Filter out common non-ticker patterns
    noise = {'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CNY', 'INR',
             'OIL', 'WTI', 'ETH', 'BTC', 'THE', 'FOR', 'AND', 'BUT',
             'NOT', 'ARE', 'WAS', 'HAS', 'HAD', 'HIS', 'HER', 'OUR',
             'ALL', 'ANY', 'ITS'}
    tickers = [t for t in tickers if t not in noise]
    return tickers


def identify_entity(text):
    """
    Identify the primary financial entity in text.
    Returns canonical entity name.
    """
    if not isinstance(text, str):
        return "NONE"

    text_lower = text.lower()

    # 1. Check for explicit $TICKER references first
    tickers = extract_ticker(text)
    if tickers:
        # Use the first recognized ticker
        for t in tickers:
            if t in TICKER_MAP:
                return TICKER_MAP[t]
        # If ticker not in map, return as $TICKER
        if tickers[0] == "OIL":
            return "Crude Oil"
        return f"${tickers[0]}"

    # 2. Check company name patterns
    for pattern, name in COMPANY_PATTERNS:
        if re.search(pattern, text_lower):
            if callable(name):
                m = re.search(pattern, text_lower)
                return name(m)
            return name

    # 3. Check index patterns
    for pattern, name in INDEX_PATTERNS:
        if re.search(pattern, text_lower):
            return name

    # 4. Check commodity patterns
    for pattern, name in COMMODITY_PATTERNS:
        if re.search(pattern, text_lower):
            return name

    # 5. Check central bank patterns
    for pattern, name in CENTRAL_BANK_PATTERNS:
        if re.search(pattern, text_lower):
            m = re.search(pattern, text_lower)
            if callable(name):
                return name(m)
            return name

    # 6. Check sector patterns
    for pattern, name in SECTOR_PATTERNS:
        if re.search(pattern, text_lower):
            return name

    # 7. Check for OPEC
    if re.search(r'\bopec\b', text_lower):
        return "OPEC"

    # 8. Check for general market references
    market_terms = [
        r'\bstock\s*market\b', r'\bmarket\s+(?:crash|rally|correction|downturn|rebound|sell[\s-]*off|cap)',
        r'\b(?:bull|bear)\s+market\b', r'\bmarkets?\s+(?:collapsed|tumbled|soared|surged|plunged|rallied)\b',
        r'\b(?:stocks|equities)\s+(?:fell|rose|dropped|climbed|surged|tumbled|rallied)\b',
        r'\bwall\s+street\b', r'\b#stocks\b',
        r'\bmarket\s+(?:volatility|sentiment|conditions|performance|outlook)\b',
    ]
    for pat in market_terms:
        if re.search(pat, text_lower):
            return "MARKET"

    # 9. Check for crypto
    crypto_patterns = [
        (r'\bbitcoin\b', "Bitcoin"),
        (r'\bethereum\b', "Ethereum"),
        (r'\bcrypto(?:currency)?\b', "Cryptocurrency"),
    ]
    for pat, name in crypto_patterns:
        if re.search(pat, text_lower):
            return name

    # 10. Fallback: check for any remaining company-ish references
    # Check for possessives like "Company's" or "the company"
    if re.search(r'\bthe\s+company\b', text_lower) and len(text) > 50:
        return "NONE"  # Generic company reference, can't determine which one

    return "NONE"


def determine_entity_sentiment(text, label, entity):
    """
    Determine sentiment toward the specific entity.
    Usually matches the label, but can differ in multi-entity texts.
    """
    if entity == "NONE" or entity == "MARKET":
        return label

    # For most cases, the sentence-level sentiment applies to the entity
    return label


def process_batch(batch_num):
    """Process a single batch file."""
    input_path = os.path.join(BASE_DIR, f"batch_{batch_num:04d}_input.csv")
    output_path = os.path.join(BASE_DIR, f"batch_{batch_num:04d}_output.csv")

    df = pd.read_csv(input_path)
    idx_col = df.columns[0]  # 'Unnamed: 0'

    results = []
    for _, row in df.iterrows():
        text = row['text'] if isinstance(row['text'], str) else ""
        label = row['label'] if isinstance(row['label'], str) else "NEUTRAL"

        entity = identify_entity(text)
        entity_sentiment = determine_entity_sentiment(text, label, entity)

        results.append({
            idx_col: row[idx_col],
            'entity': entity,
            'entity_sentiment': entity_sentiment,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    return len(results)


if __name__ == "__main__":
    total = 0
    for batch_num in range(135, 162):
        n = process_batch(batch_num)
        total += n
        print(f"Batch {batch_num:04d}: {n} rows processed -> batch_{batch_num:04d}_output.csv")
    print(f"\nDone! Total: {total} rows across {162-135} batches.")
