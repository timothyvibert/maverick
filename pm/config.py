"""Module-wide configuration."""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
# V1 reads the holdings extract (`<source>_extract_YYYYMMDD_HHMMSS.xlsx`) discovered by
# `pm.ingest.extract_loader.find_latest_extract`. There is no single
# canonical filename anymore.
EXTRACT_DATA_DIR = DATA_DIR

# ---------- Server ----------
HOST = "127.0.0.1"
PORT = 8062

# ---------- Rates ----------
# BBG-off fallback short rate for carry-on-strike in the early-exercise check on
# deep-ITM short puts. When Bloomberg is up, the live treasury curve (picked per
# leg DTE) is used instead and this scalar is unused; when it is used, the
# management fire labels the carry as estimated.
DEFAULT_RISK_FREE_RATE = 0.04

# ---------- Daily-move alert thresholds ----------
ALERT_MULTIPLIER = 2.0
TRADING_DAYS_PER_YEAR = 252

# ---------- Option contract multiplier ----------
# US listed equity options + Eurex single-stock options both use 100. Index
# options can differ but they don't appear in the holdings file.
OPTION_CONTRACT_MULTIPLIER = 100

# ---------- Non-US underlying detection ----------
NON_US_STYLE_VALUES = {"Eurozone", "Europe", "UK", "Switzerland", "Japan"}
# CINS CUSIPs start with a letter (CINS country code). US CUSIPs start with a digit.
NON_US_CINS_PREFIXES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NON_US_CURRENCY_TOKENS = (" EUR", " GBP", " GBp", " CHF", " JPY")

# Hardcoded Bloomberg exchange suffix map for known non-US underlyings.
# Keyed by the symbol as it appears in the holdings 'Symbol' column.
# The map is a SEED — extend by adding entries as new names appear in books.
# When a non-US position's symbol is NOT in this map, the parser routes it to
# other_positions with manual_review_reason='non_us_unmapped_exchange' and a
# warning, so unmapped names are visible.
#
# Notes on conflicts:
# - 'SAN' resolves to Sanofi (FP). For Banco Santander use 'SAN_ES' in the
#   Symbol column or extend the parser later to disambiguate via Style/CUSIP.
# - 'G' resolves to Generali (IM). Similar caveat.
EXCHANGE_SUFFIX_MAP: dict[str, str] = {
    # Italy — Borsa Italiana
    "RACE": "IM", "ENI": "IM", "ENEL": "IM", "ISP": "IM", "UCG": "IM",
    "STLA": "IM", "G": "IM", "MB": "IM", "PRY": "IM", "TIT": "IM",

    # Germany — XETRA
    "SAP": "GY", "SIE": "GY", "ALV": "GY", "BAS": "GY", "BAYN": "GY",
    "BMW": "GY", "DBK": "GY", "DTE": "GY", "IFX": "GY", "MUV2": "GY",
    "RWE": "GY", "VOW3": "GY", "ADS": "GY", "DHL": "GY", "AIR": "GY",
    "MBG": "GY", "P911": "GY", "HEN3": "GY", "DB1": "GY", "EOAN": "GY",

    # Netherlands — Euronext Amsterdam
    "ASML": "NA", "AD": "NA", "INGA": "NA", "KPN": "NA", "PHIA": "NA",
    "UNA": "NA", "WKL": "NA", "RAND": "NA", "REN": "NA", "PRX": "NA",
    "STLA_NL": "NA",  # Stellantis EU listing if separate

    # Switzerland — SIX Swiss Exchange
    "NESN": "SW", "NOVN": "SW", "ROG": "SW", "ABBN": "SW", "UBSG": "SW",
    "ZURN": "SW", "GIVN": "SW", "SIKA": "SW", "CFR": "SW", "ALC": "SW",
    "LONN": "SW", "HOLN": "SW", "GEBN": "SW",

    # UK — London Stock Exchange
    "VOD": "LN", "BARC": "LN", "HSBA": "LN", "BP": "LN", "GSK": "LN",
    "AZN": "LN", "ULVR": "LN", "RIO": "LN", "LLOY": "LN", "BATS": "LN",
    "DGE": "LN", "SHEL": "LN", "NWG": "LN", "TSCO": "LN", "PRU": "LN",
    "REL": "LN", "RKT": "LN", "BHP": "LN", "AAL": "LN", "NG": "LN",

    # France — Euronext Paris
    "AI": "FP", "BNP": "FP", "CS": "FP", "DG": "FP", "EL": "FP",
    "MC": "FP", "OR": "FP", "SAN": "FP", "SU": "FP", "TTE": "FP",
    "BN": "FP", "VIV": "FP", "RMS": "FP", "KER": "FP", "GLE": "FP",
    "AC": "FP", "ENGI": "FP", "CAP": "FP", "ML": "FP", "ATO": "FP",

    # Spain — BME (Bolsa de Madrid)
    "BBVA": "SM", "IBE": "SM", "ITX": "SM", "REP": "SM", "TEF": "SM",
    "ACS": "SM", "AENA": "SM", "FER": "SM",

    # Sweden — Nasdaq Stockholm
    "VOLVB": "SS", "ERICB": "SS", "ATCOA": "SS", "HEXAB": "SS",
    "INVEB": "SS", "ESSITYB": "SS", "SAND": "SS", "SHBA": "SS",

    # Denmark — Nasdaq Copenhagen
    "NOVOB": "DC", "MAERSKB": "DC", "DSV": "DC", "ORSTED": "DC",

    # Belgium — Euronext Brussels
    "UCB": "BB", "ABI": "BB", "KBC": "BB", "SOLB": "BB",

    # Norway — Oslo Børs
    "EQNR": "NO", "DNB": "NO", "TEL": "NO", "MOWI": "NO",

    # Finland — Nasdaq Helsinki
    "NOKIA": "FH", "NESTE": "FH", "KNEBV": "FH",

    # Ireland — Euronext Dublin
    "RYA": "ID", "CRH": "ID", "KSP": "ID",
}
