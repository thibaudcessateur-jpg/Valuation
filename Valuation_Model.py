import os
import math
import json
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st

# =========================================
# CONFIG DE BASE
# =========================================

EODHD_BASE_URL = "https://eodhd.com/api"
RISK_FREE_CACHE_FILE = "risk_free_cache.json"
RISK_FREE_CACHE_MAX_DAYS = 180


# =========================================
# FONCTIONS UTILITAIRES API
# =========================================

def get_api_key():
    """
    1) Essaie EODHD_API_KEY dans les variables d'environnement
    2) Sinon, demande à l'utilisateur en sidebar
    """
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("EODHD API Key", type="password")
    return api_key


def search_instrument(query: str, api_key: str, limit: int = 10):
    """
    Recherche un instrument par nom ou ticker via l'API EODHD.
    Endpoint : /search/{query}
    Retourne une liste de dicts avec au moins Code, Exchange, Name, Country, Currency.
    """
    url = f"{EODHD_BASE_URL}/search/{query}"
    params = {
        "api_token": api_key,
        "limit": limit,
        "fmt": "json",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def resolve_best_ticker(search_results, api_key, preferred_country=None):
    """
    Sélectionne le ticker le plus pertinent pour une big cap en combinant :
    1) priorité au pays d'origine
    2) priorité aux exchanges majeurs
    3) market cap (price * shares)
    """
    if not search_results:
        return None

    valid_exchanges = {"PA", "XETRA", "NASDAQ", "NYSE", "LSE", "AMS", "MIL", "SW", "BRU", "STO"}

    scored = []

    for item in search_results:
        if item.get("Type") and item.get("Type") != "Common Stock":
            continue

        exchange = item.get("Exchange")
        if exchange and exchange not in valid_exchanges:
            continue

        ticker = build_ticker_from_search_result(item)
        if not ticker:
            continue

        try:
            fundamentals = fetch_fundamentals(ticker, api_key)
            shares = get_shares_outstanding(fundamentals)
            price = fetch_eod_price(ticker, api_key)

            if not shares or not price:
                continue

            market_cap = float(shares) * float(price)

            score = 0

            # 1️⃣ Priorité pays
            country = fundamentals.get("General", {}).get("Country")
            if preferred_country and country == preferred_country:
                score += 100

            # 2️⃣ Bonus exchange principal
            if exchange in {"PA", "NYSE", "NASDAQ", "LSE"}:
                score += 50

            # 3️⃣ Market cap (log pour éviter qu’elle écrase tout)
            score += math.log10(market_cap)

            scored.append((ticker, score, country, exchange))

        except Exception:
            continue

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def build_ticker_from_search_result(item: dict) -> str:
    """
    Transforme un résultat de recherche EODHD en ticker utilisable (Code.Exchange).
    Ex : Code=MC, Exchange=PA -> 'MC.PA'
    """
    code = item.get("Code")
    exch = item.get("Exchange")
    if not code or not exch:
        return None
    return f"{code}.{exch}"


def fetch_eod_price(ticker: str, api_key: str):
    """
    Récupère le dernier cours de clôture via l'endpoint EOD.
    """
    url = f"{EODHD_BASE_URL}/eod/{ticker}"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "order": "d",
        "limit": 1,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return data[0].get("close")


def fetch_fundamentals(ticker: str, api_key: str):
    """
    Récupère les fondamentaux (General, Financials, etc.).
    Endpoint : /fundamentals/{ticker}
    """
    url = f"{EODHD_BASE_URL}/fundamentals/{ticker}"
    params = {"api_token": api_key}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# =========================================
# EXTRACTION DES DONNÉES FONDAMENTALES
# =========================================

def get_company_summary(fundamentals: dict):
    """
    Extrait quelques infos générales : nom, secteur, industrie, pays, devise.
    """
    gen = fundamentals.get("General", {}) or {}
    return {
        "Name": gen.get("Name"),
        "Code": gen.get("Code"),
        "Exchange": gen.get("Exchange"),
        "Sector": gen.get("Sector"),
        "Industry": gen.get("Industry"),
        "Country": gen.get("CountryName"),
        "Currency": gen.get("CurrencyCode"),
    }


def get_shares_outstanding(fundamentals: dict):
    """
    Nombre d'actions en circulation (Shares Outstanding).

    EODHD peut le fournir via:
    - SharesStats.SharesOutstanding
    - Highlights.SharesOutstanding (selon instruments)
    Si non disponible, le fallback MarketCap/Price est géré dans analyze_company().
    """
    try:
        if not isinstance(fundamentals, dict):
            return None
        ss = fundamentals.get("SharesStats", {}) or {}
        hi = fundamentals.get("Highlights", {}) or {}

        candidates = [
            ss.get("SharesOutstanding"),
            ss.get("sharesOutstanding"),
            hi.get("SharesOutstanding"),
            hi.get("sharesOutstanding"),
        ]
        shares = next((c for c in candidates if c not in (None, "", 0, "0")), None)
        if shares is None:
            return None
        return float(shares)
    except Exception:
        return None


def get_net_debt(fundamentals: dict):
    """
    Dette nette (approx) à partir du dernier bilan annuel.

    Convention EODHD (glossaire "Fundamentals"):
    netDebt = shortTermDebt + longTermDebtTotal - cash (ou cashAndEquivalents). 

    Priorité :
    1) champ 'netDebt' s'il est fourni
    2) (shortLongTermDebtTotal OU totalDebt) - cash
    3) (shortTermDebt + longTermDebtTotal/longTermDebt) - cash

    Remarque : si la dette ou le cash n'est pas disponible, renvoie None (pas d'invention).
    """
    try:
        bs = fundamentals.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})
        year, row = _extract_latest_year_row(bs)
        if not isinstance(row, dict) or not row:
            return None

        # 1) netDebt direct
        net_debt = pick_first_non_null(row, ["netDebt", "NetDebt", "net_debt"])
        if net_debt is not None:
            return net_debt

        # Cash (EODHD : cash et cashAndEquivalents peuvent différer selon l'exchange) 
        cash = pick_first_non_null(
            row,
            [
                "cashAndEquivalents",
                "cashAndCashEquivalents",
                "cashAndCashEquivalentsAndShortTermInvestments",
                "cash",
                "Cash",
                "cash_and_equivalents",
                "cash_and_cash_equivalents",
            ],
        )

        if cash is None:
            return None

        # 2) total debt direct
        total_debt = pick_first_non_null(
            row,
            [
                "shortLongTermDebtTotal",
                "ShortLongTermDebtTotal",
                "totalDebt",
                "TotalDebt",
                "total_debt",
                "totalDebtGrossMinorityInterest",
                "totalDebtNetMinorityInterest",
                "debt",
            ],
        )
        if total_debt is not None:
            return total_debt - cash

        # 3) short + long
        st_debt = pick_first_non_null(row, ["shortTermDebt", "ShortTermDebt", "short_term_debt", "currentDebt", "current_debt"])
        lt_debt = pick_first_non_null(
            row,
            [
                "longTermDebtTotal",
                "LongTermDebtTotal",
                "longTermDebt",
                "LongTermDebt",
                "long_term_debt_total",
                "long_term_debt",
                "longTermDebtNonCurrent",
                "longtermdebtnoncurrent",
            ],
        )
        if st_debt is None and lt_debt is None:
            return None

        return (st_debt or 0.0) + (lt_debt or 0.0) - cash

    except Exception:
        return None


def pick_first_non_null(row: dict, candidates):
    """
    Retourne la première valeur non nulle trouvée parmi les clés candidates dans `row`.
    Robustesse EODHD :
    - recherche directe (clé exacte)
    - fallback insensible à la casse + suppression des underscores/espaces/tirets.
    """
    if not isinstance(row, dict) or not row:
        return None

    def _norm_key(k: str) -> str:
        return str(k).strip().lower().replace("_", "").replace(" ", "").replace("-", "")

    # map normalisée -> valeur
    normalized = {_norm_key(k): v for k, v in row.items()}

    for key in candidates:
        # 1) exact
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass

        # 2) normalisé
        nk = _norm_key(key)
        if nk in normalized and normalized[nk] is not None:
            try:
                return float(normalized[nk])
            except (TypeError, ValueError):
                continue

    return None


def build_historical_table(fundamentals: dict, max_years: int = 5) -> pd.DataFrame:
    """
    Construit un tableau historique multi-lignes sur les dernières années :
    CA, EBIT, Net Income, Operating CF, Capex, FCF approx.
    On reste sur du yearly.
    """
    try:
        inc = fundamentals["Financials"]["Income_Statement"]["yearly"]
        cf = fundamentals["Financials"]["Cash_Flow"]["yearly"]
    except Exception:
        return pd.DataFrame()

    if not isinstance(inc, dict) or not isinstance(cf, dict):
        return pd.DataFrame()

    years = sorted(inc.keys(), reverse=True)
    years = years[:max_years]

    rows = []
    for y in years:
        inc_y = inc.get(y, {}) or {}
        cf_y = cf.get(y, {}) or {}

        # CA
        revenue = pick_first_non_null(
            inc_y,
            [
                "TotalRevenue",
                "Revenue",
                "totalRevenue",
                "SalesRevenueNet",
                "Sales",
            ],
        )

        # EBIT / résultat opérationnel
        ebit = pick_first_non_null(
            inc_y,
            [
                "OperatingIncome",
                "OperatingIncomeLoss",
                "EBIT",
                "ebit",
                "Ebit",
            ],
        )

        # Résultat net
        net_income = pick_first_non_null(
            inc_y,
            [
                "NetIncome",
                "netIncome",
                "NetIncomeCommonStockholders",
                "NetIncomeIncludingNoncontrollingInterests",
            ],
        )

        # Flux de trésorerie d'exploitation
        op_cf = pick_first_non_null(
            cf_y,
            [
                "totalCashFromOperatingActivities",
                "TotalCashFromOperatingActivities",
                "NetCashProvidedByOperatingActivities",
                "NetCashFromOperatingActivities",
                "OperatingCashFlow",
            ],
        )

        # Capex
        capex = pick_first_non_null(
            cf_y,
            [
                "capitalExpenditures",
                "CapitalExpenditures",
                "investmentsInPropertyPlantAndEquipment",
                "InvestmentsInPropertyPlantAndEquipment",
            ],
        )

        if op_cf is not None and capex is not None:
            fcf = op_cf - capex
        else:
            fcf = None

        rows.append(
            {
                "Année": y,
                "Chiffre d'affaires": revenue,
                "EBIT": ebit,
                "Résultat net": net_income,
                "Op. Cash Flow": op_cf,
                "Capex": capex,
                "FCF (approx)": fcf,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Année")
    return df


def scale_df_to_millions(df: pd.DataFrame, exclude_cols=("Année",)) -> pd.DataFrame:
    """
    Convertit toutes les colonnes numériques (sauf celles dans exclude_cols) en millions.
    Renomme ces colonnes avec un suffixe ' (M)'.
    """
    df_out = df.copy()
    numeric_cols = [
        c for c in df_out.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_out[c])
    ]
    for c in numeric_cols:
        df_out[c] = df_out[c].astype(float) / 1_000_000
    rename_map = {c: f"{c} (M)" for c in numeric_cols}
    df_out = df_out.rename(columns=rename_map)
    return df_out


def format_large_number(x: float) -> str:
    """
    Format lisible pour les grands nombres : en M ou Md selon la taille.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"

    try:
        x = float(x)
    except Exception:
        return "N/A"

    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f} Md"
    elif ax >= 1_000_000:
        return f"{x / 1_000_000:.2f} M"
    else:
        return f"{x:,.0f}"


def format_float(x, decimals: int = 2, thousands: bool = False) -> str:
    """Format float safely for UI. Returns 'N/A' if x is None/NaN/non-numeric."""
    if x is None:
        return "N/A"
    try:
        # Handle numpy scalars
        if isinstance(x, (np.generic,)):
            x = float(x)
        else:
            x = float(x)
    except Exception:
        return "N/A"
    if isinstance(x, float) and math.isnan(x):
        return "N/A"
    fmt = f"{{:,.{decimals}f}}" if thousands else f"{{:.{decimals}f}}"
    try:
        return fmt.format(x)
    except Exception:
        return "N/A"


def safe_metric(value, fmt="{:.2f}", na="N/A"):
    if value is None:
        return na
    try:
        if isinstance(value, (np.generic,)):
            value = float(value)
        else:
            value = float(value)
    except Exception:
        return na
    if isinstance(value, float) and math.isnan(value):
        return na
    try:
        return fmt.format(value)
    except Exception:
        return na


def estimate_starting_fcf(fundamentals: dict):
    """
    UFCF ≈ Free Cash Flow si dispo,
    sinon FCF = TotalCashFromOperatingActivities - CapitalExpenditures (ou équivalents).
    """
    try:
        cf = fundamentals["Financials"]["Cash_Flow"]["yearly"]
    except Exception:
        return None

    if not isinstance(cf, dict) or not cf:
        return None

    years = sorted(cf.keys())
    last_year_key = years[-1]
    row = cf[last_year_key] or {}

    # 1) Free cash-flow direct si dispo
    for key in ["freeCashFlow", "FreeCashFlow"]:
        if key in row and row[key] is not None:
            return float(row[key])

    # 2) Sinon on reconstruit : FCF = OCF - Capex
    ocf_candidates = [
        "totalCashFromOperatingActivities",
        "TotalCashFromOperatingActivities",
        "NetCashProvidedByOperatingActivities",
        "NetCashFromOperatingActivities",
    ]
    capex_candidates = [
        "capitalExpenditures",
        "CapitalExpenditures",
        "investmentsInPropertyPlantAndEquipment",
        "InvestmentsInPropertyPlantAndEquipment",
    ]

    operating_cf = next((row[k] for k in ocf_candidates if k in row and row[k] is not None), None)
    capex = next((row[k] for k in capex_candidates if k in row and row[k] is not None), None)

    if operating_cf is None or capex is None:
        try:
            st.write("⚠️ Clés Cash Flow disponibles pour", last_year_key, ":", list(row.keys()))
        except Exception:
            pass
        return None

    return float(operating_cf) - float(capex)
    
def estimate_normalized_fcf(hist_df: pd.DataFrame):
    """
    FCF normalisé (prioritaire pour Big Caps) :
    - prend la série "FCF (approx)" de l'historique
    - garde uniquement les FCF positifs (sinon non exploitable)
    - retourne la moyenne des 3 derniers FCF positifs (plus stable qu'une seule année)
    """
    if hist_df is None or hist_df.empty:
        return None
    if "FCF (approx)" not in hist_df.columns:
        return None

    s = hist_df["FCF (approx)"].dropna()
    s = s[s > 0]

    if len(s) == 0:
        return None

    return float(s.tail(3).mean())

# =========================================
# EXTRACTION BASE POUR MULTIPLES
# =========================================

def extract_base_financials(fundamentals: dict):
    """
    Extrait les valeurs de base (dernière année annuelle) nécessaires aux multiples :
    - revenue
    - ebitda
    - ebit
    - net_income
    - book_equity (fonds propres comptables)

    Bonnes pratiques EODHD : certaines clés diffèrent selon les exchanges (camelCase vs variantes). 
    """
    inc = fundamentals.get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    bs = fundamentals.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})

    revenue = ebitda = ebit = net_income = book_equity = None

    # ============================================================
    #                   INCOME STATEMENT
    # ============================================================
    if isinstance(inc, dict) and inc:
        years_inc = sorted(inc.keys())
        last_year_inc = years_inc[-1]
        row_inc = inc.get(last_year_inc, {}) or {}

        revenue = pick_first_non_null(
            row_inc,
            ["totalRevenue", "TotalRevenue", "revenue", "Revenue", "SalesRevenueNet", "Sales"],
        )

        ebitda = pick_first_non_null(
            row_inc,
            ["ebitda", "EBITDA", "Ebitda", "OperatingIncomeBeforeDepreciation"],
        )

        ebit = pick_first_non_null(
            row_inc,
            ["ebit", "EBIT", "operatingIncome", "OperatingIncome", "OperatingIncomeLoss"],
        )

        net_income = pick_first_non_null(
            row_inc,
            [
                "netIncome",
                "NetIncome",
                "net_income",
                "NetIncomeCommonStockholders",
                "NetIncomeIncludingNoncontrollingInterests",
            ],
        )

    # ============================================================
    #                     BALANCE SHEET
    # ============================================================
    if isinstance(bs, dict) and bs:
        years_bs = sorted(bs.keys())
        last_year_bs = years_bs[-1]
        row_bs = bs.get(last_year_bs, {}) or {}

        # Normaliser les clés (insensible à la casse + suppression _ / espaces / tirets)
        def _norm_key(k: str) -> str:
            return str(k).strip().lower().replace("_", "").replace(" ", "").replace("-", "")

        normalized = {_norm_key(k): v for k, v in row_bs.items()}

        def _get_float(*keys):
            for k in keys:
                v = normalized.get(_norm_key(k))
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
            return None

        # Equity direct (plusieurs variantes)
        equity = _get_float(
            "totalStockholderEquity",
            "totalStockholdersEquity",
            "totalstockholderequity",
            "totalStockholdersequity",
            "totalShareholdersEquity",
            "commonStockEquity",
            "stockholdersEquity",
            "shareholdersEquity",
            "totalEquity",
            "total_equity",
            "totalEquityGrossMinorityInterest",
            "totalEquityGrossMinorityInterest",
            "totalEquityIncludingMinorityInterest",
            "totalEquityNetMinorityInterest",
        )

        if equity is not None:
            book_equity = equity

    return {
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "net_income": net_income,
        "book_equity": book_equity,
    }


# =========================================
# MULTIPLES
# =========================================

def compute_base_multiples(price: float, shares: float, net_debt: float, base_financials: dict):
    """
    Calcule les multiples 'courants' de la société, basés sur le dernier exercice annuel.
    Renvoie un dict : pe, pb, ev_ebitda, ev_sales, ev_ebit, etc.
    """
    if shares is None or shares == 0:
        return {}

    market_cap = price * shares
    net_debt_used = net_debt if net_debt is not None else 0
    ev = market_cap + net_debt_used

    revenue = base_financials.get("revenue")
    ebitda = base_financials.get("ebitda")
    ebit = base_financials.get("ebit")
    net_income = base_financials.get("net_income")
    book_equity = base_financials.get("book_equity")

    pe = (price / (net_income / shares)) if net_income not in (None, 0) else None
    pb = (price / (book_equity / shares)) if book_equity not in (None, 0) else None
    ev_ebitda = (ev / ebitda) if ebitda not in (None, 0) else None
    ev_ebit = (ev / ebit) if ebit not in (None, 0) else None
    ev_sales = (ev / revenue) if revenue not in (None, 0) else None

    return {
        "market_cap": market_cap,
        "ev": ev,
        "pe": pe,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "ev_ebit": ev_ebit,
        "ev_sales": ev_sales,
        "revenue": revenue,
        "ebit": ebit,
        "ebitda": ebitda,
        "net_income": net_income,
        "book_equity": book_equity,
        "eps": (net_income / shares) if net_income not in (None, 0) else None,
        "bvps": (book_equity / shares) if book_equity not in (None, 0) else None,
    }


# =========================================
# HELPERS : EXTRACTION DERNIER EXERCICE
# =========================================

def _extract_latest_year_row(yearly_dict: dict):
    """
    Renvoie (year, row) pour la dernière année disponible.
    """
    if not isinstance(yearly_dict, dict) or not yearly_dict:
        return None, {}
    years = sorted(yearly_dict.keys())
    last_year = years[-1]
    return last_year, yearly_dict[last_year] or {}


def extract_balance_sheet_snapshot(fundamentals: dict):
    """
    Extrait un snapshot de bilan (dernière année annuelle).
    Renvoie (year, snapshot_dict).
    """
    bs = fundamentals.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})
    year, row = _extract_latest_year_row(bs)

    def get_first(keys):
        return pick_first_non_null(row, keys)

    snap = {
        "total_assets": get_first(["totalAssets", "TotalAssets"]),
        "total_equity": get_first(["totalStockholderEquity", "totalStockholdersEquity", "TotalStockholderEquity", "TotalStockholdersEquity", "totalEquity"]),
        "total_debt": get_first(["shortLongTermDebtTotal", "totalDebt", "TotalDebt", "shortLongTermDebt", "total_debt"]),
        "cash": get_first(["cash", "cashAndEquivalents", "CashAndCashEquivalents"]),
        "current_assets": get_first(["totalCurrentAssets", "TotalCurrentAssets", "currentAssets"]),
        "current_liabilities": get_first(["totalCurrentLiabilities", "TotalCurrentLiabilities", "currentLiabilities"]),
        "goodwill": get_first(["goodWill", "Goodwill", "goodwill"]),
        "intangibles": get_first(["intangibleAssets", "IntangibleAssets", "intangibleAssetsNet", "Intangibles", "intangible_assets"]),
    }
    return year, snap


def extract_income_snapshot(fundamentals: dict):
    """
    Extrait un snapshot d'Income Statement (dernière année annuelle).
    Renvoie (year, snapshot_dict).
    """
    inc = fundamentals.get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    year, row = _extract_latest_year_row(inc)

    def get_first(keys):
        return pick_first_non_null(row, keys)

    revenue = get_first(["totalRevenue", "TotalRevenue", "revenue", "Revenue", "SalesRevenueNet"])
    ebitda = get_first(["ebitda", "EBITDA", "Ebitda", "OperatingIncomeBeforeDepreciation"])
    ebit = get_first(["operatingIncome", "OperatingIncome", "OperatingIncomeLoss", "ebit", "EBIT"])
    net_income = get_first(["netIncome", "NetIncome", "NetIncomeCommonStockholders"])
    gross_profit = get_first(["grossProfit", "GrossProfit", "gross_profit"])

    # Champs WACC / fiscalité
    interest_expense = get_first([
        "interestExpense",
        "InterestExpense",
        "interestExpenseNonOperating",
        "InterestExpenseNonOperating",
        "interest_expense",
    ])
    # convention: expense peut être négative, on stocke en valeur absolue
    if interest_expense is not None:
        interest_expense = abs(interest_expense)

    pretax_income = get_first([
        "incomeBeforeTax",
        "IncomeBeforeTax",
        "pretaxIncome",
        "PretaxIncome",
        "incomeBeforeIncomeTaxes",
        "IncomeBeforeIncomeTaxes",
    ])

    tax_provision = get_first([
        "incomeTaxExpense",
        "IncomeTaxExpense",
        "taxProvision",
        "TaxProvision",
        "provisionForIncomeTaxes",
        "ProvisionForIncomeTaxes",
    ])
    if tax_provision is not None:
        tax_provision = abs(tax_provision)

    snap = {
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "net_income": net_income,
        "gross_profit": gross_profit,
        "interest_expense": interest_expense,
        "pretax_income": pretax_income,
        "tax_provision": tax_provision,
    }
    return year, snap

# =========================================
# WACC AUTO (EODHD) - helpers robustes
# =========================================

def fetch_latest_eod_close(ticker: str, api_key: str):
    """
    Récupère le dernier 'close' via l'endpoint EOD.
    Exemple doc EODHD (GBOND): /api/eod/UK10Y.GBOND?api_token=...&fmt=json
    Renvoie float close ou None.
    """
    if not ticker or not api_key:
        return None
    try:
        url = f"https://eodhd.com/api/eod/{ticker}"
        params = {
            "api_token": api_key,
            "fmt": "json",
            "order": "d",
            "limit": 1,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "close" in data:
            # certains endpoints peuvent renvoyer un dict unique
            try:
                return float(data.get("close"))
            except Exception:
                return None
        if not isinstance(data, list) or len(data) == 0:
            return None
        last = data[0]  # order=d => plus récent en premier
        try:
            return float(last.get("close"))
        except Exception:
            return None
    except Exception:
        return None


def load_risk_free_cache(path: str = RISK_FREE_CACHE_FILE):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_risk_free_cache(cache: dict, path: str = RISK_FREE_CACHE_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_cached_risk_free_rate(ticker: str, max_days: int = RISK_FREE_CACHE_MAX_DAYS):
    cache = load_risk_free_cache()
    entry = cache.get(ticker)
    if not entry or not isinstance(entry, dict):
        return None, None
    try:
        value = float(entry.get("value"))
        date_str = entry.get("date")
        if not date_str:
            return None, None
        date_val = datetime.strptime(date_str, "%Y-%m-%d").date()
        age_days = (datetime.utcnow().date() - date_val).days
        if age_days <= max_days:
            return value, age_days
    except Exception:
        return None, None
    return None, None


def fetch_risk_free_rate_with_cache(ticker: str, api_key: str):
    rf_close = fetch_latest_eod_close(ticker, api_key)
    if rf_close is not None:
        cache = load_risk_free_cache()
        cache[ticker] = {
            "value": float(rf_close),
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
        }
        save_risk_free_cache(cache)
        return rf_close, "api", None

    cached_value, age_days = get_cached_risk_free_rate(ticker)
    if cached_value is not None:
        return cached_value, "cache", age_days

    return None, None, None


def get_currency_code(fundamentals: dict):
    gen = fundamentals.get("General", {}) if isinstance(fundamentals, dict) else {}
    cc = gen.get("CurrencyCode") or gen.get("currencyCode") or gen.get("currency_code")
    if cc:
        return str(cc).upper()
    return None


def get_country_iso(fundamentals: dict):
    gen = fundamentals.get("General", {}) if isinstance(fundamentals, dict) else {}
    ci = gen.get("CountryISO") or gen.get("countryISO") or gen.get("country_iso")
    if ci:
        return str(ci).upper()
    return None


def get_beta(fundamentals: dict):
    tech = fundamentals.get("Technicals", {}) if isinstance(fundamentals, dict) else {}
    b = tech.get("Beta") or tech.get("beta")
    try:
        return float(b) if b is not None else None
    except Exception:
        return None


def get_market_cap(fundamentals: dict):
    hi = fundamentals.get("Highlights", {}) if isinstance(fundamentals, dict) else {}
    mc = hi.get("MarketCapitalization") or hi.get("marketCapitalization") or hi.get("market_cap")
    try:
        return float(mc) if mc is not None else None
    except Exception:
        return None


def _extract_total_debt_from_bs_row(row: dict):
    """
    Extrait une dette totale (interest-bearing) depuis une ligne de bilan EODHD.
    """
    if not row or not isinstance(row, dict):
        return None

    def _norm_key(k: str) -> str:
        return "".join(ch.lower() for ch in str(k) if ch.isalnum())

    normalized = { _norm_key(k): v for k, v in row.items() }

    def get_first(keys):
        for k in keys:
            v = normalized.get(_norm_key(k))
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return None

    # priorités usuelles EODHD
    total_debt = get_first([
        "shortLongTermDebtTotal",
        "shortLongTermDebt",
        "totalDebt",
        "TotalDebt",
        "total_debt",
    ])
    if total_debt is not None:
        return total_debt

    st = get_first(["shortTermDebt", "ShortTermDebt", "short_term_debt", "currentDebt", "CurrentDebt"])
    lt = get_first([
        "longTermDebtTotal",
        "LongTermDebtTotal",
        "longTermDebt",
        "LongTermDebt",
        "long_term_debt_total",
        "long_term_debt",
        "longTermDebtNonCurrent",
    ])

    if st is None and lt is None:
        return None
    return (st or 0.0) + (lt or 0.0)

# ... (le reste du fichier est inchangé ; contenu complet fourni dans le dépôt)

if __name__ == "__main__":
    main()
