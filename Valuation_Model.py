import os
import json
import math
import requests
import pandas as pd
import streamlit as st

# =========================================
# CONFIG DE BASE
# =========================================

EODHD_BASE_URL = "https://eodhd.com/api"


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
    Récupère le nombre d'actions si disponible.
    """
    try:
        shares = fundamentals["SharesStats"].get("SharesOutstanding")
        return shares
    except Exception:
        return None


def get_net_debt(fundamentals: dict):
    """
    Dette nette (Net Debt).

    Priorité:
    1) champ reporté netDebt (si présent dans le Balance_Sheet de Financials),
    2) calcul (shortTermDebt + longTermDebt) - cash,
    3) calcul totalDebt - cash.

    Si l'information n'est pas disponible, retourne None (pas d'invention).
    """
    try:
        _, bs_snap = extract_balance_sheet_snapshot(fundamentals)
        if not bs_snap:
            return None

        net_debt_reported = bs_snap.get("net_debt_reported")
        if net_debt_reported is not None:
            try:
                return float(net_debt_reported)
            except Exception:
                pass

        cash = bs_snap.get("cash")
        if cash is None:
            return None

        std = bs_snap.get("short_term_debt")
        ltd = bs_snap.get("long_term_debt")
        if std is not None or ltd is not None:
            return (float(std or 0.0) + float(ltd or 0.0)) - float(cash)

        total_debt = bs_snap.get("total_debt")
        if total_debt is not None:
            return float(total_debt) - float(cash)

        return None
    except Exception:
        return None

def pick_first_non_null(row: dict, candidates):
    """
    Retourne la première valeur non nulle trouvée parmi les clés candidates dans `row`.
    Si rien n'est trouvé, renvoie None.
    """
    for key in candidates:
        if key in row and row[key] is not None:
            try:
                return float(row[key])
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
    """
    inc = fundamentals.get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    bs = fundamentals.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})

    revenue = ebitda = ebit = net_income = book_equity = None

    # ---------- INCOME STATEMENT ----------
    if isinstance(inc, dict) and inc:
        years = sorted(inc.keys())
        last_year = years[-1]
        row_inc = inc.get(last_year, {}) or {}

        revenue = pick_first_non_null(
            row_inc,
            ["TotalRevenue", "Revenue", "totalRevenue", "SalesRevenueNet", "Sales"],
        )

        ebitda = pick_first_non_null(
            row_inc,
            ["EBITDA", "Ebitda", "ebitda", "OperatingIncomeBeforeDepreciation"],
        )

        ebit = pick_first_non_null(
            row_inc,
            ["OperatingIncome", "OperatingIncomeLoss", "EBIT", "Ebit", "ebit"],
        )

        net_income = pick_first_non_null(
            row_inc,
            [
                "NetIncome",
                "netIncome",
                "NetIncomeCommonStockholders",
                "NetIncomeIncludingNoncontrollingInterests",
            ],
        )

def extract_base_financials(fundamentals: dict):
    """
    Extrait les valeurs de base (dernière année annuelle) nécessaires aux multiples :
    - revenue
    - ebitda
    - ebit
    - net_income
    - book_equity (fonds propres comptables)
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
            ["TotalRevenue", "Revenue", "totalRevenue", "SalesRevenueNet", "Sales"],
        )

        ebitda = pick_first_non_null(
            row_inc,
            ["EBITDA", "Ebitda", "ebitda", "OperatingIncomeBeforeDepreciation"],
        )

        ebit = pick_first_non_null(
            row_inc,
            ["OperatingIncome", "OperatingIncomeLoss", "EBIT", "Ebit", "ebit"],
        )

        net_income = pick_first_non_null(
            row_inc,
            [
                "NetIncome",
                "netIncome",
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

        # Normaliser les clés en minuscules (insensible à la casse)
        normalized = {k.lower(): v for k, v in row_bs.items()}

        # Clés possibles pour les fonds propres (équity)
        equity_keys = [
            "totalstockholderequity",
            "totalstockholdersequity",
            "totalshareholdersequity",
            "commonstockequity",
            "stockholdersequity",
            "shareholdersequity",
            "totalequity",
            "totalequitygrossminorityinterest",
            "totalequityandminorityinterest",
        ]

        # Recherche directe dans les clés normalisées
        for key in equity_keys:
            if key in normalized and normalized[key] not in (None, 0):
                try:
                    book_equity = float(normalized[key])
                    break
                except:
                    pass

        # ---------------------------
        # Fallback automatique
        # book_equity = TotalAssets – TotalLiabilities
        # ---------------------------
        if book_equity is None:
            total_assets = (
                normalized.get("totalassets")
                or normalized.get("totalassetsreported")
                or normalized.get("assets")
            )

            total_liabilities = (
                normalized.get("totalliabilitiesnetminorityinterest")
                or normalized.get("totalliabilities")
                or normalized.get("liabilities")
            )

            if total_assets is not None and total_liabilities is not None:
                try:
                    book_equity = float(total_assets) - float(total_liabilities)
                except:
                    book_equity = None

    # ============================================================
    #                     RETURN FINAL
    # ============================================================
    return {
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "net_income": net_income,
        "book_equity": book_equity,
    }



def safe_div(num, den):
    """
    Division sécurisée :
    - renvoie None si num ou den est None
    - renvoie None si den = 0
    - évite les erreurs de type
    """
    if num is None or den in (None, 0):
        return None
    try:
        return float(num) / float(den)
    except Exception:
        return None

def compute_base_multiples(price, shares, net_debt, base_financials: dict):
    """
    Calcule les métriques de base pour les méthodes par multiples :
    - EPS, BVPS
    - Market cap, EV
    - P/E, P/B, EV/EBITDA, EV/EBIT, EV/Sales
    """
    revenue = base_financials.get("revenue")
    ebitda = base_financials.get("ebitda")
    ebit = base_financials.get("ebit")
    net_income = base_financials.get("net_income")
    book_equity = base_financials.get("book_equity")

    metrics = {}

    eps = None
    bvps = None
    market_cap = None
    ev = None

    if price is not None and shares not in (None, 0):
        market_cap = price * shares

    if shares not in (None, 0):
        if net_income is not None:
            eps = net_income / shares
        if book_equity is not None:
            bvps = book_equity / shares

    if market_cap is not None:
        ev = market_cap + (net_debt or 0)

    metrics["revenue"] = revenue
    metrics["ebitda"] = ebitda
    metrics["ebit"] = ebit
    metrics["net_income"] = net_income
    metrics["book_equity"] = book_equity
    metrics["eps"] = eps
    metrics["bvps"] = bvps
    metrics["market_cap"] = market_cap
    metrics["ev"] = ev

    # Multiples courants
    metrics["pe"] = safe_div(price, eps)
    metrics["pb"] = safe_div(price, bvps)
    metrics["ev_ebitda"] = safe_div(ev, ebitda)
    metrics["ev_ebit"] = safe_div(ev, ebit)
    metrics["ev_sales"] = safe_div(ev, revenue)

    return metrics



# =========================================
# FUNDAMENTALS HEALTH TABLE (RATIOS + SCORE)
# =========================================

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_fundamentals_cached(ticker: str, api_key: str):
    return fetch_fundamentals(ticker, api_key)

@st.cache_data(show_spinner=False, ttl=6*3600)
def fetch_eod_price_cached(ticker: str, api_key: str):
    return fetch_eod_price(ticker, api_key)


def _extract_latest_year_row(section: dict):
    """
    section attendu sous forme de dict {year: {...}}.
    Retourne (year, row) du dernier exercice disponible.
    """
    if not isinstance(section, dict) or not section:
        return None, {}
    years = sorted(section.keys())
    last_year = years[-1]
    return last_year, (section.get(last_year, {}) or {})


def extract_balance_sheet_snapshot(fundamentals: dict):
    """
    Extrait un snapshot de bilan (dernier exercice annuel) avec des clés standardisées.
    Retourne aussi l'année du snapshot.

    Objectif: être robuste aux variations de champs EODHD (camelCase / underscore / etc),
    sans inventer de données quand un champ n'existe pas.
    """
    bs = fundamentals.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})
    year, row = _extract_latest_year_row(bs)

    if not row:
        return year, {}

    normalized = {str(k).lower(): v for k, v in row.items()}

    def get_first(keys):
        for k in keys:
            v = normalized.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return None

    total_assets = get_first(["totalassets", "total_assets", "totalassetsreported", "assets"])
    total_liabilities = get_first([
        "totalliabilitiesnetminorityinterest",
        "total_liabilities_net_minority_interest",
        "totalliabilities",
        "total_liabilities",
        "liabilities",
    ])

    # Dette (EODHD peut fournir plusieurs champs)
    short_term_debt = get_first([
        "shorttermdebt",
        "short_term_debt",
        "shorttermdebtandcapitalleaseobligation",
        "short_term_debt_and_capital_lease_obligation",
    ])
    long_term_debt = get_first([
        "longtermdebt",
        "long_term_debt",
        "longtermdebtandcapitalleaseobligation",
        "long_term_debt_and_capital_lease_obligation",
        "longtermdebtnoncurrent",
        "long_term_debt_noncurrent",
    ])

    total_debt = get_first([
        "totaldebt",
        "total_debt",
        "shortlongtermdebttotal",
        "short_long_term_debt_total",
    ])
    if total_debt is None and (short_term_debt is not None or long_term_debt is not None):
        total_debt = (short_term_debt or 0.0) + (long_term_debt or 0.0)

    net_debt_reported = get_first(["netdebt", "net_debt"])

    cash = get_first([
        "cashandcashequivalents",
        "cash_and_cash_equivalents",
        "cashcashequivalentsandshortterminvestments",
        "cash_cashequivalents_and_short_term_investments",
        "cash",
    ])
    current_assets = get_first(["totalcurrentassets", "total_current_assets", "currentassets", "current_assets"])
    current_liabilities = get_first(["totalcurrentliabilities", "total_current_liabilities", "currentliabilities", "current_liabilities"])
    goodwill = get_first(["goodwill"])
    intangibles = get_first([
        "intangibleassets",
        "intangible_assets",
        "intangibleassetsexcludinggoodwill",
        "intangible_assets_excluding_goodwill",
        "intangibles",
    ])

    # Equity : on réutilise la logique robuste déjà codée dans extract_base_financials (book_equity)
    try:
        book_equity = extract_base_financials(fundamentals).get("book_equity")
        if book_equity is not None:
            book_equity = float(book_equity)
    except Exception:
        book_equity = None

    # Fallback equity si besoin : Assets - Liabilities
    if book_equity is None and total_assets is not None and total_liabilities is not None:
        book_equity = total_assets - total_liabilities

    return year, {
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_equity": book_equity,
        "total_debt": total_debt,
        "short_term_debt": short_term_debt,
        "long_term_debt": long_term_debt,
        "net_debt_reported": net_debt_reported,
        "cash": cash,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "goodwill": goodwill,
        "intangibles": intangibles,
    }


def extract_cashflow_snapshot(fundamentals: dict):
    """
    Extrait un snapshot de cash-flow (dernier exercice annuel) : CFO, capex, FCF approx.
    Retourne aussi l'année.
    """
    cf = fundamentals.get("Financials", {}).get("Cash_Flow", {}).get("yearly", {})
    year, row = _extract_latest_year_row(cf)
    if not row:
        return year, {}

    ocf = pick_first_non_null(
        row,
        [
            "totalCashFromOperatingActivities",
            "TotalCashFromOperatingActivities",
            "NetCashProvidedByOperatingActivities",
            "NetCashFromOperatingActivities",
            "OperatingCashFlow",
        ],
    )
    capex = pick_first_non_null(
        row,
        [
            "capitalExpenditures",
            "CapitalExpenditures",
            "investmentsInPropertyPlantAndEquipment",
            "InvestmentsInPropertyPlantAndEquipment",
        ],
    )

    fcf = (ocf - capex) if (ocf is not None and capex is not None) else None
    return year, {"cfo": ocf, "capex": capex, "fcf": fcf}


def extract_income_snapshot(fundamentals: dict):
    """
    Extrait un snapshot compte de résultat (dernier exercice annuel) : revenue, ebitda, ebit, net_income, gross_profit.
    Retourne aussi l'année.
    """
    inc = fundamentals.get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    year, row = _extract_latest_year_row(inc)
    if not row:
        return year, {}

    revenue = pick_first_non_null(row, ["TotalRevenue", "Revenue", "totalRevenue", "SalesRevenueNet", "Sales"])
    ebitda = pick_first_non_null(row, ["EBITDA", "Ebitda", "ebitda", "OperatingIncomeBeforeDepreciation"])
    ebit = pick_first_non_null(row, ["OperatingIncome", "OperatingIncomeLoss", "EBIT", "Ebit", "ebit"])
    net_income = pick_first_non_null(
        row,
        ["NetIncome", "netIncome", "NetIncomeCommonStockholders", "NetIncomeIncludingNoncontrollingInterests"],
    )
    gross_profit = pick_first_non_null(row, ["GrossProfit", "grossProfit"])

    return year, {
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "net_income": net_income,
        "gross_profit": gross_profit,
    }


def _linear_score(value, low, high, higher_better=True):
    """
    Score linéaire 0-10 sur une plage [low, high].
    - higher_better=True : low -> 0, high -> 10
    - higher_better=False : low -> 10, high -> 0
    """
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None

    if low is None or high is None or low == high:
        return None

    # clamp dans la plage
    if x <= low:
        s = 0.0 if higher_better else 10.0
    elif x >= high:
        s = 10.0 if higher_better else 0.0
    else:
        t = (x - low) / (high - low)
        s = 10.0 * (t if higher_better else (1 - t))

    return max(0.0, min(10.0, s))


def _color_from_score(score):
    """
    Renvoie une couleur hex (fond) en fonction du score.
    Gris si score indisponible.
    """
    if score is None:
        return "#E5E7EB"  # gris clair
    try:
        s = float(score)
    except Exception:
        return "#E5E7EB"

    if s >= 8:
        return "#86EFAC"  # vert clair
    if s >= 6:
        return "#BBF7D0"  # vert très clair
    if s >= 4:
        return "#FEF08A"  # jaune
    if s >= 2:
        return "#FDBA74"  # orange
    return "#FCA5A5"      # rouge clair


def compute_company_health_ratios(
    company: dict,
    bs_snap: dict,
    is_snap: dict,
    cf_snap: dict,
    net_debt: float,
    shares: float,
    hist_df: pd.DataFrame,
):
    """
    Calcule un set de ratios fondamentaux (automatisables) à partir des snapshots.
    Tout ce qui n'est pas calculable renvoie None (pas d'extrapolation).
    """
    ratios = {}

    total_assets = bs_snap.get("total_assets")
    total_equity = bs_snap.get("total_equity")
    total_debt = bs_snap.get("total_debt")
    cash = bs_snap.get("cash")
    current_assets = bs_snap.get("current_assets")
    current_liabilities = bs_snap.get("current_liabilities")
    goodwill = bs_snap.get("goodwill")
    intangibles = bs_snap.get("intangibles")

    revenue = is_snap.get("revenue")
    ebitda = is_snap.get("ebitda")
    ebit = is_snap.get("ebit")
    net_income = is_snap.get("net_income")
    gross_profit = is_snap.get("gross_profit")

    cfo = cf_snap.get("cfo")
    capex = cf_snap.get("capex")
    fcf = cf_snap.get("fcf")

    # --- Balance sheet ---
    ratios["equity_to_assets"] = safe_div(total_equity, total_assets)
    ratios["net_debt_to_equity"] = safe_div(net_debt, total_equity)
    ratios["net_debt_to_ebitda"] = safe_div(net_debt, ebitda)
    ratios["cash_to_debt"] = safe_div(cash, total_debt)
    ratios["current_ratio"] = safe_div(current_assets, current_liabilities)
    ratios["cash_to_current_liabilities"] = safe_div(cash, current_liabilities)

    # Qualité des actifs
    ratios["goodwill_to_assets"] = safe_div(goodwill, total_assets)
    ratios["intangibles_to_assets"] = safe_div(intangibles, total_assets)
    if goodwill is not None and intangibles is not None:
        ratios["gw_plus_intang_to_assets"] = safe_div((goodwill + intangibles), total_assets)
    else:
        ratios["gw_plus_intang_to_assets"] = None

    # --- Income statement ---
    ratios["gross_margin"] = safe_div(gross_profit, revenue)
    ratios["ebit_margin"] = safe_div(ebit, revenue)
    ratios["net_margin"] = safe_div(net_income, revenue)

    # Profitabilité (approximations transparentes)
    ratios["roe"] = safe_div(net_income, total_equity)
    ratios["roa"] = safe_div(net_income, total_assets)

    # ROIC approx (pré-tax) = EBIT / (Equity + Net Debt) si dispo
    denom = None
    if total_equity is not None and net_debt is not None:
        denom = total_equity + net_debt
    ratios["roic_approx_pretax"] = safe_div(ebit, denom)

    # --- Cash flow quality ---
    ratios["cfo_to_net_income"] = safe_div(cfo, net_income)
    ratios["fcf_to_net_income"] = safe_div(fcf, net_income)
    ratios["fcf_margin"] = safe_div(fcf, revenue)
    ratios["capex_to_revenue"] = safe_div(capex, revenue)

    # --- Historique ---
    rev_cagr = compute_revenue_cagr(hist_df)
    ratios["revenue_cagr_5y"] = rev_cagr

    # fréquence FCF négatif sur l'historique (si dispo)
    if hist_df is not None and not hist_df.empty and "FCF (approx)" in hist_df.columns:
        s = hist_df["FCF (approx)"].dropna()
        if len(s) > 0:
            ratios["fcf_negative_years_pct"] = float((s < 0).sum()) / float(len(s))
        else:
            ratios["fcf_negative_years_pct"] = None
    else:
        ratios["fcf_negative_years_pct"] = None

    return ratios


# Liste canonique des métriques affichées (clé -> label + pilier + scoring)
HEALTH_METRICS = [
    # Balance sheet
    {"pillar": "Bilan", "key": "equity_to_assets", "label": "Equity / Total Assets", "higher_better": True,  "score_low": 0.10, "score_high": 0.60, "fmt": "pct"},
    {"pillar": "Bilan", "key": "net_debt_to_equity", "label": "Net Debt / Equity", "higher_better": False, "score_low": 0.00, "score_high": 2.00, "fmt": "x"},
    {"pillar": "Bilan", "key": "net_debt_to_ebitda", "label": "Net Debt / EBITDA", "higher_better": False, "score_low": 0.00, "score_high": 4.00, "fmt": "x"},
    {"pillar": "Bilan", "key": "cash_to_debt", "label": "Cash / Total Debt", "higher_better": True,  "score_low": 0.05, "score_high": 0.50, "fmt": "x"},

    # Liquidité
    {"pillar": "Liquidité", "key": "current_ratio", "label": "Current Ratio", "higher_better": True, "score_low": 1.00, "score_high": 2.00, "fmt": "x"},
    {"pillar": "Liquidité", "key": "cash_to_current_liabilities", "label": "Cash / Current Liabilities", "higher_better": True, "score_low": 0.20, "score_high": 1.00, "fmt": "x"},

    # Qualité des actifs
    {"pillar": "Actifs", "key": "gw_plus_intang_to_assets", "label": "(Goodwill + Intangibles) / Total Assets", "higher_better": False, "score_low": 0.10, "score_high": 0.60, "fmt": "pct"},

    # Opérations / marges
    {"pillar": "Opérations", "key": "gross_margin", "label": "Gross Margin", "higher_better": True, "score_low": 0.20, "score_high": 0.60, "fmt": "pct"},
    {"pillar": "Opérations", "key": "ebit_margin", "label": "EBIT Margin", "higher_better": True, "score_low": 0.05, "score_high": 0.25, "fmt": "pct"},
    {"pillar": "Opérations", "key": "net_margin", "label": "Net Margin", "higher_better": True, "score_low": 0.03, "score_high": 0.20, "fmt": "pct"},

    # Rentabilité
    {"pillar": "Rentabilité", "key": "roe", "label": "ROE", "higher_better": True, "score_low": 0.05, "score_high": 0.20, "fmt": "pct"},
    {"pillar": "Rentabilité", "key": "roa", "label": "ROA", "higher_better": True, "score_low": 0.02, "score_high": 0.10, "fmt": "pct"},
    {"pillar": "Rentabilité", "key": "roic_approx_pretax", "label": "ROIC approx (pré-tax)", "higher_better": True, "score_low": 0.05, "score_high": 0.20, "fmt": "pct"},

    # Cash-flow
    {"pillar": "Cash-flow", "key": "cfo_to_net_income", "label": "CFO / Net Income", "higher_better": True, "score_low": 0.80, "score_high": 1.50, "fmt": "x"},
    {"pillar": "Cash-flow", "key": "fcf_to_net_income", "label": "FCF / Net Income", "higher_better": True, "score_low": 0.60, "score_high": 1.20, "fmt": "x"},
    {"pillar": "Cash-flow", "key": "fcf_margin", "label": "FCF Margin", "higher_better": True, "score_low": 0.02, "score_high": 0.15, "fmt": "pct"},
    {"pillar": "Cash-flow", "key": "capex_to_revenue", "label": "Capex / Revenue", "higher_better": False, "score_low": 0.02, "score_high": 0.15, "fmt": "pct"},

    # Historique
    {"pillar": "Croissance", "key": "revenue_cagr_5y", "label": "Revenue CAGR (approx, 5y)", "higher_better": True, "score_low": 0.00, "score_high": 0.12, "fmt": "pct"},
    {"pillar": "Croissance", "key": "fcf_negative_years_pct", "label": "% années FCF négatif (hist.)", "higher_better": False, "score_low": 0.00, "score_high": 0.50, "fmt": "pct"},
]


def _format_metric_value(value, fmt):
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if fmt == "pct":
        return x * 100.0
    return x


def build_health_table(company_ratios: dict, peer_distribution: dict | None = None) -> pd.DataFrame:
    """
    Construit le tableau Health.
    - Si peer_distribution est fourni (dict key -> list[values]) : score = percentile sectoriel (comparables).
    - Sinon : score = basé sur des seuils génériques (non sectorisés).
    """
    rows = []
    for m in HEALTH_METRICS:
        key = m["key"]
        val = company_ratios.get(key)

        median_peer = None
        percentile = None
        score = None
        scoring_method = None

        if peer_distribution and key in peer_distribution and peer_distribution[key]:
            values = [v for v in peer_distribution[key] if v is not None]
            values = [float(v) for v in values if not (isinstance(v, float) and math.isnan(v))]
            if values and val is not None:
                values_sorted = sorted(values)
                median_peer = values_sorted[len(values_sorted)//2]
                # percentile
                pct = 100.0 * (sum(1 for x in values_sorted if x <= val) / len(values_sorted))
                if not m["higher_better"]:
                    pct = 100.0 - pct
                percentile = pct
                score = max(0.0, min(10.0, pct / 10.0))
                scoring_method = "Comparables (percentile)"
        else:
            score = _linear_score(
                value=val,
                low=m.get("score_low"),
                high=m.get("score_high"),
                higher_better=m.get("higher_better", True),
            )
            scoring_method = "Seuils génériques (non sectorisé)"

        rows.append({
            "Pilier": m["pillar"],
            "Métrique": m["label"],
            "Valeur": _format_metric_value(val, m.get("fmt")),
            "Médiane comparables": _format_metric_value(median_peer, m.get("fmt")) if median_peer is not None else None,
            "Percentile": percentile,
            "Score /10": score,
            "Méthode scoring": scoring_method,
            "_color": _color_from_score(score),
        })

    df = pd.DataFrame(rows)

    # Score par pilier
    pillar_scores = (
        df.groupby("Pilier")["Score /10"]
        .apply(lambda s: float(s.dropna().mean()) if len(s.dropna()) else None)
        .reset_index()
        .rename(columns={"Score /10": "Score moyen /10"})
    )

    return df, pillar_scores


def compute_peer_distribution(peer_tickers: list, api_key: str) -> dict:
    """
    Calcule une distribution de ratios à partir d'une liste de tickers EODHD (comparables).
    Retourne un dict {ratio_key: [values...]}.
    """
    dist = {m["key"]: [] for m in HEALTH_METRICS}

    for t in peer_tickers:
        t = (t or "").strip()
        if not t:
            continue

        try:
            fundamentals_p = fetch_fundamentals_cached(t, api_key)
            net_debt_p = get_net_debt(fundamentals_p)
            shares_p = get_shares_outstanding(fundamentals_p)

            y_bs, bs_p = extract_balance_sheet_snapshot(fundamentals_p)
            y_is, is_p = extract_income_snapshot(fundamentals_p)
            y_cf, cf_p = extract_cashflow_snapshot(fundamentals_p)
            hist_p = build_historical_table(fundamentals_p, max_years=5)

            ratios_p = compute_company_health_ratios(
                company={},
                bs_snap=bs_p,
                is_snap=is_p,
                cf_snap=cf_p,
                net_debt=net_debt_p,
                shares=shares_p,
                hist_df=hist_p,
            )

            for k in dist.keys():
                dist[k].append(ratios_p.get(k))

        except Exception:
            # On ignore les peers qui échouent (données manquantes, erreur API, etc.)
            continue

    return dist


def fetch_sector_peers_via_screener(
    api_key: str,
    sector: str,
    exchange: str | None,
    limit: int = 25,
    exclude_ticker: str | None = None,
):
    """
    Récupère une liste de tickers EODHD via l'endpoint Screener, filtrés par secteur (+ exchange si fourni).

    - Retour: liste de tickers au format CODE.EXCHANGE (ex: AAPL.US).
    - Proxy sectoriel: ce n'est PAS un jugement qualitatif des comparables.
    - Si l'endpoint screener n'est pas accessible (plan), retourne [].

    L'API Screener accepte des filtres sector/industry et exchange. (Chaque requête consomme des API calls.)
    """
    if not sector:
        return []

    exch = (exchange or "").lower().strip()
    sec = str(sector).strip()

    # L'API screener recommande "match" pour les secteurs à deux mots.
    op = "match" if " " in sec else "="

    filters = []
    if exch:
        filters.append(["exchange", "=", exch])
    filters.append(["sector", op, sec])

    params = {
        "api_token": api_key,
        "sort": "market_capitalization.desc",
        "filters": json.dumps(filters),
        "limit": int(max(1, min(100, limit))),
        "offset": 0,
    }

    try:
        url = f"{EODHD_BASE_URL}/screener"
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []

        out = []
        for row in data:
            if not isinstance(row, dict):
                continue
            code_ = row.get("code") or row.get("Code") or row.get("ticker") or row.get("Ticker")
            exch_ = row.get("exchange") or row.get("Exchange") or exch
            if not code_ or not exch_:
                continue
            t = f"{str(code_).upper()}.{str(exch_).upper()}"
            if exclude_ticker and t.upper() == exclude_ticker.upper():
                continue
            out.append(t)

        # Dé-doublonnage en conservant l'ordre
        seen = set()
        uniq = []
        for t in out:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq
    except Exception:
        return []

def style_health_table(df: pd.DataFrame):
    """
    Renvoie un Styler Streamlit-friendly : applique une couleur de fond au score.
    """
    if df is None or df.empty:
        return df

    def _apply(row):
        color = row.get("_color", "#E5E7EB")
        return [""] * (len(row) - 1) + [""]

    # Mise en forme : colorer uniquement la colonne Score /10
    def color_score(val, row_color):
        return f"background-color: {row_color};"

    def apply_colors(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        if "Score /10" in data.columns and "_color" in data.columns:
            for i in data.index:
                styles.loc[i, "Score /10"] = f"background-color: {data.loc[i, '_color']};"
        return styles

    df_disp = df.drop(columns=["_color"], errors="ignore").copy()

    # Formats
    if "Valeur" in df_disp.columns:
        # on ne force pas de format global; Streamlit affichera proprement
        pass

    return df_disp.style.apply(apply_colors, axis=None)


# =========================================
# MOTEUR DCF
# =========================================

def project_fcf(fcf_start: float, growth_rate: float, years: int):
    """
    Projette un FCF sur 'years' années avec une croissance annuelle constante.
    Retourne la liste FCF1...FCFn.
    """
    fcfs = []
    current_fcf = fcf_start
    for _ in range(1, years + 1):
        current_fcf *= (1 + growth_rate)
        fcfs.append(current_fcf)
    return fcfs


def discount_cash_flows(fcfs, wacc: float):
    """
    Actualise une liste de FCF au WACC. Retourne (liste actualisée, somme).
    """
    discounted = []
    total = 0.0
    for t, fcf in enumerate(fcfs, start=1):
        pv = fcf / ((1 + wacc) ** t)
        discounted.append(pv)
        total += pv
    return discounted, total


def terminal_value(last_fcf: float, wacc: float, g: float):
    """
    Valeur terminale (Gordon-Shapiro) : TV = FCF_{n+1} / (WACC - g)
    """
    fcf_next = last_fcf * (1 + g)
    if wacc <= g:
        return None
    return fcf_next / (wacc - g)


def dcf_fair_value_per_share(
    fcf_start: float,
    growth_fcf: float,
    years: int,
    wacc: float,
    g_terminal: float,
    net_debt: float,
    shares: float,
):
    """
    Calcule une juste valeur par action pour un ensemble de paramètres DCF.
    Retourne (fair_value_per_share, EV, equity_value, tv_discounted, sum_discounted_fcfs).
    """
    if shares is None or shares <= 0 or fcf_start is None:
        return None, None, None, None, None

    projected_fcfs = project_fcf(fcf_start, growth_fcf, years)
    discounted_fcfs, sum_discounted_fcfs = discount_cash_flows(projected_fcfs, wacc)

    tv = terminal_value(projected_fcfs[-1], wacc, g_terminal)
    if tv is None:
        return None, None, None, None, None

    tv_discounted = tv / ((1 + wacc) ** years)
    ev = sum_discounted_fcfs + tv_discounted

    net_debt_used = net_debt if net_debt is not None else 0.0
    equity_value = ev - net_debt_used
    fair_value_per_share = equity_value / shares

    return fair_value_per_share, ev, equity_value, tv_discounted, sum_discounted_fcfs


def build_sensitivity_matrix(
    fcf_start: float,
    growth_fcf: float,
    years: int,
    base_wacc: float,
    base_g: float,
    net_debt: float,
    shares: float,
):
    """
    Construit une matrice de sensibilité DCF en faisant varier WACC et g.
    Les cellules contiennent la juste valeur par action.
    """
    wacc_values = sorted(
        {
            max(0.01, base_wacc - 0.01),
            max(0.01, base_wacc - 0.005),
            base_wacc,
            base_wacc + 0.005,
            base_wacc + 0.01,
        }
    )
    g_values = sorted(
        {
            max(0.0, base_g - 0.005),
            base_g,
            base_g + 0.005,
        }
    )

    g_values = [g for g in g_values if g < max(wacc_values)]

    data = {}
    for g in g_values:
        row = []
        for w in wacc_values:
            fv, _, _, _, _ = dcf_fair_value_per_share(
                fcf_start=fcf_start,
                growth_fcf=growth_fcf,
                years=years,
                wacc=w,
                g_terminal=g,
                net_debt=net_debt,
                shares=shares,
            )
            row.append(fv if fv is not None else float("nan"))
        data[f"g = {g*100:.2f} %"] = row

    index_labels = [f"WACC = {w*100:.2f} %" for w in wacc_values]
    df_matrix = pd.DataFrame(data, index=index_labels)
    return df_matrix


# =========================================
# MÉTHODES PAR MULTIPLES (FAIR VALUES)
# =========================================

def pe_valuation(eps, pe_target):
    if eps is None or pe_target is None:
        return None
    return eps * pe_target


def pb_valuation(bvps, pb_target):
    if bvps is None or pb_target is None:
        return None
    return bvps * pb_target


def ev_ebitda_valuation(ebitda, net_debt, shares, ev_ebitda_target):
    if ebitda is None or ev_ebitda_target is None or shares in (None, 0):
        return None
    ev_target = ebitda * ev_ebitda_target
    equity_target = ev_target - (net_debt or 0)
    return equity_target / shares


def ev_ebit_valuation(ebit, net_debt, shares, ev_ebit_target):
    if ebit is None or ev_ebit_target is None or shares in (None, 0):
        return None
    ev_target = ebit * ev_ebit_target
    equity_target = ev_target - (net_debt or 0)
    return equity_target / shares


def ev_sales_valuation(sales, net_debt, shares, ev_sales_target):
    if sales is None or ev_sales_target is None or shares in (None, 0):
        return None
    ev_target = sales * ev_sales_target
    equity_target = ev_target - (net_debt or 0)
    return equity_target / shares


# =========================================
# CLASSIFICATION & PONDÉRATION
# =========================================

def compute_revenue_cagr(hist_df: pd.DataFrame):
    """
    Calcule un CAGR approximatif du chiffre d'affaires si possible.
    """
    if hist_df is None or hist_df.empty or "Chiffre d'affaires" not in hist_df.columns:
        return None

    df = hist_df.dropna(subset=["Chiffre d'affaires"])
    if df.shape[0] < 2:
        return None

    df = df.sort_values("Année")
    rev_start = df["Chiffre d'affaires"].iloc[0]
    rev_end = df["Chiffre d'affaires"].iloc[-1]

    if rev_start in (None, 0) or rev_end is None:
        return None

    n_years = df.shape[0] - 1
    try:
        cagr = (rev_end / rev_start) ** (1 / n_years) - 1
        return cagr
    except Exception:
        return None


def classify_company_profile(company: dict, base_metrics: dict, hist_df: pd.DataFrame):
    """
    Classe la société en Small / Mid / Large + quelques infos utiles.
    """
    sector = (company.get("Sector") or "").lower()
    market_cap = base_metrics.get("market_cap")
    revenue = base_metrics.get("revenue")
    ebit = base_metrics.get("ebit")

    # Capitalisation (seuils plus réalistes)
    if market_cap is None:
        cap_size = "Unknown"
    elif market_cap < 2_000_000_000:        # < 2 Mds
        cap_size = "SmallCap"
    elif market_cap < 10_000_000_000:       # 2–10 Mds
        cap_size = "MidCap"
    else:
        cap_size = "LargeCap"

    # Marge EBIT
    ebit_margin = None
    if ebit is not None and revenue not in (None, 0):
        ebit_margin = ebit / revenue

    # Croissance CA
    rev_cagr = compute_revenue_cagr(hist_df)

    # Tag sectoriel spécial pour financières
    is_financial = any(
        kw in sector
        for kw in ["financial", "bank", "insurance", "assurance"]
    )

    profile = {
        "sector": company.get("Sector"),
        "cap_size": cap_size,
        "market_cap": market_cap,
        "ebit_margin": ebit_margin,
        "revenue_cagr": rev_cagr,
        "is_financial": is_financial,
    }
    return profile


def get_valuation_weights(profile: dict):
    """
    Pondérations par défaut des méthodes selon le profil de la société.
    Les poids sont des hypothèses modélisées, pas des vérités.
    """
    cap_size = profile.get("cap_size", "Unknown")
    is_financial = profile.get("is_financial", False)

    # Cas spéciaux : financières → P/B et P/E
    if is_financial:
        return {
            "DCF": 0.0,
            "PE": 0.3,
            "EV_EBITDA": 0.0,
            "EV_EBIT": 0.0,
            "EV_SALES": 0.0,
            "PB": 0.7,
        }

    if cap_size == "LargeCap":
        return {
            "DCF": 0.6,
            "PE": 0.15,
            "EV_EBITDA": 0.25,
            "EV_EBIT": 0.0,
            "EV_SALES": 0.0,
            "PB": 0.0,
        }
    elif cap_size == "MidCap":
        return {
            "DCF": 0.4,
            "PE": 0.2,
            "EV_EBITDA": 0.3,
            "EV_EBIT": 0.0,
            "EV_SALES": 0.1,
            "PB": 0.0,
        }
    elif cap_size == "SmallCap":
        return {
            "DCF": 0.0,    # DCF considéré non pertinent ici
            "PE": 0.2,
            "EV_EBITDA": 0.3,
            "EV_EBIT": 0.0,
            "EV_SALES": 0.5,
            "PB": 0.0,
        }

    # Valeur par défaut si on ne sait pas classifier
    return {
        "DCF": 0.4,
        "PE": 0.2,
        "EV_EBITDA": 0.2,
        "EV_EBIT": 0.0,
        "EV_SALES": 0.2,
        "PB": 0.0,
    }

def clamp(x, lo, hi):
    if x is None:
        return None
    return max(lo, min(hi, x))


def safe_positive(x):
    return x is not None and x > 0


def default_target_multiples(profile: dict, base_metrics: dict):
    """
    Cibles de multiples plus réalistes (value/neutre/prudent) :

    Objectif principal :
    - éviter que toutes les fair values ≈ Price * k (effet mécanique quand target = current * coeff identique)
    - garder des bornes réalistes
    - rester robuste si certains multiples courants ne sont pas exploitables
    - IMPORTANT : clés en MAJUSCULES (compatibles avec compute_multiples_valuations et l'affichage)
    """

    style = (profile or {}).get("style", "Core")           # "Value", "Core", "Growth"
    quality = (profile or {}).get("quality", "Normal")     # "High", "Normal", "Low"
    growth = (profile or {}).get("growth", "Normal")       # "High", "Normal", ...
    sector = (profile or {}).get("sector", None)

    pe_current = base_metrics.get("pe")
    pb_current = base_metrics.get("pb")
    ev_ebitda_current = base_metrics.get("ev_ebitda")
    ev_ebit_current = base_metrics.get("ev_ebit")
    ev_sales_current = base_metrics.get("ev_sales")

    # -----------------------------
    # 1) Tier
    # -----------------------------
    tier = "core"
    if quality == "High" and growth in ("High", "Structural", "AboveAverage"):
        tier = "quality_growth"
    elif style == "Growth":
        tier = "growth"
    elif style == "Value":
        tier = "value"
    elif quality == "Low":
        tier = "low_quality"

    # -----------------------------
    # 2) Ajustements DIFFÉRENTS par méthode
    # (c’est ça qui casse l’effet "tout = Price * k")
    # -----------------------------
    tier_mult = {
        "low_quality": {
            "PE": 0.85, "PB": 0.80, "EV_EBITDA": 0.85, "EV_EBIT": 0.80, "EV_SALES": 0.90
        },
        "value": {
            "PE": 0.92, "PB": 0.88, "EV_EBITDA": 0.93, "EV_EBIT": 0.90, "EV_SALES": 0.95
        },
        "core": {
            "PE": 1.05, "PB": 1.00, "EV_EBITDA": 1.03, "EV_EBIT": 1.00, "EV_SALES": 1.04
        },
        "growth": {
            "PE": 1.18, "PB": 1.10, "EV_EBITDA": 1.12, "EV_EBIT": 1.10, "EV_SALES": 1.12
        },
        "quality_growth": {
            "PE": 1.28, "PB": 1.18, "EV_EBITDA": 1.18, "EV_EBIT": 1.15, "EV_SALES": 1.16
        },
    }
    m = tier_mult[tier]

    # -----------------------------
    # 3) Bornes prudentes
    # -----------------------------
    software_like = sector in ("Technology", "Software", "Information Technology")

    pe_bounds = (10, 40) if not software_like else (15, 55)
    pb_bounds = (0.6, 8.0) if not software_like else (1.0, 15.0)
    ev_ebitda_bounds = (6, 22) if not software_like else (9, 35)
    ev_ebit_bounds = (7, 26) if not software_like else (10, 45)
    ev_sales_bounds = (0.6, 6.0) if not software_like else (1.2, 12.0)

    targets = {}

    # -----------------------------
    # 4) Construction des cibles
    # -> On ancre sur le courant si exploitable, sinon on fallback sur une cible "absolue"
    # -----------------------------
    # PE
    if safe_positive(pe_current):
        targets["PE"] = clamp(pe_current * m["PE"], *pe_bounds)
    else:
        base_pe = 14 if tier in ("value", "low_quality") else 18
        if tier in ("growth", "quality_growth"):
            base_pe = 24 if software_like else 20
        targets["PE"] = clamp(base_pe, *pe_bounds)

    # PB
    if safe_positive(pb_current):
        targets["PB"] = clamp(pb_current * m["PB"], *pb_bounds)
    else:
        base_pb = 1.2 if tier in ("value", "low_quality") else 1.8
        if tier in ("growth", "quality_growth"):
            base_pb = 4.5 if software_like else 2.5
        targets["PB"] = clamp(base_pb, *pb_bounds)

    # EV/EBITDA
    if safe_positive(ev_ebitda_current):
        targets["EV_EBITDA"] = clamp(ev_ebitda_current * m["EV_EBITDA"], *ev_ebitda_bounds)
    else:
        base = 9 if tier in ("value", "low_quality") else 12
        if tier in ("growth", "quality_growth"):
            base = 20 if software_like else 15
        targets["EV_EBITDA"] = clamp(base, *ev_ebitda_bounds)

    # EV/EBIT
    if safe_positive(ev_ebit_current):
        targets["EV_EBIT"] = clamp(ev_ebit_current * m["EV_EBIT"], *ev_ebit_bounds)
    else:
        base = 11 if tier in ("value", "low_quality") else 14
        if tier in ("growth", "quality_growth"):
            base = 24 if software_like else 17
        targets["EV_EBIT"] = clamp(base, *ev_ebit_bounds)

    # EV/Sales
    if safe_positive(ev_sales_current):
        targets["EV_SALES"] = clamp(ev_sales_current * m["EV_SALES"], *ev_sales_bounds)
    else:
        base = 1.4 if tier in ("value", "low_quality") else 2.0
        if tier in ("growth", "quality_growth"):
            base = 6.0 if software_like else 3.0
        targets["EV_SALES"] = clamp(base, *ev_sales_bounds)

    return targets




def compute_multiples_valuations(base_metrics: dict, net_debt, shares, targets: dict, base_financials: dict = None):
    """
    Calcule les fair values par méthode de multiples en utilisant les cibles.
    Retourne un dict par méthode : multiple courant, multiple cible, fair value.

    IMPORTANT :
    - base_metrics = ratios déjà calculés (PE, EV/EBITDA, EV/Sales, PB...)
    - base_financials = agrégats comptables bruts (revenue, ebitda, ebit, net_income, book_equity)
      -> utilisé pour décider si un multiple est interprétable.
    """

    price = None
    if base_metrics.get("market_cap") is not None and shares not in (None, 0):
        price = base_metrics["market_cap"] / shares

    # Agrégats : on prend en priorité base_metrics s'ils existent,
    # sinon fallback sur base_financials (cas le plus fréquent chez toi)
    base_financials = base_financials or {}

    eps = base_metrics.get("eps")
    bvps = base_metrics.get("bvps")

    revenue = base_metrics.get("revenue")
    if revenue is None:
        revenue = base_financials.get("revenue")

    ebitda = base_metrics.get("ebitda")
    if ebitda is None:
        ebitda = base_financials.get("ebitda")

    ebit = base_metrics.get("ebit")
    if ebit is None:
        ebit = base_financials.get("ebit")

    current = {
        "PE": base_metrics.get("pe"),
        "EV_EBITDA": base_metrics.get("ev_ebitda"),
        "EV_EBIT": base_metrics.get("ev_ebit"),
        "EV_SALES": base_metrics.get("ev_sales"),
        "PB": base_metrics.get("pb"),
    }

    # Nettoyage marché : on désactive uniquement les méthodes réellement non interprétables
    targets_clean = dict(targets or {})

    # PE : EPS doit être > 0
    if eps is None or eps <= 0:
        targets_clean["PE"] = None

    # PB : BVPS doit être > 0
    if bvps is None or bvps <= 0:
        targets_clean["PB"] = None

    # EV/EBITDA : EBITDA doit être > 0
    if ebitda is None or ebitda <= 0:
        targets_clean["EV_EBITDA"] = None

    # EV/EBIT : EBIT doit être > 0
    if ebit is None or ebit <= 0:
        targets_clean["EV_EBIT"] = None

    # EV/Sales : Revenue doit être > 0
    if revenue is None or revenue <= 0:
        targets_clean["EV_SALES"] = None

    # Valorisations
    fair_pe = pe_valuation(eps, targets_clean.get("PE"))
    fair_pb = pb_valuation(bvps, targets_clean.get("PB"))

    fair_ev_ebitda = ev_ebitda_valuation(
        ebitda, net_debt, shares, targets_clean.get("EV_EBITDA")
    )
    fair_ev_ebit = ev_ebit_valuation(
        ebit, net_debt, shares, targets_clean.get("EV_EBIT")
    )
    fair_ev_sales = ev_sales_valuation(
        revenue, net_debt, shares, targets_clean.get("EV_SALES")
    )

    valuations = {
        "PE": {
            "current_multiple": current["PE"],
            "target_multiple": targets_clean.get("PE"),
            "fair_value": fair_pe,
        },
        "PB": {
            "current_multiple": current["PB"],
            "target_multiple": targets_clean.get("PB"),
            "fair_value": fair_pb,
        },
        "EV_EBITDA": {
            "current_multiple": current["EV_EBITDA"],
            "target_multiple": targets_clean.get("EV_EBITDA"),
            "fair_value": fair_ev_ebitda,
        },
        "EV_EBIT": {
            "current_multiple": current["EV_EBIT"],
            "target_multiple": targets_clean.get("EV_EBIT"),
            "fair_value": fair_ev_ebit,
        },
        "EV_SALES": {
            "current_multiple": current["EV_SALES"],
            "target_multiple": targets_clean.get("EV_SALES"),
            "fair_value": fair_ev_sales,
        },
    }

    return valuations



def combine_global_valuation(dcf_value: float, multiples_vals: dict, weights: dict, price: float):
    """
    Combine DCF + multiples avec pondération automatique.
    Ne prend en compte que les méthodes pour lesquelles on a une fair value.
    """
    contributions = []
    total_weight_used = 0.0

    method_labels = {
        "DCF": "DCF (intrinsèque)",
        "PE": "P/E",
        "PB": "P/B",
        "EV_EBITDA": "EV/EBITDA",
        "EV_EBIT": "EV/EBIT",
        "EV_SALES": "EV/Sales",
    }

    details = []

    for key, w in weights.items():
        if w <= 0:
            continue

        if key == "DCF":
            fv = dcf_value
        else:
            info = multiples_vals.get(key)
            fv = info.get("fair_value") if info else None

        if fv is None:
            continue

        contributions.append(fv * w)
        total_weight_used += w

        upside = None
        if price not in (None, 0):
            upside = (fv / price - 1) * 100

        details.append(
            {
                "Méthode": method_labels.get(key, key),
                "Poids utilisé": w,
                "Fair value / action": fv,
                "Upside (%)": upside,
            }
        )

    if total_weight_used == 0:
        return {
            "fair_value_global": None,
            "details": [],
        }

    fair_value_global = sum(contributions) / total_weight_used

    if price not in (None, 0):
        upside_global = (fair_value_global / price - 1) * 100
    else:
        upside_global = None

    return {
        "fair_value_global": fair_value_global,
        "upside_global": upside_global,
        "details": details,
        "weight_sum_used": total_weight_used,
    }


# =========================================
# PIPELINE PRINCIPAL POUR UNE SOCIÉTÉ
# =========================================
def analyze_company(query: str, api_key: str, years: int, wacc: float, growth_fcf: float, g_terminal: float):
    """
    Pipeline complet :
    - Recherche par nom/ticker
    - Résolution du ticker EODHD (priorité pays d'origine)
    - Récupération fondamentaux + prix
    - Extraction des tableaux historiques
    - Multiples & profil société
    - DCF (uniquement si pertinent)
    - Classification + pondération + synthèse globale
    """
    ticker = None
    search_results = []

    # =========================
    # Résolution du ticker
    # =========================
    if "." in query and " " not in query:
        ticker = query.strip()
    else:
        search_results = search_instrument(query.strip(), api_key)
        if not search_results:
            raise ValueError("Aucun instrument trouvé pour cette recherche.")

        # Pays préféré (mapping simple pour éviter les mauvaises cotations)
        q = query.lower().strip()
        preferred_country = None

        # France (Euronext Paris)
        if q in ["airbus", "lvmh", "total", "totalenergies", "sanofi", "danone", "vinci", "schneider", "orange", "axa"]:
            preferred_country = "France"
        # Allemagne
        elif q in ["siemens", "sap", "bmw", "mercedes", "volkswagen", "basf"]:
            preferred_country = "Germany"
        # UK
        elif q in ["shell", "bp", "unilever", "astra", "astrazeneca"]:
            preferred_country = "United Kingdom"
        # Suisse
        elif q in ["nestle", "roche", "novartis"]:
            preferred_country = "Switzerland"
        # USA (facultatif : utile si tu veux forcer les tickers US)
        elif q in ["apple", "microsoft", "amazon", "google", "alphabet", "meta", "tesla", "nvidia", "berkshire"]:
            preferred_country = "USA"

        # ✅ Résolution robuste (priorité pays + exchanges majeurs + market cap)
        ticker = resolve_best_ticker(search_results, api_key, preferred_country)

        if ticker is None:
            raise ValueError("Impossible de construire un ticker valide à partir du résultat de recherche.")

    # =========================
    # Prix de marché
    # =========================
    price = fetch_eod_price(ticker, api_key)
    if price is None:
        raise ValueError("Impossible de récupérer le cours de marché.")

    # =========================
    # Fondamentaux bruts
    # =========================
    fundamentals = fetch_fundamentals(ticker, api_key)
    company = get_company_summary(fundamentals)
    shares = get_shares_outstanding(fundamentals)
    net_debt = get_net_debt(fundamentals)
    hist_df = build_historical_table(fundamentals, max_years=5)

    # =========================
    # Snapshots fondamentaux (bilan / IS / CF) + ratios santé
    # =========================
    bs_year, bs_snap = extract_balance_sheet_snapshot(fundamentals)
    is_year, is_snap = extract_income_snapshot(fundamentals)
    cf_year, cf_snap = extract_cashflow_snapshot(fundamentals)

    health_ratios = compute_company_health_ratios(
        company=company,
        bs_snap=bs_snap,
        is_snap=is_snap,
        cf_snap=cf_snap,
        net_debt=net_debt,
        shares=shares,
        hist_df=hist_df,
    )

    # =========================
    # Multiples & profil
    # =========================
    base_financials = extract_base_financials(fundamentals)
    base_metrics = compute_base_multiples(price, shares, net_debt, base_financials)
    profile = classify_company_profile(company, base_metrics, hist_df)

    # =========================
    # Estimation du FCF de départ (pour DCF éventuel)
    # =========================
    fcf_last = estimate_starting_fcf(fundamentals)
    fcf_norm = estimate_normalized_fcf(hist_df)

    fcf_start = fcf_norm if fcf_norm is not None else fcf_last

    # =========================
    # DCF : seulement si la société n'est PAS small cap et si données suffisantes
    # =========================
    dcf_allowed = (
        profile.get("cap_size") != "SmallCap"
        and fcf_start is not None
        and shares not in (None, 0)
    )

    if dcf_allowed:
        fv_dcf, ev, equity_value, tv_discounted, sum_disc_fcfs = dcf_fair_value_per_share(
            fcf_start=fcf_start,
            growth_fcf=growth_fcf,
            years=years,
            wacc=wacc,
            g_terminal=g_terminal,
            net_debt=net_debt,
            shares=shares,
        )

        if fv_dcf is not None and price not in (None, 0):
            upside_dcf = (fv_dcf / price - 1) * 100
        else:
            upside_dcf = None

        # Projections FCF & sensibilité
        projected_fcfs = project_fcf(fcf_start, growth_fcf, years)
        discounted_fcfs, _ = discount_cash_flows(projected_fcfs, wacc)
        proj_df = pd.DataFrame(
            {
                "Année": [f"Année {i}" for i in range(1, years + 1)],
                "FCF projeté": projected_fcfs,
                "FCF actualisé": discounted_fcfs,
            }
        )

        sens_matrix = build_sensitivity_matrix(
            fcf_start=fcf_start,
            growth_fcf=growth_fcf,
            years=years,
            base_wacc=wacc,
            base_g=g_terminal,
            net_debt=net_debt,
            shares=shares,
        )
    else:
        # DCF non pertinent ou impossible → on neutralise toutes les sorties DCF
        fv_dcf = None
        ev = None
        equity_value = None
        tv_discounted = None
        sum_disc_fcfs = None
        upside_dcf = None
        proj_df = pd.DataFrame()
        sens_matrix = pd.DataFrame()

    # =========================
    # Multiples : cibles & valorisations (INCHANGÉ)
    # =========================
    weights = get_valuation_weights(profile)
    targets = default_target_multiples(profile, base_metrics)
    multiples_vals = compute_multiples_valuations(base_metrics, net_debt, shares, targets, base_financials)

    # Synthèse globale DCF + multiples (si DCF absent, la pondération DCF est simplement ignorée)
    global_val = combine_global_valuation(
        dcf_value=fv_dcf,
        multiples_vals=multiples_vals,
        weights=weights,
        price=price,
    )

    return {
        "ticker": ticker,
        "search_results": search_results,
        "company": company,
        "price": price,
        "shares": shares,
        "net_debt": net_debt,
        "hist_df": hist_df,
        "health": {
            "bs_year": bs_year,
            "is_year": is_year,
            "cf_year": cf_year,
            "balance_sheet": bs_snap,
            "income_statement": is_snap,
            "cash_flow": cf_snap,
            "ratios": health_ratios,
        },
        "fcf_start": fcf_start,
        "proj_df": proj_df,
        "dcf": {
            "fair_value_per_share": fv_dcf,
            "ev": ev,
            "equity_value": equity_value,
            "tv_discounted": tv_discounted,
            "sum_disc_fcfs": sum_disc_fcfs,
            "upside_pct": upside_dcf,
        },
        "sensitivity": sens_matrix,
        "base_financials": base_financials,
        "base_metrics": base_metrics,
        "profile": profile,
        "weights": weights,
        "target_multiples": targets,
        "multiples_valuations": multiples_vals,
        "global_valuation": global_val,
    }


# =========================================
# STREAMLIT APP
# =========================================

def main():
    st.set_page_config(
        page_title="DCF Valuation Pro - EODHD",
        layout="wide"
    )

    # En-tête
    st.markdown(
        """
        <div style="
            background-color:#0F172A;
            padding:1.5rem 1rem;
            border-radius:1rem;
            margin-bottom:1.5rem;
        ">
            <h1 style="color:white; margin:0;">📊 Application professionnelle de valorisation DCF & Multiples</h1>
            <p style="color:#E5E7EB; margin:0.3rem 0 0;">
                Analyse fondamentale d'une société via l'API EODHD : données historiques, projections sur 5 ans,
                valorisation DCF détaillée, méthodes par multiples et synthèse globale pondérée.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("⚙️ Paramètres généraux")

    api_key = get_api_key()

    query = st.sidebar.text_input(
        "Nom de la société ou ticker (ex : 'LVMH', 'AAPL.US', 'Airbus')",
        value="AAPL.US"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Paramètres DCF (base case)")

    years = st.sidebar.slider(
        "Horizon de projection (années)",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )
    wacc_input = st.sidebar.number_input(
        "WACC (%)",
        min_value=5.5,
        max_value=8.0,
        value=6.8,
        step=0.1,
    )
    growth_fcf_input = st.sidebar.number_input(
        "Croissance annuelle FCF (%)",
        min_value=-1.5,
        max_value=4.0,
        value=3.0,
        step=0.1,
    )
    g_terminal_input = st.sidebar.number_input(
        "Croissance long terme g (%)",
        min_value=1.0,
        max_value=2.0,
        value=1.75,
        step=0.05,
    )

    wacc = wacc_input / 100.0

    st.sidebar.markdown('---')
    st.sidebar.header('Benchmark comparables (optionnel)')
    peer_tickers_raw = st.sidebar.text_input(
        "Tickers EODHD comparables (ex: MSFT.US, GOOGL.US, AMZN.US)",
        value="",
    )

    auto_sector_benchmark = st.sidebar.checkbox(
        "Auto comparables (même secteur via Screener EODHD)",
        value=False,
        help="Utilise /api/screener pour générer des comparables du même secteur (proxy). "
             "Nécessite que le Screener soit disponible sur ton plan et consomme des appels API.",
    )
    auto_peers_limit = st.sidebar.slider(
        "Nombre de comparables (auto)",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
    )

    wacc = wacc_input / 100.0
    growth_fcf = growth_fcf_input / 100.0
    g_terminal = g_terminal_input / 100.0

    if not api_key:
        st.warning("➡️ Saisis ta clé API EODHD dans la sidebar pour lancer une analyse.")
        return

    col_query, col_btn = st.columns([3, 1])
    with col_query:
        st.text_input(
            "Rappel : société / ticker analysé",
            value=query,
            disabled=True,
        )
    with col_btn:
        run_button = st.button("Analyser la société", type="primary")

    if not run_button:
        st.stop()

    try:
        with st.spinner("Analyse en cours..."):
            result = analyze_company(
                query=query,
                api_key=api_key,
                years=years,
                wacc=wacc,
                growth_fcf=growth_fcf,
                g_terminal=g_terminal,
            )
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
        st.stop()
        return

    # =========================================
    # MISE EN PAGE AVEC TABS
    # =========================================

    company = result["company"]
    dcf = result["dcf"]
    profile = result["profile"]
    base_metrics = result["base_metrics"]
    multiples_vals = result["multiples_valuations"]
    global_val = result["global_valuation"]
    targets = result["target_multiples"]
    weights = result["weights"]
    dcf_active = dcf.get("fair_value_per_share") is not None

    # =========================================
    # TABLEAU SANTÉ FINANCIÈRE (ratios scorés)
    # =========================================
    company_ratios = (result.get("health", {}) or {}).get("ratios", {}) or {}

    peer_distribution = None
peer_list = []

# 1) Comparables manuels (si fournis)
try:
    raw = (peer_tickers_raw or "").strip()
    if raw:
        peer_list = [x.strip() for x in raw.split(",") if x.strip()]
        if peer_list:
            peer_distribution = compute_peer_distribution(peer_list, api_key)
except Exception:
    peer_distribution = None

# 2) Auto comparables sectoriels via Screener (si activé et pas de liste manuelle)
if peer_distribution is None and auto_sector_benchmark:
    try:
        sector = company.get("Sector")
        exchange = (result.get("ticker") or "").split(".")[-1] if result.get("ticker") else None
        auto_peers = fetch_sector_peers_via_screener(
            api_key=api_key,
            sector=sector,
            exchange=exchange,
            limit=auto_peers_limit,
            exclude_ticker=result.get("ticker"),
        )
        if auto_peers:
            peer_list = auto_peers
            peer_distribution = compute_peer_distribution(peer_list, api_key)
    except Exception:
        peer_distribution = None

    health_df_raw, pillar_scores = build_health_table(company_ratios, peer_distribution=peer_distribution)
    health_df_styled = style_health_table(health_df_raw)


    # Bandeau résumé
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Société", company.get("Name", "N/A"))
        st.metric("Ticker EODHD", result["ticker"])
    with col2:
        st.metric("Secteur", company.get("Sector", "N/A"))
        st.metric("Industrie", company.get("Industry", "N/A"))
    with col3:
        st.metric("Pays", company.get("Country", "N/A"))
        st.metric("Devise", company.get("Currency", "N/A"))
    with col4:
        st.metric("Prix de marché", f"{result['price']:.2f}")
        if dcf_active:
            st.metric("Juste valeur DCF", f"{dcf['fair_value_per_share']:.2f}")
        else:
            st.metric("Juste valeur DCF", "N/A")

    upside_pct = dcf.get("upside_pct")
    if dcf_active and upside_pct is not None:
        upside_color = "🟢" if upside_pct > 0 else "🔴"
        st.markdown(
            f"**Upside / Downside DCF :** {upside_color} **{upside_pct:.1f} %**"
        )
    else:
        st.markdown(
            "**DCF non utilisé pour ce profil (small cap ou données insuffisantes).**"
        )


    # Tabs
    tab_health, tab_resume, tab_hist, tab_proj, tab_dcf, tab_mult, tab_synth = st.tabs(
        [
            "Santé financière (ratios)",
            "Résumé DCF",
            "Historique 5 ans",
            "Projections FCF",
            "DCF & Sensibilité",
            "Multiples & Comparables",
            "Synthèse globale",
        ]
    )

    
    # ----- TAB 0 : Santé financière -----
    with tab_health:
        st.subheader("🩺 Santé financière (ratios scorés)")

if peer_distribution is None:
    st.info("Scoring : seuils génériques (non sectorisé). Ajoute des comparables manuels ou active l'auto sectoriel pour obtenir percentiles & médianes.")
else:
    st.success(f"Scoring sectorisé : comparables = {len(peer_list)} (percentiles).")

if peer_list:
    with st.expander("Voir la liste des comparables utilisés"):
        st.write(", ".join(peer_list[:100]))

        # Indication de méthode
        if peer_list:
            st.caption("Scoring basé sur le percentile au sein des comparables fournis (comparables = benchmark).")
        else:
            st.caption("Scoring basé sur des seuils génériques (aucun benchmark comparables/secteur fourni).")

        if health_df_raw is None or health_df_raw.empty:
            st.warning("Aucun ratio exploitable n'a pu être calculé avec les données disponibles.")
        else:
            st.dataframe(health_df_styled, use_container_width=True)

        st.markdown("#### Score moyen par pilier")
        if pillar_scores is not None and not pillar_scores.empty:
            pillar_scores_disp = pillar_scores.copy()
            pillar_scores_disp["Score moyen /10"] = pillar_scores_disp["Score moyen /10"].round(2)
            st.dataframe(pillar_scores_disp, use_container_width=True)
        else:
            st.info("Score par pilier indisponible.")

        # Snapshots (optionnel)
        with st.expander("Voir les données sources (dernier exercice)"):
            h = result.get("health", {}) or {}
            st.write("Année bilan :", h.get("bs_year"))
            st.write("Année compte de résultat :", h.get("is_year"))
            st.write("Année cash-flow :", h.get("cf_year"))
            st.json({
                "Balance Sheet": h.get("balance_sheet", {}),
                "Income Statement": h.get("income_statement", {}),
                "Cash Flow": h.get("cash_flow", {}),
            })


# ----- TAB 1 : Résumé DCF -----
    with tab_resume:
        st.subheader("🎯 Résumé de la valorisation DCF (base case)")

        if not dcf_active:
            st.warning(
                "Le modèle DCF n'est pas utilisé pour cette société "
                "(profil small cap ou données de cash-flow insuffisantes). "
                "Les valorisations reposent principalement sur les méthodes par multiples."
            )
        else:
            ev = dcf.get("ev", 0) or 0
            sum_disc_fcfs = dcf.get("sum_disc_fcfs", 0) or 0
            tv_discounted = dcf.get("tv_discounted", 0) or 0
            equity_value = dcf.get("equity_value", 0) or 0
            fair_value_per_share = dcf.get("fair_value_per_share", 0) or 0

            shares = (result.get("shares", 0) or 0)
            net_debt = (result.get("net_debt", 0) or 0)
            fcf_start = (result.get("fcf_start", 0) or 0)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Valeur d'entreprise (EV)", format_large_number(ev))
                st.metric("Somme FCF actualisés", format_large_number(sum_disc_fcfs))
            with col_b:
                st.metric("Valeur terminale actualisée", format_large_number(tv_discounted))
                st.metric("Valeur des capitaux propres", format_large_number(equity_value))
            with col_c:
                st.metric("Juste valeur / action", f"{fair_value_per_share:,.2f}")
                st.metric("Nombre d'actions", format_large_number(shares))

            st.markdown("#### Hypothèses retenues (base case)")
            st.write(f"- Horizon de projection : **{years} ans**")
            st.write(f"- WACC : **{wacc_input:.2f} %**")
            st.write(f"- Croissance FCF : **{growth_fcf_input:.2f} % par an**")
            st.write(f"- g de long terme : **{g_terminal_input:.2f} %**")
            st.write(f"- Dette nette utilisée : **{format_large_number(net_debt)}**")
            st.write(f"- FCF de départ estimé : **{format_large_number(fcf_start)}**")

            st.info(
                "Ce résumé présente le scénario central (base case). "
                "La robustesse de la valorisation est analysée dans l'onglet « DCF & Sensibilité » "
                "et complétée par les méthodes par multiples."
            )


      # ----- TAB 2 : Historique -----
    with tab_hist:
        st.subheader("📚 Données historiques (5 dernières années)")

        hist_df = result["hist_df"]
        if hist_df.empty:
            st.warning("Impossible de construire un historique complet à partir des données disponibles.")
        else:
            df_display = scale_df_to_millions(hist_df)
            numeric_cols = [c for c in df_display.columns if c != "Année"]
            for c in numeric_cols:
                df_display[c] = df_display[c].astype(float).round(2)

            st.dataframe(df_display, use_container_width=True)
            st.caption("Unités : millions de la devise de reporting.")

    # ----- TAB 3 : Projections FCF -----
    with tab_proj:
        st.subheader("📈 Projections de FCF sur 5 ans (base case)")

        proj_df = result["proj_df"]

        if (not dcf_active) or (proj_df is None) or proj_df.empty:
            st.warning(
                "Projections de FCF non disponibles ou non pertinentes pour cette société "
                "(profil small cap ou absence de données suffisantes)."
            )
        else:
            proj_df = proj_df.copy()
            proj_df["FCF projeté"] = proj_df["FCF projeté"].round(0)
            proj_df["FCF actualisé"] = proj_df["FCF actualisé"].round(0)
            st.dataframe(proj_df, use_container_width=True)

            st.markdown(
                "Les projections sont basées sur un FCF de départ estimé à partir du dernier "
                "**Operating Cash Flow - Capex**, et une croissance constante de "
                f"**{growth_fcf_input:.2f} %/an**."
            )

    # ----- TAB 4 : DCF & Sensibilité -----
    with tab_dcf:
        st.subheader("🧮 DCF détaillé et matrice de sensibilité")

        sens_df = result["sensitivity"]

        if (not dcf_active) or (sens_df is None) or sens_df.empty:
            st.warning(
                "Matrice de sensibilité DCF non disponible pour cette société "
                "(profil small cap ou données cash-flow insuffisantes)."
            )
        else:
            st.markdown("#### Matrice de sensibilité (juste valeur / action)")
            sens_df = sens_df.round(2)
            st.dataframe(sens_df, use_container_width=True)

            st.markdown(
                """
                - **Lignes** : différentes hypothèses de WACC autour de la valeur de base.  
                - **Colonnes** : différentes hypothèses de croissance de long terme *g*.  
                - Chaque cellule représente la **juste valeur par action** selon ces hypothèses.
                """
            )

        st.markdown("#### Rappel des paramètres du scénario central")
        st.write(f"- WACC base : **{wacc_input:.2f} %**")
        st.write(f"- g base : **{g_terminal_input:.2f} %**")
        st.write(f"- Croissance FCF : **{growth_fcf_input:.2f} %/an**")
        st.write(f"- Horizon : **{years} ans**")

        st.info(
            "Le DCF reste la méthode intrinsèque principale pour les sociétés matures "
            "avec des cash-flows prévisibles (notamment large caps). "
            "Pour les small caps, d'autres méthodes (EV/Sales, EV/EBITDA...) sont privilégiées."
        )

    # ----- TAB 5 : Multiples & Comparables -----
    with tab_mult:
        st.subheader("📊 Multiples & valorisations par comparables")

        price = result["price"]
        eps = base_metrics.get("eps")
        bvps = base_metrics.get("bvps")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("EPS (approx)", f"{eps:.2f}" if eps is not None else "N/A")
            st.metric("BVPS (approx)", f"{bvps:.2f}" if bvps is not None else "N/A")
        with col_m2:
            st.metric("P/E courant", f"{base_metrics.get('pe'):.1f}" if base_metrics.get("pe") else "N/A")
            st.metric("P/B courant", f"{base_metrics.get('pb'):.1f}" if base_metrics.get("pb") else "N/A")
        with col_m3:
            st.metric("EV/EBITDA courant", f"{base_metrics.get('ev_ebitda'):.1f}" if base_metrics.get("ev_ebitda") else "N/A")
            st.metric("EV/Sales courant", f"{base_metrics.get('ev_sales'):.1f}" if base_metrics.get("ev_sales") else "N/A")

        st.markdown("#### Multiples cibles et fair values implicites")

        rows = []
        method_order = ["PE", "PB", "EV_EBITDA", "EV_EBIT", "EV_SALES"]
        labels = {
            "PE": "P/E",
            "PB": "P/B",
            "EV_EBITDA": "EV/EBITDA",
            "EV_EBIT": "EV/EBIT",
            "EV_SALES": "EV/Sales",
        }

        for key in method_order:
            info = multiples_vals.get(key)
            if not info:
                continue

            fv = info.get("fair_value")
            if fv is None:
                continue

            cur_mult = info.get("current_multiple")
            tgt_mult = info.get("target_multiple")

            upside = None
            if price not in (None, 0):
                upside = (fv / price - 1) * 100

            rows.append(
                {
                    "Méthode": labels.get(key, key),
                    "Multiple courant": cur_mult,
                    "Multiple cible (hyp.)": tgt_mult,
                    "Fair value / action": fv,
                    "Upside (%)": upside,
                }
            )

        if rows:
            df_mult = pd.DataFrame(rows)
            df_mult["Multiple courant"] = df_mult["Multiple courant"].round(2)
            df_mult["Multiple cible (hyp.)"] = df_mult["Multiple cible (hyp.)"].round(2)
            df_mult["Fair value / action"] = df_mult["Fair value / action"].round(2)
            df_mult["Upside (%)"] = df_mult["Upside (%)"].round(1)

            st.dataframe(df_mult, use_container_width=True)
        else:
            st.warning("Impossible de calculer des valorisations par multiples exploitables pour cette société.")

        st.info(
            "Les multiples cibles affichés sont des hypothèses génériques (à affiner par secteur et par style d'investissement). "
            "L'objectif ici est de montrer la cohérence ou l'écart entre la valorisation DCF et les valorisations relatives."
        )

    # ----- TAB 6 : Synthèse globale -----
    with tab_synth:
        st.subheader("🧷 Synthèse globale de valorisation")

        price = result["price"]
        fair_global = global_val.get("fair_value_global")
        upside_global = global_val.get("upside_global")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Taille de capitalisation", profile.get("cap_size", "N/A"))
            st.metric("Market cap approx.", format_large_number(profile.get("market_cap")))
        with col_s2:
            rev_cagr = profile.get("revenue_cagr")
            if rev_cagr is not None:
                rev_cagr_str = f"{rev_cagr*100:.1f} %/an"
            else:
                rev_cagr_str = "N/A"
            ebit_margin = profile.get("ebit_margin")
            if ebit_margin is not None:
                ebit_margin_str = f"{ebit_margin*100:.1f} %"
            else:
                ebit_margin_str = "N/A"
            st.metric("Croissance CA (approx)", rev_cagr_str)
            st.metric("Marge EBIT (approx)", ebit_margin_str)
        with col_s3:
            st.metric("Juste valeur DCF", f"{dcf['fair_value_per_share']:.2f}")
            st.metric(
                "Juste valeur globale pondérée",
                f"{fair_global:.2f}" if fair_global is not None else "N/A",
            )

        if upside_global is not None:
            color_glob = "🟢" if upside_global > 0 else "🔴"
            st.markdown(
                f"**Upside / Downside global (DCF + multiples pondérés) :** "
                f"{color_glob} **{upside_global:.1f} %**"
            )

        st.markdown("#### Détail des méthodes prises en compte dans la synthèse")

        details = global_val.get("details", [])
        if details:
            df_det = pd.DataFrame(details)
            df_det["Fair value / action"] = df_det["Fair value / action"].round(2)
            df_det["Upside (%)"] = df_det["Upside (%)"].round(1)
            df_det["Poids utilisé"] = df_det["Poids utilisé"].round(2)
            st.dataframe(df_det, use_container_width=True)
        else:
            st.warning(
                "Aucune méthode n'a pu être prise en compte dans la synthèse globale "
                "(problème de données ou pondérations nulles)."
            )

        st.markdown("#### Rappel de la logique de pondération automatique")

        st.write(
            "- **Large Caps** : DCF dominant (60 %), complété par EV/EBITDA et P/E.  \n"
            "- **Mid Caps** : DCF 40 %, EV/EBITDA 30 %, P/E 20 %, EV/Sales 10 %.  \n"
            "- **Small Caps** : DCF neutralisé (0 %), focus sur EV/Sales, EV/EBITDA, P/E.  \n"
            "- **Financières** : P/B et P/E privilégiés, DCF mis de côté."
        )

        st.info(
            "Ces pondérations sont des hypothèses de travail inspirées des pratiques des analystes professionnels. "
            "L'intérêt de ton outil est justement d'expliciter ces choix, de les affiner, "
            "et de montrer que tu sais adapter la méthode au profil de la société analysée."
        )


if __name__ == "__main__":
    main()
