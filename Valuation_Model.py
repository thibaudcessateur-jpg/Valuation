# ============================================================
#                 VALUATION MODEL PRO — PARTIE 1
#          Imports + Configuration + API Utilities
# ============================================================

import os
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------
# CONFIG API
# ---------------------------

EODHD_BASE_URL = "https://eodhd.com/api"


# ============================================================
#                    API UTILITIES
# ============================================================

def get_api_key():
    """
    Récupère la clé API EODHD :
    1) depuis variable d'environnement
    2) sinon depuis sidebar Streamlit
    """
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("🔑 API Key EODHD", type="password")
    return api_key


def search_instrument(query: str, api_key: str, limit: int = 10):
    """
    Recherche un instrument par nom ou ticker.
    Permet à l’utilisateur d'écrire « LVMH », « Apple », « Airbus », etc.
    """
    url = f"{EODHD_BASE_URL}/search/{query}"
    params = {
        "api_token": api_key,
        "limit": limit,
        "fmt": "json",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def build_ticker_from_search_result(item: dict):
    """
    Transforme un résultat de /search en ticker utilisable.
    Exemple : Code=MC, Exchange=PA → "MC.PA"
    """
    code = item.get("Code")
    exch = item.get("Exchange")
    if not code or not exch:
        return None
    return f"{code}.{exch}"


def fetch_eod_price(ticker: str, api_key: str):
    """
    Récupère le dernier cours de clôture.
    """
    url = f"{EODHD_BASE_URL}/eod/{ticker}"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "order": "d",
        "limit": 1,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return data[0].get("close")


def fetch_fundamentals(ticker: str, api_key: str):
    """
    Récupère les fondamentaux d'une société :
    - General
    - Financials (balance sheet, income statement, cash flow)
    - SharesStats
    """
    url = f"{EODHD_BASE_URL}/fundamentals/{ticker}"
    params = {"api_token": api_key}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


# ============================================================
#             FORMAT & UTILITY FUNCTIONS
# ============================================================

def format_large_number(x):
    """
    Format lisible pour grands nombres :
    1250000000 → "1.25 Md"
    45000000   → "45 M"
    """
    if x is None:
        return "N/A"
    try:
        x = float(x)
    except:
        return "N/A"

    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f} Md"
    elif ax >= 1_000_000:
        return f"{x / 1_000_000:.2f} M"
    else:
        return f"{x:.0f}"


def safe_div(a, b):
    """Division sécurisée (retourne None si impossible)."""
    if a is None or b in (None, 0):
        return None
    try:
        return a / b
    except:
        return None
# ============================================================
#            PARTIE 2 — EXTRACTION FINANCIÈRE & MULTIPLES
# ============================================================

# ---------------------------
# Extraction informations générales société
# ---------------------------

def get_company_summary(fundamentals: dict):
    """Retire les infos générales d’une société."""
    gen = fundamentals.get("General", {}) or {}
    return {
        "Name": gen.get("Name"),
        "ISIN": gen.get("ISIN"),
        "Sector": gen.get("Sector"),
        "Industry": gen.get("Industry"),
        "Country": gen.get("CountryName"),
        "Currency": gen.get("CurrencyCode"),
    }


# ---------------------------
# Extraction des états financiers
# ---------------------------

def pick_first_non_null(row: dict, keys):
    """Renvoie la première clé non nulle trouvée dans row."""
    for k in keys:
        if k in row and row[k] not in (None, "", "NaN"):
            try:
                return float(row[k])
            except:
                continue
    return None


def extract_historical_financials(fundamentals: dict, max_years: int = 5):
    """
    Reconstruit un tableau historique multi-années (CA, EBIT, Net Income, OCF, Capex, FCF).
    """
    try:
        inc = fundamentals["Financials"]["Income_Statement"]["yearly"]
        cf = fundamentals["Financials"]["Cash_Flow"]["yearly"]
    except:
        return pd.DataFrame()

    years = sorted(inc.keys(), reverse=True)[:max_years]
    rows = []

    for y in years:
        inc_y = inc.get(y, {}) or {}
        cf_y = cf.get(y, {}) or {}

        revenue = pick_first_non_null(inc_y, ["TotalRevenue", "Revenue", "Sales"])
        ebit    = pick_first_non_null(inc_y, ["OperatingIncome", "EBIT"])
        net_inc = pick_first_non_null(inc_y, ["NetIncome", "NetIncomeCommonStockholders"])

        op_cf   = pick_first_non_null(cf_y, [
            "TotalCashFromOperatingActivities",
            "NetCashProvidedByOperatingActivities",
            "OperatingCashFlow"
        ])

        capex   = pick_first_non_null(cf_y, [
            "CapitalExpenditures",
            "InvestmentsInPropertyPlantAndEquipment"
        ])

        fcf = op_cf - capex if (op_cf is not None and capex is not None) else None

        rows.append({
            "Année": y,
            "Chiffre d'affaires": revenue,
            "EBIT": ebit,
            "Résultat net": net_inc,
            "Op. Cash Flow": op_cf,
            "Capex": capex,
            "FCF (approx)": fcf
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Année")


def extract_financial_base(fundamentals: dict):
    """
    Extrait une dernière année de données comptables utile pour les multiples :
    - Revenue
    - EBITDA (fallback si absent)
    - EBIT
    - Net Income
    - Book Value
    """
    inc = fundamentals.get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    bs  = fundamentals.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})

    revenue = ebitda = ebit = net_income = book_equity = None

    if inc:
        last = sorted(inc.keys())[-1]
        row = inc[last]

        revenue = pick_first_non_null(row, ["TotalRevenue", "Revenue", "Sales"])
        ebit    = pick_first_non_null(row, ["OperatingIncome", "EBIT"])
        net_income = pick_first_non_null(row, ["NetIncome", "NetIncomeCommonStockholders"])

        # EBITDA : fallback si non fourni
        ebitda = pick_first_non_null(row, ["EBITDA", "Ebitda"])
        if ebitda is None and ebit is not None:
            depreciation = pick_first_non_null(row, ["Depreciation", "TotalDepreciation"])
            if depreciation is not None:
                ebitda = ebit + depreciation

    if bs:
        last = sorted(bs.keys())[-1]
        row = bs[last]

        # On prend toutes les variantes Equity possibles
        book_equity = pick_first_non_null(row, [
            "TotalShareholdersEquity",
            "TotalStockholderEquity",
            "TotalEquity",
            "TotalEquityGrossMinorityInterest",
        ])

    return {
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "net_income": net_income,
        "book_equity": book_equity
    }


# ---------------------------
# Shares + Net Debt
# ---------------------------

def extract_shares_outstanding(fundamentals: dict):
    return fundamentals.get("SharesStats", {}).get("SharesOutstanding")


def extract_net_debt(fundamentals: dict):
    """Dette nette = TotalDebt – Cash."""
    try:
        bs = fundamentals["Financials"]["Balance_Sheet"]["yearly"]
    except:
        return None

    last = sorted(bs.keys())[-1]
    row = bs[last]

    debt = pick_first_non_null(row, ["TotalDebt", "ShortLongTermDebtTotal"])
    cash = pick_first_non_null(row, ["CashAndCashEquivalents", "Cash"])

    if debt is None or cash is None:
        return None

    return debt - cash


# ============================================================
#                CLASSIFICATION SMALL / MID / LARGE
# ============================================================

def classify_company(profile_market_cap):
    """
    Classification réaliste :
    - Small cap   : < 2 Mds
    - Mid cap     : 2–10 Mds
    - Large cap   : > 10 Mds
    """
    if profile_market_cap is None:
        return "SmallCap"   # Conservateur : micro-cap ou données manquantes

    if profile_market_cap < 2_000_000_000:
        return "SmallCap"
    elif profile_market_cap < 10_000_000_000:
        return "MidCap"
    else:
        return "LargeCap"


# ============================================================
#                   MULTIPLES + FAIR VALUES
# ============================================================

def compute_base_multiples(price, shares, net_debt, base):
    """
    Calcule EPS, BVPS, Market cap, EV, et tous les multiples de base.
    """
    market_cap = None
    if price is not None and shares not in (None, 0):
        market_cap = price * shares

    ev = None
    if market_cap is not None:
        ev = market_cap + (net_debt or 0)

    eps = safe_div(base["net_income"], shares)
    bvps = safe_div(base["book_equity"], shares)

    multiples = {
        "market_cap": market_cap,
        "ev": ev,
        "eps": eps,
        "bvps": bvps,
        "pe": safe_div(price, eps),
        "pb": safe_div(price, bvps),
        "ev_ebitda": safe_div(ev, base["ebitda"]),
        "ev_ebit": safe_div(ev, base["ebit"]),
        "ev_sales": safe_div(ev, base["revenue"]),
    }

    return multiples


# ---------------------------
# Méthodes de valorisation par multiples
# ---------------------------

def fv_pe(eps, pe_target):
    return eps * pe_target if (eps is not None and pe_target) else None

def fv_pb(bvps, pb_target):
    return bvps * pb_target if (bvps is not None and pb_target) else None

def fv_ev_ebitda(ebitda, net_debt, shares, mult):
    if None in (ebitda, shares, mult) or shares == 0:
        return None
    ev = ebitda * mult
    eq = ev - (net_debt or 0)
    return eq / shares

def fv_ev_sales(revenue, net_debt, shares, mult):
    if None in (revenue, shares, mult) or shares == 0:
        return None
    ev = revenue * mult
    eq = ev - (net_debt or 0)
    return eq / shares


# ============================================================
#                   PONDÉRATION AUTOMATIQUE
# ============================================================

def get_auto_weights(cap_type: str):
    """
    Pondérations professionnelles :
    - SmallCap  → pas de DCF
    - MidCap    → DCF partiel
    - LargeCap  → DCF dominant
    """
    if cap_type == "SmallCap":
        return {
            "DCF": 0.0,
            "PE": 0.25,
            "PB": 0.15,
            "EV_SALES": 0.40,
            "EV_EBITDA": 0.20
        }

    if cap_type == "MidCap":
        return {
            "DCF": 0.40,
            "PE": 0.20,
            "PB": 0.10,
            "EV_SALES": 0.10,
            "EV_EBITDA": 0.20
        }

    if cap_type == "LargeCap":
        return {
            "DCF": 0.60,
            "PE": 0.20,
            "PB": 0.05,
            "EV_SALES": 0.0,
            "EV_EBITDA": 0.15
        }

    return {   # fallback
        "DCF": 0.40,
        "PE": 0.20,
        "PB": 0.10,
        "EV_SALES": 0.10,
        "EV_EBITDA": 0.20
    }
# ============================================================
#                  PARTIE 3 — MOTEUR DCF COMPLET
# ============================================================

# ---------------------------
# Projection des FCF
# ---------------------------

def project_fcf(fcf_start, growth, years):
    """
    Projette les FCF sur un horizon donné.
    """
    if fcf_start is None:
        return []

    fcf_list = [fcf_start]
    for _ in range(1, years):
        fcf_list.append(fcf_list[-1] * (1 + growth))
    return fcf_list


# ---------------------------
# Actualisation des FCF
# ---------------------------

def discount_cash_flows(fcfs, wacc):
    """
    Actualise une série de FCF.
    """
    if not fcfs:
        return [], 0

    disc = []
    total = 0
    for i, f in enumerate(fcfs, start=1):
        fv = f / ((1 + wacc) ** i)
        disc.append(fv)
        total += fv

    return disc, total


# ---------------------------
# Valeur terminale Gordon-Shapiro
# ---------------------------

def terminal_value(last_fcf, wacc, g_terminal):
    if None in (last_fcf, wacc, g_terminal):
        return None
    if wacc <= g_terminal:
        return None  # éviter anomalies mathématiques
    return last_fcf * (1 + g_terminal) / (wacc - g_terminal)


# ---------------------------
# Modèle DCF complet
# ---------------------------

def dcf_fair_value_per_share(
    fcf_start,
    growth_fcf,
    years,
    wacc,
    g_terminal,
    net_debt,
    shares
):
    """
    Calcule la juste valeur d'une action selon un modèle DCF.
    Retour :
    - fair_value_per_share
    - EV
    - equity_value
    - tv_discounted
    - sum_discounted_fcfs
    """

    # Vérifications
    if shares is None or shares <= 0:
        return None, None, None, None, None
    if fcf_start is None:
        return None, None, None, None, None

    # Projections
    projected_fcfs = project_fcf(fcf_start, growth_fcf, years)
    discounted_fcfs, sum_disc_fcfs = discount_cash_flows(projected_fcfs, wacc)

    # TV
    tv = terminal_value(projected_fcfs[-1], wacc, g_terminal)
    if tv is None:
        tv_discounted = None
    else:
        tv_discounted = tv / ((1 + wacc) ** years)

    # EV
    if tv_discounted is None:
        ev = None
    else:
        ev = sum_disc_fcfs + tv_discounted

    # Conversion en equity
    if ev is None:
        equity_value = None
        fair_value = None
    else:
        nd = net_debt or 0
        equity_value = ev - nd
        fair_value = equity_value / shares

    return fair_value, ev, equity_value, tv_discounted, sum_disc_fcfs


# ---------------------------
# Matrice de sensibilité DCF
# ---------------------------

def build_sensitivity_matrix(
    fcf_start,
    growth_fcf,
    years,
    base_wacc,
    base_g,
    net_debt,
    shares
):
    """
    Matrice de sensibilité :
    - 3 WACC : base - 0.5 %, base, base + 0.5 %
    - 3 g terminal : base - 0.5 %, base, base + 0.5 %
    """
    if fcf_start is None:
        return pd.DataFrame()

    wacc_values = [
        base_wacc - 0.005,
        base_wacc,
        base_wacc + 0.005,
    ]

    g_values = [
        base_g - 0.005,
        base_g,
        base_g + 0.005,
    ]

    matrix = {}

    for w in wacc_values:
        row = []
        for g in g_values:
            fv, _, _, _, _ = dcf_fair_value_per_share(
                fcf_start=fcf_start,
                growth_fcf=growth_fcf,
                years=years,
                wacc=w,
                g_terminal=g,
                net_debt=net_debt,
                shares=shares,
            )
            row.append(fv)
        matrix[f"WACC {w*100:.2f}%"] = row

    df = pd.DataFrame(matrix, index=[f"g {g*100:.2f}%" for g in g_values])
    return df
# ============================================================
#                     PARTIE 4 — PIPELINE
# ============================================================

def estimate_starting_fcf(fundamentals: dict):
    """
    FCF de départ = Operating Cash Flow – Capex
    Si non disponible → None
    """
    try:
        cf = fundamentals["Financials"]["Cash_Flow"]["yearly"]
        last = sorted(cf.keys())[-1]
        row = cf[last]
    except:
        return None

    op_cf = pick_first_non_null(row, [
        "TotalCashFromOperatingActivities",
        "NetCashProvidedByOperatingActivities",
        "OperatingCashFlow"
    ])

    capex = pick_first_non_null(row, [
        "CapitalExpenditures",
        "InvestmentsInPropertyPlantAndEquipment"
    ])

    if op_cf is None or capex is None:
        return None

    return op_cf - capex


def compute_multiples_fair_values(base_multiples, base_fin, net_debt, shares, weights):
    """
    Calcule les valeurs intrinsèques via :
    - P/E
    - P/B
    - EV/EBITDA
    - EV/Sales
    Retourne un dict :
        {
            "PE": valeur_par_action_ou_None,
            ...
        }
    """
    results = {}

    # PE
    results["PE"] = fv_pe(base_multiples["eps"], 15)  # multiple par défaut

    # PB
    results["PB"] = fv_pb(base_multiples["bvps"], 1.5)

    # EV/EBITDA
    results["EV_EBITDA"] = fv_ev_ebitda(
        base_fin["ebitda"],
        net_debt,
        shares,
        mult=8
    )

    # EV/Sales
    results["EV_SALES"] = fv_ev_sales(
        base_fin["revenue"],
        net_debt,
        shares,
        mult=2
    )

    return results


def compute_global_valuation(dcf_value, multiples_values, weights):
    """
    Combine DCF + multiples avec les pondérations automatiques.
    Ignore automatiquement les valeurs None.
    """
    total_weight = 0
    weighted_sum = 0

    for method, fv in multiples_values.items():
        w = weights.get(method, 0)
        if fv is not None:
            total_weight += w
            weighted_sum += fv * w

    # DCF
    if dcf_value is not None:
        total_weight += weights.get("DCF", 0)
        weighted_sum += dcf_value * weights.get("DCF", 0)

    if total_weight == 0:
        return None

    return weighted_sum / total_weight


# ============================================================
#                  PIPELINE FINAL analyze_company()
# ============================================================

def analyze_company(
    query: str,
    api_key: str,
    years: int,
    wacc: float,
    growth_fcf: float,
    g_terminal: float
):
    """
    Pipeline complet d'analyse :
    1. Résolution ticker
    2. Extraction prix, fondamentaux
    3. Extraction états financiers
    4. Classification Small/Mid/Large
    5. Désactivation automatique du DCF pour les SmallCaps
    6. Calcul des multiples
    7. Synthèse globale
    """

    # -----------------------------
    # 1) Résolution du ticker
    # -----------------------------
    if "." in query and " " not in query:
        ticker = query.strip()
        search_results = []
    else:
        search_results = search_instrument(query.strip(), api_key)
        if not search_results:
            raise ValueError("Aucun instrument trouvé.")
        ticker = build_ticker_from_search_result(search_results[0])
        if ticker is None:
            raise ValueError("Impossible de construire le ticker.")

    # -----------------------------
    # 2) Prix marché + fondamentaux
    # -----------------------------
    price = fetch_eod_price(ticker, api_key)
    fundamentals = fetch_fundamentals(ticker, api_key)

    company = get_company_summary(fundamentals)
    shares = extract_shares_outstanding(fundamentals)
    net_debt = extract_net_debt(fundamentals)

    # historiques
    hist_df = extract_historical_financials(fundamentals, max_years=5)

    # base metrics
    base_fin = extract_financial_base(fundamentals)
    base_mult = compute_base_multiples(price, shares, net_debt, base_fin)

    # -----------------------------
    # 3) Classification
    # -----------------------------
    cap_type = classify_company(base_mult["market_cap"])

    # -----------------------------
    # 4) Estimation FCF + activation DCF
    # -----------------------------
    fcf_start = estimate_starting_fcf(fundamentals)

    dcf_allowed = (
        cap_type != "SmallCap" and
        fcf_start is not None and
        shares not in (None, 0)
    )

    # -----------------------------
    # 5) DCF SI autorisé
    # -----------------------------
    if dcf_allowed:
        fv_dcf, ev, equity_value, tv_discounted, sum_disc_fcfs = dcf_fair_value_per_share(
            fcf_start,
            growth_fcf,
            years,
            wacc,
            g_terminal,
            net_debt,
            shares
        )

        proj_fcfs = project_fcf(fcf_start, growth_fcf, years)
        disc_fcfs, _ = discount_cash_flows(proj_fcfs, wacc)

        proj_df = pd.DataFrame({
            "Année": [f"Year {i}" for i in range(1, years+1)],
            "FCF projeté": proj_fcfs,
            "FCF actualisé": disc_fcfs
        })

        sens_df = build_sensitivity_matrix(
            fcf_start,
            growth_fcf,
            years,
            base_wacc=wacc,
            base_g=g_terminal,
            net_debt=net_debt,
            shares=shares
        )
    else:
        fv_dcf = None
        proj_df = pd.DataFrame()
        sens_df = pd.DataFrame()
        ev = equity_value = tv_discounted = sum_disc_fcfs = None

    # -----------------------------
    # 6) Multiples & pondérations
    # -----------------------------
    weights = get_auto_weights(cap_type)

    multiples_vals = compute_multiples_fair_values(
        base_mult,
        base_fin,
        net_debt,
        shares,
        weights
    )

    # -----------------------------
    # 7) Synthèse globale
    # -----------------------------
    global_fv = compute_global_valuation(fv_dcf, multiples_vals, weights)

    # -----------------------------
    # 8) Retour complet
    # -----------------------------
    return {
        "ticker": ticker,
        "company": company,
        "price": price,
        "shares": shares,
        "net_debt": net_debt,
        "cap_type": cap_type,

        "hist_df": hist_df,
        "base_fin": base_fin,
        "multiples": base_mult,

        "fcf_start": fcf_start,
        "dcf": {
            "active": dcf_allowed,
            "fair_value": fv_dcf,
            "ev": ev,
            "equity_value": equity_value,
            "tv_discounted": tv_discounted,
            "sum_fcf": sum_disc_fcfs,
            "proj_df": proj_df,
            "sens_df": sens_df,
        },

        "multiples_fv": multiples_vals,
        "weights": weights,
        "global_fv": global_fv,
    }
# ============================================================
#                  PARTIE 5 — UI STREAMLIT PRO
# ============================================================

def main():

    st.set_page_config(
        page_title="Valuation Pro — DCF & Multiples",
        layout="wide",
        page_icon="📊"
    )

    # ---------------------------
    # HEADER ESTHÉTIQUE
    # ---------------------------
    st.markdown(
        """
        <div style="background-color:#0A1A2F;padding:20px;border-radius:8px;margin-bottom:20px;">
            <h1 style="color:white;text-align:center;margin:0;">
                📊 Valuation Model Pro — DCF & Multiples
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    # SIDEBAR — INPUTS
    # ---------------------------

    st.sidebar.header("⚙️ Paramètres d'analyse")

    api_key = get_api_key()

    ticker_or_name = st.sidebar.text_input(
        "Nom ou ticker de la société",
        placeholder="Ex : AAPL, Microsoft, LVMH, Dassault…"
    )

    years = st.sidebar.slider("Horizon DCF (années)", 3, 10, 5)
    wacc = st.sidebar.slider("WACC (%)", 4.0, 12.0, 8.0) / 100
    g_fcf = st.sidebar.slider("Croissance FCF (%)", -5.0, 10.0, 2.0) / 100
    g_terminal = st.sidebar.slider("g Terminal (%)", 0.0, 5.0, 1.5) / 100

    run_button = st.sidebar.button("🚀 Lancer l'analyse")

    if not run_button:
        st.info("🔍 Entrez un ticker ou un nom de société puis lancez l'analyse.")
        return

    if not api_key:
        st.error("❌ Veuillez entrer une API Key EODHD.")
        return

    # ---------------------------
    # EXECUTION ANALYSE
    # ---------------------------

    try:
        result = analyze_company(
            ticker_or_name,
            api_key=api_key,
            years=years,
            wacc=wacc,
            growth_fcf=g_fcf,
            g_terminal=g_terminal
        )
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse : {str(e)}")
        return

    company = result["company"]
    price = result["price"]
    cap_type = result["cap_type"]
    dcf_data = result["dcf"]
    multiples_base = result["multiples"]
    multiples_fv = result["multiples_fv"]
    global_fv = result["global_fv"]

    st.markdown(f"## 🏢 {company.get('Name', 'N/A')}  \n"
                f"**Ticker : {result['ticker']}** — {company.get('Sector', '')}")

    # ============================================================
    # BANDEAU DE METRICS
    # ============================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Prix de marché", f"{price:.2f}")

    with col2:
        mc = multiples_base.get("market_cap")
        st.metric("Market Cap", format_large_number(mc))

    with col3:
        nd = result["net_debt"]
        st.metric("Dette nette", format_large_number(nd))

    with col4:
        st.metric("Catégorie", cap_type)

    # ============================================================
    # ONGLET SELON TYPE DE SOCIÉTÉ
    # ============================================================

    if cap_type == "SmallCap":
        # -------------------------------------------------------
        #              UI SPÉCIALE SMALL CAP
        # -------------------------------------------------------

        st.warning(
            "⚠️ Société classée **Small Cap** — Le DCF n’est pas utilisé car les "
            "cash-flows futurs sont trop incertains."
        )

        tabs = st.tabs(["📈 Historique 5 ans", "📊 Multiples", "💰 Valorisation globale"])

        # ---------------- HISTORIQUE ----------------
        with tabs[0]:
            st.subheader("📈 Historique financier (5 ans)")
            st.dataframe(result["hist_df"], use_container_width=True)

        # ---------------- MULTIPLES ----------------
        with tabs[1]:
            st.subheader("📊 Multiples & Fair Values")
            df_mult = pd.DataFrame([
                {"Méthode": "P/E", "Fair Value": multiples_fv["PE"]},
                {"Méthode": "P/B", "Fair Value": multiples_fv["PB"]},
                {"Méthode": "EV/EBITDA", "Fair Value": multiples_fv["EV_EBITDA"]},
                {"Méthode": "EV/Sales", "Fair Value": multiples_fv["EV_SALES"]},
            ])
            df_mult["Fair Value"] = df_mult["Fair Value"].apply(lambda x: f"{x:.2f}" if x else "N/A")
            st.dataframe(df_mult, use_container_width=True)

        # ---------------- SYNTHESE ----------------
        with tabs[2]:
            st.subheader("💰 Valorisation Globale (Multiples uniquement)")
            if global_fv:
                upside = (global_fv / price - 1) * 100 if price else None
                st.metric("Juste valeur globale", f"{global_fv:.2f}")
                st.metric("Upside potentiel", f"{upside:.1f} %")
            else:
                st.error("Impossible d'établir une valorisation globale fiable.")

        return  # STOP ICI POUR LES SMALL CAPS

    # ============================================================
    # UI POUR MID + LARGE CAPS (DCF ACTIVÉ)
    # ============================================================

    tabs = st.tabs([
        "📈 Historique 5 ans",
        "💵 Résumé DCF",
        "📊 DCF détaillé",
        "📉 Sensibilité",
        "📚 Multiples",
        "💰 Valorisation globale"
    ])

    # ---------------- HISTORIQUE ----------------
    with tabs[0]:
        st.subheader("📈 Historique financier (5 ans)")
        st.dataframe(result["hist_df"], use_container_width=True)

    # ---------------- RESUME DCF ----------------
    with tabs[1]:
        st.subheader("💵 Résumé du DCF")
        if not dcf_data["active"]:
            st.error("❌ DCF désactivé (données insuffisantes)")
        else:
            fv = dcf_data["fair_value"]
            ev = dcf_data["ev"]
            eq = dcf_data["equity_value"]
            sum_fcfs = dcf_data["sum_fcf"]
            tv = dcf_data["tv_discounted"]

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("EV", format_large_number(ev))
                st.metric("TV actualisée", format_large_number(tv))
            with colB:
                st.metric("Somme FCF actualisés", format_large_number(sum_fcfs))
                st.metric("Valeur Equity", format_large_number(eq))
            with colC:
                st.metric("Juste valeur / action", f"{fv:.2f}")
                st.metric("Shares", format_large_number(result["shares"]))

    # ---------------- DCF DETAIL ----------------
    with tabs[2]:
        st.subheader("📊 Détails FCF")
        st.dataframe(dcf_data["proj_df"], use_container_width=True)

    # ---------------- SENSIBILITÉ ----------------
    with tabs[3]:
        st.subheader("📉 Matrice de sensibilité DCF")
        st.dataframe(dcf_data["sens_df"].round(2), use_container_width=True)

    # ---------------- MULTIPLES ----------------
    with tabs[4]:
        st.subheader("📚 Valorisation par multiples")

        df_mult = pd.DataFrame([
            {"Méthode": "P/E", "Fair Value": multiples_fv["PE"]},
            {"Méthode": "P/B", "Fair Value": multiples_fv["PB"]},
            {"Méthode": "EV/EBITDA", "Fair Value": multiples_fv["EV_EBITDA"]},
            {"Méthode": "EV/Sales", "Fair Value": multiples_fv["EV_SALES"]},
        ])
        df_mult["Fair Value"] = df_mult["Fair Value"].apply(lambda x: f"{x:.2f}" if x else "N/A")

        st.dataframe(df_mult, use_container_width=True)

    # ---------------- SYNTHESE GLOBALE ----------------
    with tabs[5]:
        st.subheader("💰 Synthèse Globale")

        if global_fv:
            upside = (global_fv / price - 1) * 100 if price else None
            st.metric("Juste valeur globale", f"{global_fv:.2f}")
            st.metric("Upside potentiel", f"{upside:.1f} %")
        else:
            st.error("Impossible d'établir une valorisation globale fiable.")


# Launch
if __name__ == "__main__":
    main()
