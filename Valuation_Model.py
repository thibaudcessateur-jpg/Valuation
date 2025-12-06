import os
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
    Dette nette ≈ TotalDebt - CashAndEquivalents (dernière année annuelle disponible).
    """
    try:
        balance_sheet = fundamentals["Financials"]["Balance_Sheet"]["yearly"]
        years = sorted(balance_sheet.keys())
        if not years:
            return None
        last_year_key = years[-1]
        last_year_bs = balance_sheet[last_year_key]

        total_debt = last_year_bs.get("TotalDebt")
        cash = last_year_bs.get("CashAndCashEquivalents")

        if total_debt is None or cash is None:
            return None
        return total_debt - cash
    except Exception:
        return None


def build_historical_table(fundamentals: dict, max_years: int = 5) -> pd.DataFrame:
    """
    Construit un tableau historique multi-lignes sur les dernières années :
    CA, EBIT, Net Income, Operating CF, Capex, FCF calculé.
    On reste sur du yearly pour la lisibilité.
    """
    try:
        inc = fundamentals["Financials"]["Income_Statement"]["yearly"]
        cf = fundamentals["Financials"]["Cash_Flow"]["yearly"]
    except Exception:
        return pd.DataFrame()

    years = sorted(inc.keys(), reverse=True)  # tri décroissant
    years = years[:max_years]  # on garde les N dernières

    rows = []
    for y in years:
        inc_y = inc.get(y, {}) or {}
        cf_y = cf.get(y, {}) or {}

        revenue = inc_y.get("TotalRevenue") or inc_y.get("Revenue")
        ebit = inc_y.get("OperatingIncome") or inc_y.get("EBIT")
        net_income = inc_y.get("NetIncome")

        op_cf = cf_y.get("OperatingCashFlow")
        capex = cf_y.get("CapitalExpenditures")

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
    df = df.sort_values("Année")  # on remet en ordre croissant pour lecture
    return df

def estimate_starting_fcf(fundamentals: dict):
    """
    UFCF ≈ Free Cash Flow si dispo,
    sinon FCF = TotalCashFromOperatingActivities - CapitalExpenditures (ou équivalents).

    On gère plusieurs variantes de noms de champs possibles dans l'API EODHD.
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

    # 1) Si EODHD fournit déjà le free cash-flow, on le prend directement
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
        # Petit debug utile : voir les clés réellement dispo dans le cash-flow
        try:
            import streamlit as st
            st.write("⚠️ Clés Cash Flow disponibles pour", last_year_key, ":", list(row.keys()))
        except Exception:
            # si on est hors Streamlit (test en script), on ignore
            pass
        return None

    return float(operating_cf) - float(capex)




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
    Si impossible (wacc <= g, données manquantes), retourne (None, ...).
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

    if net_debt is None:
        net_debt_used = 0.0
    else:
        net_debt_used = net_debt

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
    - Lignes : WACC (base ± 1 % + base)
    - Colonnes : g (base ± 0.5 % + base)
    Les cellules contiennent la juste valeur par action.
    """
    # On construit des listes de WACC et g en pourcentage (décimaux)
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

    # On s'assure que g < min(WACC) pour éviter des cas invalides
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
        # colonnes nommées par g (%)
        data[f"g = {g*100:.2f} %"] = row

    index_labels = [f"WACC = {w*100:.2f} %" for w in wacc_values]
    df_matrix = pd.DataFrame(data, index=index_labels)
    return df_matrix


# =========================================
# PIPELINE PRINCIPAL POUR UNE SOCIÉTÉ
# =========================================

def analyze_company(query: str, api_key: str, years: int, wacc: float, growth_fcf: float, g_terminal: float):
    """
    Pipeline complet :
    - Recherche par nom/ticker
    - Résolution du ticker EODHD
    - Récupération fondamentaux + prix
    - Extraction des tableaux historiques
    - DCF + matrice de sensibilité
    Retourne un dict avec toutes les infos utiles.
    """
    # 1) Si l'utilisateur a directement donné un ticker complet genre 'AAPL.US', on l'utilise tel quel
    #    (on fait quand même un try/except, mais ça évite la recherche).
    ticker = None
    search_results = []

    if "." in query and " " not in query:
        ticker = query.strip()
    else:
        # Recherche par nom ou ticker
        search_results = search_instrument(query.strip(), api_key)
        if not search_results:
            raise ValueError("Aucun instrument trouvé pour cette recherche.")
        # Pour l'instant on prend le premier résultat (tu pourras ensuite ajouter un selectbox)
        ticker = build_ticker_from_search_result(search_results[0])
        if ticker is None:
            raise ValueError("Impossible de construire un ticker valide à partir du résultat de recherche.")

    # 2) Prix + fondamentaux
    price = fetch_eod_price(ticker, api_key)
    if price is None:
        raise ValueError("Impossible de récupérer le cours de marché.")

    fundamentals = fetch_fundamentals(ticker, api_key)
    company = get_company_summary(fundamentals)
    shares = get_shares_outstanding(fundamentals)
    net_debt = get_net_debt(fundamentals)
    hist_df = build_historical_table(fundamentals, max_years=5)
    fcf_start = estimate_starting_fcf(fundamentals)

    if fcf_start is None:
        raise ValueError("Impossible d'estimer un FCF de départ à partir des états financiers.")

    # 3) DCF base case
    fv, ev, equity_value, tv_discounted, sum_disc_fcfs = dcf_fair_value_per_share(
        fcf_start=fcf_start,
        growth_fcf=growth_fcf,
        years=years,
        wacc=wacc,
        g_terminal=g_terminal,
        net_debt=net_debt,
        shares=shares,
    )

    if fv is None:
        raise ValueError("Impossible de calculer une juste valeur par action avec ces paramètres (WACC/g).")

    upside = (fv / price - 1) * 100

    # 4) Projection FCF sur 5 ans (base case)
    projected_fcfs = project_fcf(fcf_start, growth_fcf, years)
    discounted_fcfs, _ = discount_cash_flows(projected_fcfs, wacc)
    proj_df = pd.DataFrame(
        {
            "Année": [f"Année {i}" for i in range(1, years + 1)],
            "FCF projeté": projected_fcfs,
            "FCF actualisé": discounted_fcfs,
        }
    )

    # 5) Matrice de sensibilité
    sens_matrix = build_sensitivity_matrix(
        fcf_start=fcf_start,
        growth_fcf=growth_fcf,
        years=years,
        base_wacc=wacc,
        base_g=g_terminal,
        net_debt=net_debt,
        shares=shares,
    )

    return {
        "ticker": ticker,
        "search_results": search_results,
        "company": company,
        "price": price,
        "shares": shares,
        "net_debt": net_debt,
        "hist_df": hist_df,
        "fcf_start": fcf_start,
        "proj_df": proj_df,
        "dcf": {
            "fair_value_per_share": fv,
            "ev": ev,
            "equity_value": equity_value,
            "tv_discounted": tv_discounted,
            "sum_disc_fcfs": sum_disc_fcfs,
            "upside_pct": upside,
        },
        "sensitivity": sens_matrix,
    }


# =========================================
# STREAMLIT APP
# =========================================

def main():
    st.set_page_config(
        page_title="DCF Valuation Pro - EODHD",
        layout="wide"
    )

    # En-tête style pro
    st.markdown(
        """
        <div style="
            background-color:#0F172A;
            padding:1.5rem 1rem;
            border-radius:1rem;
            margin-bottom:1.5rem;
        ">
            <h1 style="color:white; margin:0;">📊 Application professionnelle de valorisation DCF</h1>
            <p style="color:#E5E7EB; margin:0.3rem 0 0;">
                Analyse fondamentale d'une société via l'API EODHD : données historiques, projections sur 5 ans,
                valorisation DCF détaillée et matrice de sensibilité (WACC / g).
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
        min_value=1.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
    )
    growth_fcf_input = st.sidebar.number_input(
        "Croissance annuelle FCF (%)",
        min_value=-10.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
    )
    g_terminal_input = st.sidebar.number_input(
        "Croissance long terme g (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.25,
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

    # =========================================
    # MISE EN PAGE AVEC TABS
    # =========================================

    company = result["company"]
    dcf = result["dcf"]

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
        st.metric("Juste valeur (DCF)", f"{dcf['fair_value_per_share']:.2f}")

    upside_color = "🟢" if dcf["upside_pct"] > 0 else "🔴"
    st.markdown(
        f"**Upside / Downside estimé :** {upside_color} **{dcf['upside_pct']:.1f} %**"
    )

    # Tabs
    tab_resume, tab_hist, tab_proj, tab_dcf = st.tabs(
        ["Résumé DCF", "Historique 5 ans", "Projections FCF", "DCF & Sensibilité"]
    )

    # ----- TAB 1 : Résumé DCF -----
    with tab_resume:
        st.subheader("🎯 Résumé de la valorisation DCF (base case)")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Valeur d'entreprise (EV)", f"{dcf['ev']:,.0f}")
            st.metric("Somme FCF actualisés", f"{dcf['sum_disc_fcfs']:,.0f}")
        with col_b:
            st.metric("Valeur terminale actualisée", f"{dcf['tv_discounted']:,.0f}")
            st.metric("Valeur des capitaux propres", f"{dcf['equity_value']:,.0f}")
        with col_c:
            st.metric("Juste valeur / action", f"{dcf['fair_value_per_share']:.2f}")
            st.metric("Nombre d'actions", f"{result['shares']:,.0f}")

        st.markdown("#### Hypothèses retenues (base case)")
        st.write(f"- Horizon de projection : **{years} ans**")
        st.write(f"- WACC : **{wacc_input:.2f} %**")
        st.write(f"- Croissance FCF : **{growth_fcf_input:.2f} % par an**")
        st.write(f"- g de long terme : **{g_terminal_input:.2f} %**")
        st.write(f"- Dette nette utilisée : **{(result['net_debt'] or 0):,.0f}**")
        st.write(f"- FCF de départ estimé : **{result['fcf_start']:,.0f}**")

        st.info(
            "Ce résumé présente le scénario central (base case). "
            "La robustesse de la valorisation est analysée dans l'onglet 'DCF & Sensibilité'."
        )

    # ----- TAB 2 : Historique -----
    with tab_hist:
        st.subheader("📚 Données historiques (5 dernières années)")

        hist_df = result["hist_df"]
        if hist_df.empty:
            st.warning("Impossible de construire un historique complet à partir des données disponibles.")
        else:
            df_display = hist_df.copy()
            numeric_cols = [c for c in df_display.columns if c != "Année"]
            for c in numeric_cols:
                df_display[c] = df_display[c].astype(float).round(0)
            st.dataframe(df_display, use_container_width=True)

    # ----- TAB 3 : Projections FCF -----
    with tab_proj:
        st.subheader("📈 Projections de FCF sur 5 ans (base case)")

        proj_df = result["proj_df"].copy()
        proj_df["FCF projeté"] = proj_df["FCF projeté"].round(0)
        proj_df["FCF actualisé"] = proj_df["FCF actualisé"].round(0)
        st.dataframe(proj_df, use_container_width=True)

        st.markdown(
            "Les projections sont basées sur un FCF de départ estimé à partir du dernier **Operating Cash Flow - Capex**, "
            f"et une croissance constante de **{growth_fcf_input:.2f} %/an**."
        )

    # ----- TAB 4 : DCF & Sensibilité -----
    with tab_dcf:
        st.subheader("🧮 DCF détaillé et matrice de sensibilité")

        st.markdown("#### Matrice de sensibilité (juste valeur / action)")
        sens_df = result["sensitivity"].round(2)
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
            "En pratique, tu peux te servir de la matrice pour fixer une **marge de sécurité** : "
            "par exemple, retenir une valorisation cohérente même avec un WACC légèrement plus élevé "
            "et un g plus faible que le scénario central."
        )


if __name__ == "__main__":
    main()
