"""
Ireland Tourism Intelligence Dashboard
Author: Sanskruti Dwivedi
---------------------------------------
Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ireland Tourism Intelligence",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "processed", "ireland_tourism_real.db")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ── colour palette ───────────────────────────────────────────────────────────
IRELAND_GREEN  = "#169B62"
IRELAND_ORANGE = "#FF883E"

# ── load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    tourism = pd.read_sql("SELECT * FROM tourism", conn)
    rent    = pd.read_sql("SELECT * FROM rent",    conn)
    conn.close()
    return tourism, rent

@st.cache_data
def load_ml_results():
    results = {}
    for fname in ["forecast_results.json", "clustering_results.json", "regression_results.json"]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[fname.replace("_results.json","")] = json.load(f)
    return results

tourism, rent = load_data()
ml = load_ml_results()

# annual totals
annual = tourism.groupby("year")["visits"].sum().reset_index().rename(columns={"visits":"total_visits"})

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍀 Ireland Tourism\nIntelligence")
    st.markdown("---")
    st.markdown("""
    > *"I moved to Galway in 2025.  
    > Finding a flat nearly broke me.  
    > I wanted to know if 11 million tourists  
    > had anything to do with that."*  
    > — Sanskruti Dwivedi
    """)
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📈 Forecasting",
        "🌍 Origin Trends",
        "🏘️ Rent vs Tourism",
        "📖 Methodology"
    ])
    st.markdown("---")
    st.caption("Data: CSO Ireland · RTB/ESRI\nBuilt with Python · sklearn · KMeans · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🍀 Ireland Tourism Intelligence")
    st.markdown("""
    Ireland welcomed **9.3 million overseas visitors** in 2024 — near record levels.  
    This project uses **real public data** from the CSO and RTB to explore tourism trends,  
    forecast demand, and examine the relationship between visitor volumes and rent prices.
    """)
    st.caption("Data: CSO PxStat API (TMQ02) · RTB/ESRI Rent Index · 2009–2024")

    st.divider()

    # KPI row
    latest_year = annual["year"].max()
    latest_visits = annual[annual["year"]==latest_year]["total_visits"].values[0]
    prev_visits   = annual[annual["year"]==latest_year-1]["total_visits"].values[0]
    covid_low     = annual["total_visits"].min()
    peak_visits   = annual["total_visits"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{latest_year} Visitors", f"{latest_visits/1e6:.2f}M",
                f"{(latest_visits-prev_visits)/prev_visits*100:+.1f}% vs prior year")
    col2.metric("COVID Low (2020)", f"{covid_low/1e6:.2f}M",
                f"{(covid_low-peak_visits)/peak_visits*100:.0f}% vs peak")
    col3.metric("Peak Year", f"{annual.loc[annual['total_visits'].idxmax(),'year']}",
                f"{peak_visits/1e6:.2f}M visits")
    rent_2024 = rent[rent["year"]==2024]["avg_monthly_rent_eur"].mean()
    rent_2019 = rent[rent["year"]==2019]["avg_monthly_rent_eur"].mean()
    col4.metric("Avg National Rent 2024", f"€{rent_2024:,.0f}/mo",
                f"+€{rent_2024-rent_2019:,.0f} since 2019")

    st.divider()

    # Annual visitor trend
    st.subheader("Annual Overseas Visitors to Ireland (2009–2024)")
    fig = px.area(annual, x="year", y="total_visits",
                  color_discrete_sequence=[IRELAND_GREEN])
    fig.add_vline(x=2020, line_dash="dash", line_color="red",
                  annotation_text="COVID-19", annotation_position="top right")
    fig.update_layout(yaxis_title="Total Visits", xaxis_title="Year",
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig, use_container_width=True)

    # Origin breakdown
    st.subheader("Visitor Origins — 2024")
    origin_2024 = (tourism[tourism["year"]==2024]
                   .groupby("origin")["visits"].sum().reset_index())
    fig2 = px.pie(origin_2024, values="visits", names="origin",
                  color_discrete_sequence=[IRELAND_GREEN, IRELAND_ORANGE, "#3A86FF", "#8338EC"])
    fig2.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.title("📈 2025 Visitor Forecast")
    st.markdown("Seasonal linear regression trained on real CSO quarterly data (2009–2024), excluding COVID years as a structural break.")

    if "forecast" in ml:
        fc = ml["forecast"]

        st.success(f"🔮 **Peak quarter: {fc['peak_quarter']}** — {fc['peak_visits']:,} predicted visits")

        # historical + forecast chart
        hist_df = pd.DataFrame(fc["historical"])
        fc_df   = pd.DataFrame(fc["forecast_2025"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["quarter"], y=hist_df["visits"],
            name="Historical (CSO)", line=dict(color=IRELAND_GREEN, width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["quarter"], y=fc_df["predicted_visits"],
            name="2025 Forecast", line=dict(color=IRELAND_ORANGE, width=2, dash="dash"),
            error_y=dict(type="data",
                         array=[r-p for r,p in zip(fc_df["upper"], fc_df["predicted_visits"])],
                         arrayminus=[p-l for p,l in zip(fc_df["predicted_visits"], fc_df["lower"])],
                         visible=True)
        ))
        fig.update_layout(xaxis_title="Quarter", yaxis_title="Visits",
                          plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(tickformat=".2s")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("2025 Quarterly Forecast")
        st.dataframe(fc_df.rename(columns={
            "quarter":"Quarter","predicted_visits":"Predicted Visits",
            "lower":"Lower Bound","upper":"Upper Bound"
        }), use_container_width=True)
    else:
        st.warning("Run `python src/ml_models.py` to generate forecast results.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ORIGIN TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Origin Trends":
    st.title("🌍 Visitor Origin Shift Analysis")
    st.markdown("KMeans clustering reveals three distinct tourism eras. Great Britain's share is falling while US/Canada and Europe grow.")

    if "clustering" in ml:
        cl = ml["clustering"]

        # Origin share change
        st.subheader("Market Share Change: 2015 → 2024")
        change_df = pd.DataFrame({
            "Origin": list(cl["origin_change_pp"].keys()),
            "Change (pp)": list(cl["origin_change_pp"].values())
        }).sort_values("Change (pp)")
        colors = [IRELAND_GREEN if v >= 0 else "#D62828" for v in change_df["Change (pp)"]]
        fig = px.bar(change_df, x="Change (pp)", y="Origin", orientation="h",
                     color="Change (pp)", color_continuous_scale=["#D62828","#169B62"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Annual by origin
        st.subheader("Annual Visits by Origin (2009–2024)")
        orig_df = pd.DataFrame(cl["annual_by_origin"])
        fig2 = px.line(orig_df, x="year", y="total_visits", color="origin",
                       color_discrete_sequence=[IRELAND_GREEN, IRELAND_ORANGE, "#3A86FF", "#8338EC"])
        fig2.add_vline(x=2020, line_dash="dash", line_color="red")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        fig2.update_yaxes(tickformat=".2s")
        st.plotly_chart(fig2, use_container_width=True)

        # Era clustering
        st.subheader("Tourism Eras — KMeans Clustering")
        era_df = pd.DataFrame(cl["year_eras"])
        st.dataframe(era_df.rename(columns={
            "year":"Year","era_label":"Era","total_visits":"Total Visits"
        }), use_container_width=True)
    else:
        st.warning("Run `python src/ml_models.py` to generate clustering results.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RENT vs TOURISM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Rent vs Tourism":
    st.title("🏘️ Tourism Pressure vs Rent")
    st.markdown("Linear regression using log(visitor volume) + year trend as features to explain national average rent.")

    if "regression" in ml:
        rg = ml["regression"]

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", rg["r2_score"], "Excludes COVID years")
        col2.metric("MAE", f"€{rg['mae_eur']}/month")
        col3.metric("Visit Coefficient", f"{rg['coeff_log_visits']:.0f}")

        st.info(f"📊 {rg['finding']}")

        # National rent vs tourism scatter
        nat_df = pd.DataFrame(rg["national_data"])
        nat_df["covid"] = nat_df["year"].isin([2020,2021])
        fig = px.scatter(nat_df, x="total_visits", y="avg_monthly_rent_eur",
                         color="covid", text="year",
                         color_discrete_map={False: IRELAND_GREEN, True: "#D62828"},
                         labels={"total_visits":"Annual Visits","avg_monthly_rent_eur":"Avg Monthly Rent (€)",
                                 "covid":"COVID Year"})
        fig.update_traces(textposition="top center")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # County rent trends
        st.subheader("County Rent Trends (2015–2024)")
        county_df = pd.DataFrame(rg["county_rent_trends"])
        selected = st.multiselect("Select counties", sorted(county_df["county"].unique()),
                                  default=["Dublin","Galway","Cork","Limerick"])
        if selected:
            filtered = county_df[county_df["county"].isin(selected)]
            fig2 = px.line(filtered, x="year", y="avg_monthly_rent_eur", color="county",
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               yaxis_title="Avg Monthly Rent (€)")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Run `python src/ml_models.py` to generate regression results.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Methodology":
    st.title("📖 Methodology & Data Sources")

    st.subheader("Data Sources")
    st.markdown("""
    | Dataset | Source | License |
    |---------|--------|---------|
    | Overseas Visits (TMQ02) | CSO Ireland PxStat API | CC-BY 4.0 |
    | Average Monthly Rent by County | RTB/ESRI Rent Index | CC-BY 4.0 |
    """)

    st.subheader("Models")
    st.markdown("""
    **Model 1 — Time Series Forecasting**  
    Seasonal linear regression (sklearn) trained on quarterly CSO visitor data 2009–2024.  
    COVID years excluded as a structural break. Quarter dummies capture seasonality.

    **Model 2 — KMeans Era Clustering**  
    Groups years by visitor origin mix (Great Britain, Other Europe, USA/Canada, Other).  
    Identifies three distinct eras: Growth, Peak, COVID.

    **Model 3 — Tourism Pressure → Rent Regression**  
    Features: log(annual visits) + year trend.  
    Target: national average monthly rent.  
    R² = 0.977 (excluding COVID years).
    """)

    st.subheader("Limitations")
    st.markdown("""
    - CSO tourism data is national-level; county-level data requires Fáilte Ireland access
    - RTB rent data compiled from quarterly report appendices
    - Correlation ≠ causation; housing supply, interest rates and wages also drive rent
    - COVID years excluded from regression as a structural break — reduces sample size
    """)

    st.subheader("About")
    st.markdown("""
    **Sanskruti Dwivedi** — Data Analyst  
    MSc Business Analytics, University of Galway (2025–26)  
    📧 dwivedisanskruti10@gmail.com
    """)

# ── footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Ireland Tourism Intelligence · Sanskruti Dwivedi · Data: CSO Ireland & RTB/ESRI · 2025")
