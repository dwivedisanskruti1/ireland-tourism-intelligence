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
from plotly.subplots import make_subplots

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ireland Tourism Intelligence",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "processed", "ireland_tourism.db")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ── colour palette ───────────────────────────────────────────────────────────
IRELAND_GREEN  = "#169B62"
IRELAND_ORANGE = "#FF883E"
IRELAND_WHITE  = "#FFFFFF"
SOFT_BG        = "#F0F4F0"

# ── load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    visitors      = pd.read_sql("SELECT * FROM visitors", conn)
    rent          = pd.read_sql("SELECT * FROM rent",     conn)
    accommodation = pd.read_sql("SELECT * FROM accommodation", conn)
    conn.close()
    visitors["date"] = pd.to_datetime(visitors["date"])
    return visitors, rent, accommodation

@st.cache_data
def load_ml_results():
    results = {}
    for fname in ["forecast_results.json", "clustering_results.json", "regression_results.json"]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[fname.replace("_results.json","")] = json.load(f)
    return results

visitors, rent, accommodation = load_data()
ml = load_ml_results()

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg", width=80)
    st.title("🍀 Ireland Tourism\nIntelligence")
    st.markdown("---")
    st.markdown("""
    > *"I moved to Galway in 2025.  
    > Finding a flat nearly broke me.  
    > I wanted to know if 11 million tourists  
    > had anything to do with that."*  
    > — Sanskruti Dwivedi, Data Analyst
    """)
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📈 Forecasting",
        "🗺️ Hidden Gems",
        "🏘️ Rent vs Tourism",
        "📖 Story"
    ])
    st.markdown("---")
    st.caption("Data: CSO Ireland · Fáilte Ireland · RTB  \nBuilt with Python · Prophet · KMeans · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🍀 Is Ireland Losing Its Soul to Tourism?")
    st.markdown("""
    Ireland welcomed **11+ million overseas visitors** in 2023.  
    That's great for the economy — but someone has to live here too.  
    This project uses **real data** to explore who's coming, where they go,  
    and whether the tourist boom is squeezing the people who call Ireland home.
    """)

    # KPI row
    v2023 = visitors[visitors["year"] == 2023]
    r2023 = rent[rent["year"] == 2023]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Visitors 2023",
                f"{v2023['visitors'].sum()/1e6:.1f}M",
                "+8.2% vs 2022")
    col2.metric("Total Tourist Spend",
                f"€{v2023['total_spend_eur'].sum()/1e9:.1f}B")
    col3.metric("Avg Dublin Rent 2023",
                f"€{r2023[r2023['county']=='Dublin']['avg_monthly_rent_eur'].mean():.0f}",
                "+6.8% YoY")
    col4.metric("Overtouristed Counties", "1 (Dublin)",
                "47.8 gem score")

    st.markdown("---")

    # visitor trend
    st.subheader("📅 Visitor Trend 2015–2024 (National)")
    monthly_nat = (
        visitors.groupby("date")["visitors"].sum().reset_index()
    )
    fig = px.area(monthly_nat, x="date", y="visitors",
                  color_discrete_sequence=[IRELAND_GREEN])
    fig.add_vrect(x0="2020-03-01", x1="2021-06-01",
                  fillcolor="red", opacity=0.1,
                  annotation_text="Covid-19", annotation_position="top left")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="", yaxis_title="Monthly Visitors",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # origin breakdown
    st.subheader("🌍 Where Do Visitors Come From? (2023)")
    origin_data = (
        v2023.groupby("visitor_origin")["visitors"]
        .sum().reset_index()
        .sort_values("visitors", ascending=False)
    )
    fig2 = px.pie(origin_data, values="visitors", names="visitor_origin",
                  color_discrete_sequence=px.colors.sequential.Greens_r,
                  hole=0.4)
    fig2.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.title("📈 Prophet Forecasting — What's Coming in 2025?")
    st.markdown("""
    Using Facebook's **Prophet** model trained on 10 years of monthly data,  
    we forecast visitor volumes through 2025.  
    The model accounts for **COVID disruption**, **seasonal peaks**, and **long-term growth**.
    """)

    if "forecast" in ml:
        fc = ml["forecast"]

        # National 2025 forecast
        fc_df = pd.DataFrame(fc["forecast_2025"])
        fc_df["month"] = pd.to_datetime(fc_df["month"])

        st.subheader(f"🇮🇪 National Forecast 2025 — Peak: **{fc['peak_month']}**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc_df["month"], y=fc_df["predicted_visitors"],
            mode="lines+markers", name="Forecast",
            line=dict(color=IRELAND_GREEN, width=3)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["month"], fc_df["month"][::-1]]),
            y=pd.concat([fc_df["upper_bound"], fc_df["lower_bound"][::-1]]),
            fill="toself", fillcolor="rgba(22,155,98,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval"
        ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Month", yaxis_title="Predicted Visitors"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"🔮 **Model predicts July 2025 will be the busiest month** with {fc['peak_visitors']:,} visitors nationally.")

        # County forecasts
        st.subheader("County-Level Forecasts: Dublin, Galway, Kerry")
        county_fc = fc.get("county_forecasts", {})
        cols = st.columns(3)
        for i, (county, data) in enumerate(county_fc.items()):
            df_c = pd.DataFrame(data)
            df_c["month"] = pd.to_datetime(df_c["month"])
            with cols[i]:
                fig_c = px.bar(df_c, x="month", y="predicted_visitors",
                               title=f"{county}",
                               color_discrete_sequence=[IRELAND_GREEN])
                fig_c.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False, height=300,
                    xaxis_title="", yaxis_title="Visitors"
                )
                st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.warning("Run ml_models.py first to generate forecast results.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HIDDEN GEMS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Hidden Gems":
    st.title("🗺️ KMeans Clustering — Ireland's Hidden Gem Counties")
    st.markdown("""
    We clustered all 20 counties using **tourist volume, spend, nights stayed,  
    Airbnb density, and hotel occupancy** to identify three types:

    - 🔥 **Overtouristed** — under serious pressure  
    - ⚖️ **Balanced** — manageable but growing  
    - 🌿 **Hidden Gem** — undervisited, high quality, worth discovering
    """)

    if "clustering" in ml:
        cl = ml["clustering"]
        cl_df = pd.DataFrame(cl["counties"])

        # Gem score bar chart
        st.subheader("🏅 Hidden Gem Score by County")
        fig = px.bar(
            cl_df.sort_values("gem_score"),
            x="gem_score", y="county",
            color="cluster_label",
            orientation="h",
            color_discrete_map={
                "🌿 Hidden Gem": IRELAND_GREEN,
                "⚖️  Balanced":  IRELAND_ORANGE,
                "🔥 Overtouristed": "#E63946"
            },
            labels={"gem_score": "Gem Score (higher = less pressure)", "county": ""}
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=600, legend_title="Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cards for top hidden gems
        st.subheader("🌿 Top Hidden Gem Counties")
        gems = [c for c in cl["counties"] if "Hidden Gem" in c["cluster_label"]][:6]
        cols = st.columns(3)
        for i, gem in enumerate(gems):
            with cols[i % 3]:
                st.metric(
                    label=f"🌿 {gem['county']}",
                    value=f"Gem Score: {gem['gem_score']}",
                    delta=f"€{gem['avg_spend']:.0f} avg spend"
                )

        # Scatter: visitors vs spend coloured by cluster
        st.subheader("Visitors vs Average Spend (2023)")
        fig2 = px.scatter(
            cl_df, x="total_visitors", y="avg_spend",
            color="cluster_label", size="avg_nights",
            text="county",
            color_discrete_map={
                "🌿 Hidden Gem": IRELAND_GREEN,
                "⚖️  Balanced":  IRELAND_ORANGE,
                "🔥 Overtouristed": "#E63946"
            },
            labels={"total_visitors": "Total Visitors (2023)",
                    "avg_spend": "Avg Daily Spend (€)"}
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Run ml_models.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RENT VS TOURISM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Rent vs Tourism":
    st.title("🏘️ Regression — Does Tourism Push Rent Up?")
    st.markdown("""
    The question every Galway student and Dublin renter is asking.  
    We ran a **Linear Regression** with tourist volume, Airbnb density,  
    avg visitor spend, and year trend as features.
    """)

    if "regression" in ml:
        rg = ml["regression"]

        col1, col2 = st.columns(2)
        col1.metric("R² Score", rg["r2_score"],
                    help="How much of rent variation is explained by tourism factors")
        col2.metric("Mean Absolute Error", f"€{rg['mae_eur']}/month")

        st.info(f"📌 **Key Finding:** {rg['finding']}")

        # Coefficients
        st.subheader("What Drives Rent? — Feature Coefficients")
        coeff_df = pd.DataFrame(rg["coefficients"])
        fig = px.bar(
            coeff_df.sort_values("coefficient"),
            x="coefficient", y="feature", orientation="h",
            color="coefficient",
            color_continuous_scale=["#E63946", "#ffffff", IRELAND_GREEN],
            labels={"coefficient": "Impact on Monthly Rent (€)", "feature": ""}
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # County scatter: visitors vs rent 2023
        st.subheader("County View: Visitors vs Rent (2023)")
        insight_df = pd.DataFrame(rg["county_insight_2023"])
        fig2 = px.scatter(
            insight_df, x="total_visitors", y="avg_monthly_rent_eur",
            text="county", trendline="ols",
            labels={"total_visitors": "Annual Visitors",
                    "avg_monthly_rent_eur": "Avg Monthly Rent (€)"},
            color_discrete_sequence=[IRELAND_GREEN]
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

        # Rent trend over years — top 4 counties
        st.subheader("Rent Growth Over Time — Most Pressured Counties")
        rent_trend = rent[rent["county"].isin(["Dublin","Galway","Cork","Kerry"])]
        rent_annual = (
            rent_trend.groupby(["year","county"])["avg_monthly_rent_eur"]
            .mean().reset_index()
        )
        fig3 = px.line(
            rent_annual, x="year", y="avg_monthly_rent_eur",
            color="county", markers=True,
            color_discrete_sequence=[IRELAND_GREEN, IRELAND_ORANGE, "#1D3557", "#E63946"],
            labels={"avg_monthly_rent_eur":"Avg Monthly Rent (€)", "year":"Year"}
        )
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Run ml_models.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — THE STORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Story":
    st.title("📖 The Story Behind the Data")

    st.markdown("""
    ## Why I built this

    I moved to Galway in September 2025 to study Business Analytics at University of Galway.  
    Within two weeks of arriving, three different landlords had already rented their apartments  
    to short-term holiday lets instead of students. One was listed on Airbnb for €280/night —  
    the same place I was told wasn't available for long-term rent.

    I'm a data analyst. So I did what data analysts do: I looked for the numbers.

    ---

    ## What surprised me

    **1. Dublin is genuinely in a category of its own.**  
    Every other county clusters into a manageable range. Dublin doesn't.  
    Its tourist volume is 4–5x the next highest county — and its rent is 30% above the second-highest.  
    That's not a coincidence.

    **2. The hidden gems are hiding in plain sight.**  
    Laois. Roscommon. Offaly. These counties score near-perfect on the gem index:  
    lower crowds, decent visitor spend, longer stays. If Fáilte Ireland's regional  
    strategy actually worked, these would be household names.

    **3. Airbnb density matters more than raw visitor numbers.**  
    In the regression model, the strongest predictor of high rent isn't how many  
    tourists visit — it's how many Airbnb listings exist per hotel room.  
    Short-term lets are converting housing stock into tourist infrastructure.  
    That's the mechanism.

    ---

    ## What I'd tell Fáilte Ireland

    - Invest in signage, transport, and infrastructure in Roscommon, Leitrim, and Laois  
    - Tax incentives for guesthouses in undervisited counties  
    - Enforce short-term let registration (it's been promised, it needs teeth)  
    - Publish county-level Airbnb density data publicly — transparency creates accountability

    ---

    ## Limitations & what I wish I had

    - Actual Airbnb listing data (scraped, not estimated)
    - HSE waiting list data by county to correlate healthcare pressure
    - Day-tripper vs overnight visitor distinction — the CSO data blurs this
    - Wage data by county to compute rent-to-income ratios

    *These limitations don't invalidate the findings — they just point to where  
    the next analyst should look.*

    ---

    **Built by Sanskruti Dwivedi** | MSc Business Analytics, University of Galway  
    [LinkedIn](https://www.linkedin.com/in/sanskruti-dwivedi) · [GitHub](https://github.com/sanskrutidwivedi)
    """)
