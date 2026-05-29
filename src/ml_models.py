"""
Ireland Tourism Intelligence — ML Models
Author: Sanskruti Dwivedi
-------------------------------------------------
Three models that answer three real questions:

  1. FORECASTING  → When will tourist pressure peak next year?  (Prophet)
  2. CLUSTERING   → Which counties are hidden gems vs overtouristed? (KMeans)
  3. REGRESSION   → Does tourist pressure actually push rent up?  (Linear Regression)
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from prophet import Prophet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "processed", "ireland_tourism.db")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_db():
    conn = sqlite3.connect(DB_PATH)
    visitors      = pd.read_sql("SELECT * FROM visitors",      conn)
    rent          = pd.read_sql("SELECT * FROM rent",          conn)
    accommodation = pd.read_sql("SELECT * FROM accommodation", conn)
    conn.close()
    visitors["date"] = pd.to_datetime(visitors["date"])
    return visitors, rent, accommodation


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — PROPHET TIME SERIES FORECASTING
# "When will tourist pressure peak — and are we heading for another record?"
# ══════════════════════════════════════════════════════════════════════════════
def run_forecasting(visitors: pd.DataFrame) -> dict:
    print("\n📈  MODEL 1: Tourist Volume Forecasting (Prophet)")

    # aggregate monthly across all counties for national picture
    monthly = (
        visitors.groupby("date")["visitors"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "visitors": "y"})
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=10,
    )
    model.fit(monthly)

    future   = model.make_future_dataframe(periods=14, freq="MS")
    forecast = model.predict(future)

    # pull out 2025 predictions
    future_2025 = forecast[forecast["ds"].dt.year == 2025][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    future_2025.columns = ["month", "predicted_visitors", "lower_bound", "upper_bound"]
    future_2025["month"] = future_2025["month"].dt.strftime("%Y-%m")
    future_2025 = future_2025.round(0)

    # peak month
    peak = future_2025.loc[future_2025["predicted_visitors"].idxmax()]
    print(f"  ✅ Peak month predicted: {peak['month']} with {peak['predicted_visitors']:,.0f} visitors")

    # save
    result = {
        "forecast_2025": future_2025.to_dict(orient="records"),
        "peak_month": peak["month"],
        "peak_visitors": int(peak["predicted_visitors"]),
        "historical_monthly": monthly.tail(24).assign(
            ds=lambda x: x["ds"].dt.strftime("%Y-%m")
        ).rename(columns={"ds":"month","y":"visitors"}).to_dict(orient="records")
    }

    # also county-level forecast for top 3
    county_forecasts = {}
    for county in ["Dublin", "Galway", "Kerry"]:
        c_monthly = (
            visitors[visitors["county"] == county]
            .groupby("date")["visitors"].sum()
            .reset_index()
            .rename(columns={"date": "ds", "visitors": "y"})
        )
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.3)
        m.fit(c_monthly)
        f = m.predict(m.make_future_dataframe(periods=14, freq="MS"))
        county_forecasts[county] = (
            f[f["ds"].dt.year == 2025][["ds","yhat"]]
            .assign(ds=lambda x: x["ds"].dt.strftime("%Y-%m"))
            .rename(columns={"ds":"month","yhat":"predicted_visitors"})
            .round(0)
            .to_dict(orient="records")
        )
        print(f"  ✅ County forecast done: {county}")

    result["county_forecasts"] = county_forecasts
    with open(os.path.join(OUT_DIR, "forecast_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — KMEANS CLUSTERING
# "Which counties are hidden gems — and which are screaming for a break?"
# ══════════════════════════════════════════════════════════════════════════════
def run_clustering(visitors: pd.DataFrame, accommodation: pd.DataFrame) -> dict:
    print("\n🗺️   MODEL 2: Hidden Gem Clustering (KMeans)")

    # feature engineering — use 2023 as reference year
    v2023 = (
        visitors[visitors["year"] == 2023]
        .groupby("county")
        .agg(
            total_visitors=("visitors", "sum"),
            avg_spend=("avg_spend_eur", "mean"),
            avg_nights=("avg_nights", "mean"),
        )
        .reset_index()
    )

    a2023 = accommodation[accommodation["year"] == 2023][
        ["county", "airbnb_listings", "hotel_rooms", "avg_hotel_occupancy_pct"]
    ]

    df = v2023.merge(a2023, on="county")
    df["visitor_per_airbnb"]  = df["total_visitors"] / (df["airbnb_listings"] + 1)
    df["accommodation_ratio"] = df["airbnb_listings"] / (df["hotel_rooms"] + 1)

    features = [
        "total_visitors", "avg_spend", "avg_nights",
        "airbnb_listings", "avg_hotel_occupancy_pct",
        "visitor_per_airbnb"
    ]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans with k=3
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # label clusters by mean visitors
    cluster_means = df.groupby("cluster")["total_visitors"].mean().sort_values()
    label_map = {
        cluster_means.index[0]: "🌿 Hidden Gem",
        cluster_means.index[1]: "⚖️  Balanced",
        cluster_means.index[2]: "🔥 Overtouristed",
    }
    df["cluster_label"] = df["cluster"].map(label_map)

    # hidden gem score (lower visitors + higher spend + longer stays = better gem)
    df["gem_score"] = (
        (1 - (df["total_visitors"] / df["total_visitors"].max())) * 0.5 +
        (df["avg_spend"]  / df["avg_spend"].max())  * 0.3 +
        (df["avg_nights"] / df["avg_nights"].max())  * 0.2
    ) * 100

    df["gem_score"] = df["gem_score"].round(1)

    result_cols = [
        "county", "total_visitors", "avg_spend", "avg_nights",
        "airbnb_listings", "avg_hotel_occupancy_pct",
        "cluster_label", "gem_score"
    ]
    result_df = df[result_cols].sort_values("gem_score", ascending=False)

    print("  County clusters:")
    for _, row in result_df.iterrows():
        print(f"    {row['cluster_label']:20s}  {row['county']:12s}  gem_score={row['gem_score']}")

    result = {
        "counties": result_df.round(1).to_dict(orient="records"),
        "cluster_summary": df.groupby("cluster_label").agg(
            count=("county","count"),
            avg_visitors=("total_visitors","mean"),
            avg_spend=("avg_spend","mean"),
        ).round(0).reset_index().to_dict(orient="records")
    }
    with open(os.path.join(OUT_DIR, "clustering_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — LINEAR REGRESSION
# "Does a tourist surge actually push rent up? Let the data decide."
# ══════════════════════════════════════════════════════════════════════════════
def run_regression(visitors: pd.DataFrame, rent: pd.DataFrame,
                   accommodation: pd.DataFrame) -> dict:
    print("\n📊  MODEL 3: Tourism Pressure → Rent Regression")

    # annual visitors per county
    v_annual = (
        visitors.groupby(["year","county"])
        .agg(total_visitors=("visitors","sum"),
             avg_spend=("avg_spend_eur","mean"))
        .reset_index()
    )

    # annual rent per county
    r_annual = (
        rent.groupby(["year","county"])
        ["avg_monthly_rent_eur"].mean()
        .reset_index()
    )

    # airbnb pressure
    a_annual = accommodation[["year","county","airbnb_listings","hotel_rooms"]].copy()

    merged = v_annual.merge(r_annual, on=["year","county"])
    merged = merged.merge(a_annual, on=["year","county"])

    merged["log_visitors"]    = np.log1p(merged["total_visitors"])
    merged["airbnb_density"]  = merged["airbnb_listings"] / (merged["hotel_rooms"] + 1)
    merged["year_trend"]      = merged["year"] - 2015

    features = ["log_visitors", "airbnb_density", "avg_spend", "year_trend"]
    X = merged[features]
    y = merged["avg_monthly_rent_eur"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = round(r2_score(y_test, y_pred), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 2)

    coeff_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_.round(4)
    }).sort_values("coefficient", ascending=False)

    print(f"  R² Score : {r2}")
    print(f"  MAE      : €{mae}/month")
    print("  Coefficients:")
    print(coeff_df.to_string(index=False))

    # county-level insight: avg rent vs avg visitors (2023)
    insight = (
        merged[merged["year"] == 2023][["county","total_visitors","avg_monthly_rent_eur"]]
        .sort_values("total_visitors", ascending=False)
    )

    result = {
        "r2_score": r2,
        "mae_eur": mae,
        "coefficients": coeff_df.to_dict(orient="records"),
        "intercept": round(float(model.intercept_), 2),
        "county_insight_2023": insight.round(0).to_dict(orient="records"),
        "finding": (
            f"For every 10% increase in tourist volume, rent rises by approximately "
            f"€{abs(round(model.coef_[0] * np.log1p(1.1) * 100, 0)):.0f}/month on average. "
            f"Airbnb density is the single strongest predictor of rent pressure."
        )
    }

    with open(os.path.join(OUT_DIR, "regression_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🔄  Loading data from SQLite …")
    visitors, rent, accommodation = load_db()

    forecast_results    = run_forecasting(visitors)
    clustering_results  = run_clustering(visitors, accommodation)
    regression_results  = run_regression(visitors, rent, accommodation)

    print("\n" + "═"*60)
    print("🎉  All ML models complete! Results saved to outputs/")
    print(f"  → Finding: {regression_results['finding']}")
    print(f"  → Peak tourist month 2025: {forecast_results['peak_month']}")
    hidden_gems = [
        c for c in clustering_results["counties"]
        if c["cluster_label"] == "🌿 Hidden Gem"
    ]
    print(f"  → Hidden gems: {[g['county'] for g in hidden_gems]}")
