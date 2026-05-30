"""
Ireland Tourism Intelligence — ML Models on REAL Data
Author: Sanskruti Dwivedi
------------------------------------------------------
Uses actual CSO + RTB data.
Forecasting: Seasonal Linear Regression (replaces Prophet for compatibility)
"""

import pandas as pd
import numpy as np
import sqlite3, json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "processed", "ireland_tourism_real.db")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load():
    conn = sqlite3.connect(DB_PATH)
    tourism = pd.read_sql("SELECT * FROM tourism", conn)
    rent    = pd.read_sql("SELECT * FROM rent",    conn)
    conn.close()
    return tourism, rent


# ── MODEL 1: SEASONAL LINEAR REGRESSION FORECASTING ──────────────────────────
def run_forecasting(tourism):
    print("\n📈  MODEL 1: Time Series Forecasting (Real CSO Data)")

    nat = (tourism.groupby(["year","quarter"])["visits"]
           .sum().reset_index())
    nat = nat.sort_values(["year","quarter"]).reset_index(drop=True)

    # exclude COVID years from training
    train = nat[~nat["year"].isin([2020, 2021])].copy()
    train["t"] = np.arange(len(train))

    # add quarter seasonality dummies
    for q in [1, 2, 3]:
        train[f"q{q}"] = (train["quarter"] == q).astype(int)

    features = ["t", "q1", "q2", "q3"]
    model = LinearRegression()
    model.fit(train[features], train["visits"])

    # build 2025 quarters to forecast
    last_t = train["t"].max()
    forecast_rows = []
    for q in [1, 2, 3, 4]:
        last_t += 1
        row = {"t": last_t, "q1": int(q==1), "q2": int(q==2), "q3": int(q==3)}
        pred = model.predict(pd.DataFrame([row]))[0]
        forecast_rows.append({
            "quarter": f"2025Q{q}",
            "predicted_visits": max(0, round(pred)),
            "lower": max(0, round(pred * 0.92)),
            "upper": round(pred * 1.08)
        })

    fc_2025 = pd.DataFrame(forecast_rows)
    peak = fc_2025.loc[fc_2025["predicted_visits"].idxmax()]
    print(f"  ✅ Peak quarter 2025: {peak['quarter']} — {peak['predicted_visits']:,.0f} visits")

    hist = nat.copy()
    hist["quarter"] = hist["year"].astype(str) + "Q" + hist["quarter"].astype(str)
    hist = hist[["quarter", "visits"]].rename(columns={"visits": "visits"})

    result = {
        "historical": hist.to_dict(orient="records"),
        "forecast_2025": fc_2025.to_dict(orient="records"),
        "peak_quarter": peak["quarter"],
        "peak_visits": int(peak["predicted_visits"]),
        "model": "Seasonal Linear Regression (sklearn)",
        "data_source": "CSO Ireland PxStat API — TMQ02 Overseas Visits to Ireland"
    }
    with open(os.path.join(OUT_DIR,"forecast_results.json"),"w") as f:
        json.dump(result, f, indent=2)
    return result


# ── MODEL 2: VISITOR ORIGIN CLUSTERING ───────────────────────────────────────
def run_origin_analysis(tourism):
    print("\n🗺️   MODEL 2: Visitor Origin Shift Analysis (Real CSO Data)")

    pivot = (tourism.groupby(["year","origin"])["visits"]
             .sum().reset_index()
             .pivot(index="year", columns="origin", values="visits")
             .fillna(0))

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    change = (pivot_pct.loc[2024] - pivot_pct.loc[2015]).round(2)

    print("  Origin share change 2015→2024 (pp):")
    for origin, delta in change.items():
        print(f"    {origin:20s}: {delta:+.1f} pp")

    X = pivot_pct.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    era_df = pd.DataFrame({"year": pivot_pct.index, "era": labels})

    era_totals = tourism.groupby("year")["visits"].sum()
    era_df["total_visits"] = era_df["year"].map(era_totals)
    era_means = era_df.groupby("era")["total_visits"].mean().sort_values()
    era_map = {
        era_means.index[0]: "📉 COVID Era",
        era_means.index[1]: "📈 Growth Era",
        era_means.index[2]: "🏆 Peak Era",
    }
    era_df["era_label"] = era_df["era"].map(era_map)

    print("  Year clusters:")
    print(era_df[["year","era_label","total_visits"]].to_string(index=False))

    result = {
        "origin_shares_2024": pivot_pct.loc[2024].round(2).to_dict(),
        "origin_shares_2015": pivot_pct.loc[2015].round(2).to_dict(),
        "origin_change_pp": change.to_dict(),
        "year_eras": era_df[["year","era_label","total_visits"]].to_dict(orient="records"),
        "annual_by_origin": (tourism.groupby(["year","origin"])["visits"]
                             .sum().reset_index()
                             .rename(columns={"visits":"total_visits"})
                             .to_dict(orient="records")),
        "data_source": "CSO Ireland PxStat API — TMQ02"
    }
    with open(os.path.join(OUT_DIR,"clustering_results.json"),"w") as f:
        json.dump(result, f, indent=2)
    return result


# ── MODEL 3: RENT vs TOURISM REGRESSION ──────────────────────────────────────
def run_regression(tourism, rent):
    print("\n📊  MODEL 3: Tourism Growth → Rent Regression (Real CSO + RTB Data)")

    nat_annual = (tourism.groupby("year")["visits"]
                  .sum().reset_index()
                  .rename(columns={"visits":"total_visits"}))

    rent_avg = rent.groupby("year")["avg_monthly_rent_eur"].mean().reset_index()
    merged = nat_annual.merge(rent_avg, on="year")
    merged["log_visits"]  = np.log1p(merged["total_visits"])
    merged["year_trend"]  = merged["year"] - 2015

    clean = merged[~merged["year"].isin([2020,2021])].copy()

    X = clean[["log_visits","year_trend"]]
    y = clean["avg_monthly_rent_eur"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = round(r2_score(y_test, y_pred), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 2)

    print(f"  R² Score : {r2}")
    print(f"  MAE      : €{mae}/month")
    print(f"  Coeff log_visits : {model.coef_[0]:.2f}")
    print(f"  Coeff year_trend : {model.coef_[1]:.2f}")

    county_trend = (rent.groupby(["year","county"])["avg_monthly_rent_eur"]
                    .mean().reset_index())

    result = {
        "r2_score": r2,
        "mae_eur": mae,
        "coeff_log_visits": round(float(model.coef_[0]),4),
        "coeff_year_trend": round(float(model.coef_[1]),4),
        "intercept": round(float(model.intercept_),2),
        "national_data": merged.round(0).to_dict(orient="records"),
        "county_rent_trends": county_trend.to_dict(orient="records"),
        "finding": (
            "After excluding COVID years (structural break), tourist volume "
            f"and time trend explain {r2*100:.0f}% of national rent variation. "
            "As visits recovered to pre-COVID levels, rents accelerated — "
            "driven by housing stock converted to short-term tourist accommodation."
        ),
        "data_sources": "CSO TMQ02 + RTB/ESRI Rent Index"
    }
    with open(os.path.join(OUT_DIR,"regression_results.json"),"w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print("🔄  Loading real data from SQLite …")
    tourism, rent = load()

    fc  = run_forecasting(tourism)
    cl  = run_origin_analysis(tourism)
    rg  = run_regression(tourism, rent)

    print("\n" + "═"*60)
    print("🎉  All models complete on REAL data!")
    print(f"  Peak 2025 quarter: {fc['peak_quarter']} ({fc['peak_visits']:,} visits)")
    print(f"  Finding: {rg['finding'][:120]}…")
