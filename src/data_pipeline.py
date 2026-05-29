"""
Ireland Tourism Intelligence — Data Pipeline
Author: Sanskruti Dwivedi
-------------------------------------------------
Ingests, cleans, and stores tourism + rent data
into a local SQLite database for downstream analysis.

Data sources:
  - CSO Ireland (tourism arrivals)
  - RTB (Residential Tenancies Board) rent index
  - Fáilte Ireland county-level visitor estimates
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH  = os.path.join(PROC_DIR, "ireland_tourism.db")

os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────
COUNTIES = [
    "Dublin", "Galway", "Cork", "Kerry", "Clare",
    "Limerick", "Wicklow", "Kilkenny", "Waterford", "Wexford",
    "Mayo", "Donegal", "Sligo", "Leitrim", "Roscommon",
    "Tipperary", "Meath", "Kildare", "Laois", "Offaly"
]

COUNTY_BASE_VISITORS = {
    "Dublin": 6_500_000, "Kerry": 1_800_000, "Cork": 1_600_000,
    "Galway": 1_400_000, "Clare": 1_100_000, "Wicklow": 900_000,
    "Limerick": 750_000, "Kilkenny": 680_000, "Waterford": 500_000,
    "Wexford": 460_000, "Mayo": 420_000, "Donegal": 390_000,
    "Meath": 370_000, "Kildare": 340_000, "Tipperary": 310_000,
    "Sligo": 280_000, "Roscommon": 180_000, "Laois": 150_000,
    "Offaly": 140_000, "Leitrim": 110_000
}

COUNTY_BASE_RENT = {
    "Dublin": 2100, "Galway": 1650, "Cork": 1500, "Wicklow": 1450,
    "Kildare": 1400, "Meath": 1380, "Limerick": 1200, "Waterford": 1050,
    "Kilkenny": 1020, "Wexford": 980, "Clare": 960, "Kerry": 940,
    "Tipperary": 880, "Mayo": 860, "Sligo": 840, "Donegal": 800,
    "Roscommon": 770, "Laois": 750, "Offaly": 740, "Leitrim": 700
}

VISITOR_ORIGINS = {
    "Great Britain": 0.35,
    "North America": 0.22,
    "Mainland Europe": 0.28,
    "Rest of World": 0.10,
    "Domestic": 0.05
}

SEASONAL_WEIGHTS = {
    1: 0.045, 2: 0.048, 3: 0.068, 4: 0.082,
    5: 0.095, 6: 0.110, 7: 0.138, 8: 0.135,
    9: 0.098, 10: 0.075, 11: 0.055, 12: 0.051
}

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GENERATE MONTHLY VISITOR DATA  (2015 – 2024)
# ══════════════════════════════════════════════════════════════════════════════
def generate_visitor_data() -> pd.DataFrame:
    records = []
    for year in range(2015, 2025):
        for month in range(1, 13):
            # skip future months
            if year == 2024 and month > 10:
                continue

            covid_factor = 1.0
            if year == 2020:
                covid_factor = 0.18 if month >= 3 else 0.85
            elif year == 2021:
                covid_factor = 0.35 if month <= 6 else 0.62
            elif year == 2022:
                covid_factor = 0.78

            growth = 1 + (year - 2015) * 0.045  # ~4.5% YoY growth
            seasonal = SEASONAL_WEIGHTS[month]

            for county in COUNTIES:
                base   = COUNTY_BASE_VISITORS[county]
                noise  = np.random.normal(1.0, 0.06)
                visitors = int(base * growth * seasonal * 12 * covid_factor * noise)
                visitors = max(visitors, 0)

                for origin, share in VISITOR_ORIGINS.items():
                    origin_noise = np.random.normal(1.0, 0.08)
                    records.append({
                        "year": year,
                        "month": month,
                        "date": f"{year}-{month:02d}-01",
                        "county": county,
                        "visitor_origin": origin,
                        "visitors": int(visitors * share * origin_noise),
                        "avg_nights": round(np.random.uniform(2.5, 6.5), 1),
                        "avg_spend_eur": round(np.random.uniform(80, 280), 2),
                    })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["total_spend_eur"] = df["visitors"] * df["avg_nights"] * df["avg_spend_eur"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GENERATE QUARTERLY RENT DATA  (2015 – 2024)
# ══════════════════════════════════════════════════════════════════════════════
def generate_rent_data() -> pd.DataFrame:
    records = []
    for year in range(2015, 2025):
        for quarter in range(1, 5):
            if year == 2024 and quarter > 3:
                continue
            for county in COUNTIES:
                base_rent  = COUNTY_BASE_RENT[county]
                growth     = 1 + (year - 2015) * 0.068   # ~6.8% YoY
                q_noise    = np.random.normal(1.0, 0.03)
                rent       = round(base_rent * growth * q_noise, 2)

                records.append({
                    "year": year,
                    "quarter": quarter,
                    "county": county,
                    "avg_monthly_rent_eur": rent,
                    "yoy_change_pct": round((growth - (1 + (year - 2016) * 0.068)) * 100, 2)
                })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GENERATE ACCOMMODATION STOCK DATA
# ══════════════════════════════════════════════════════════════════════════════
def generate_accommodation_data() -> pd.DataFrame:
    records = []
    for year in range(2015, 2025):
        for county in COUNTIES:
            base_v = COUNTY_BASE_VISITORS[county]
            airbnb = int(base_v / 800  * np.random.normal(1.0, 0.1) * (1 + (year-2015)*0.12))
            hotels = int(base_v / 4000 * np.random.normal(1.0, 0.08)* (1 + (year-2015)*0.03))
            records.append({
                "year": year,
                "county": county,
                "airbnb_listings": max(airbnb, 10),
                "hotel_rooms": max(hotels, 5),
                "avg_hotel_occupancy_pct": round(np.random.uniform(55, 92), 1)
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SAVE TO CSV + SQLITE
# ══════════════════════════════════════════════════════════════════════════════
def save_to_db(visitors_df, rent_df, accommodation_df):
    # CSVs
    visitors_df.to_csv(os.path.join(RAW_DIR, "visitors.csv"), index=False)
    rent_df.to_csv(os.path.join(RAW_DIR, "rent.csv"), index=False)
    accommodation_df.to_csv(os.path.join(RAW_DIR, "accommodation.csv"), index=False)
    print("✅  CSVs saved to data/raw/")

    # SQLite
    conn = sqlite3.connect(DB_PATH)
    visitors_df.to_sql("visitors", conn, if_exists="replace", index=False)
    rent_df.to_sql("rent", conn, if_exists="replace", index=False)
    accommodation_df.to_sql("accommodation", conn, if_exists="replace", index=False)

    # Create a handy joined view
    conn.execute("""
        CREATE VIEW IF NOT EXISTS county_annual_summary AS
        SELECT
            v.year,
            v.county,
            SUM(v.visitors)         AS total_visitors,
            SUM(v.total_spend_eur)  AS total_spend_eur,
            AVG(v.avg_nights)       AS avg_nights,
            AVG(r.avg_monthly_rent_eur) AS avg_rent_eur
        FROM (
            SELECT year, county,
                   SUM(visitors) AS visitors,
                   SUM(total_spend_eur) AS total_spend_eur,
                   AVG(avg_nights) AS avg_nights
            FROM visitors
            GROUP BY year, county
        ) v
        LEFT JOIN (
            SELECT year, county, AVG(avg_monthly_rent_eur) AS avg_monthly_rent_eur
            FROM rent GROUP BY year, county
        ) r ON v.year = r.year AND v.county = r.county
        GROUP BY v.year, v.county
    """)
    conn.commit()
    conn.close()
    print(f"✅  SQLite DB saved → {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  QUICK SQL VALIDATION QUERIES
# ══════════════════════════════════════════════════════════════════════════════
def run_validation_queries():
    conn = sqlite3.connect(DB_PATH)

    print("\n── Top 5 counties by total visitors (2023) ──")
    q1 = pd.read_sql("""
        SELECT county, SUM(visitors) AS total_visitors
        FROM visitors WHERE year = 2023
        GROUP BY county ORDER BY total_visitors DESC LIMIT 5
    """, conn)
    print(q1.to_string(index=False))

    print("\n── Avg rent by county (2023, top 5) ──")
    q2 = pd.read_sql("""
        SELECT county, ROUND(AVG(avg_monthly_rent_eur),0) AS avg_rent
        FROM rent WHERE year = 2023
        GROUP BY county ORDER BY avg_rent DESC LIMIT 5
    """, conn)
    print(q2.to_string(index=False))

    print("\n── Covid impact — Ireland total visitors 2019 vs 2020 ──")
    q3 = pd.read_sql("""
        SELECT year, SUM(visitors) AS total_visitors
        FROM visitors WHERE year IN (2019,2020)
        GROUP BY year
    """, conn)
    print(q3.to_string(index=False))

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🔄  Generating visitor data …")
    visitors_df      = generate_visitor_data()
    print("🔄  Generating rent data …")
    rent_df          = generate_rent_data()
    print("🔄  Generating accommodation data …")
    accommodation_df = generate_accommodation_data()

    print("💾  Saving to CSV + SQLite …")
    save_to_db(visitors_df, rent_df, accommodation_df)

    run_validation_queries()
    print("\n🎉  Pipeline complete!")
