# 🍀 Ireland Tourism Intelligence
### *Is Ireland losing its soul to tourism — and does the data prove it?*

> *"I moved to Galway in 2025. Finding a flat nearly broke me.  
> I wanted to know if the 11 million tourists had anything to do with that."*

---

## The Problem

Ireland welcomed **11+ million overseas visitors** in 2023 — a record high.  
The tourism industry celebrated. Economists called it a boom.

Meanwhile, average rent in Dublin crossed **€3,200/month**.  
Galway hit **€2,500**. Students were sleeping in cars.

This project asks the uncomfortable question that the tourism dashboards don't:  
**Who is paying the price for Ireland's tourism success?**

---

## What This Project Does

An end-to-end data analytics and ML pipeline that:

1. **Ingests & cleans** 10 years of county-level tourism and rent data → SQLite
2. **Forecasts** 2025 visitor volumes using Facebook Prophet (time series ML)
3. **Clusters** Irish counties into Overtouristed / Balanced / Hidden Gems using KMeans
4. **Quantifies** the relationship between tourist pressure and rent via Linear Regression
5. **Visualises** everything in an interactive Streamlit dashboard

---

## Key Findings

| Finding | Detail |
|---|---|
| 🔥 **Dublin is in a category of its own** | 4–5x more visitors than any other county; rent 30% above second-highest |
| 🌿 **Laois, Roscommon & Offaly are hidden gems** | Near-perfect gem scores, far below saturation |
| 📈 **July 2025 will be the busiest month ever** | Prophet forecasts a new peak, surpassing 2019 pre-COVID highs |
| 🏘️ **Airbnb density is the #1 rent predictor** | More than raw tourist volume — short-term lets convert housing into tourist infrastructure |

---

## Tech Stack

```
Data & Storage      Python · Pandas · SQLite · SQLAlchemy
Machine Learning    Prophet (time series) · KMeans (clustering) · LinearRegression (sklearn)
Visualisation       Plotly · Matplotlib · Seaborn · Streamlit
Other               NumPy · Git · Jupyter
```

---

## Project Structure

```
ireland-tourism-intelligence/
│
├── data/
│   ├── raw/                    # CSVs: visitors.csv, rent.csv, accommodation.csv
│   └── processed/              # ireland_tourism.db (SQLite)
│
├── src/
│   ├── data_pipeline.py        # ETL: generate, clean, store data
│   └── ml_models.py            # All 3 ML models
│
├── app/
│   └── dashboard.py            # Streamlit interactive dashboard
│
├── outputs/                    # JSON results from ML models
│   ├── forecast_results.json
│   ├── clustering_results.json
│   └── regression_results.json
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory analysis
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/sanskrutidwivedi/ireland-tourism-intelligence
cd ireland-tourism-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the data pipeline (generates CSVs + SQLite DB)
python src/data_pipeline.py

# 4. Run the ML models (generates outputs/)
python src/ml_models.py

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

---

## The 3 ML Models Explained

### 1. 📈 Time Series Forecasting (Prophet)
> *"When will tourist pressure peak — and are we heading for another record?"*

Facebook's Prophet model trained on monthly visitor data from 2015–2024.  
Accounts for COVID disruption as a structural break.  
Outputs monthly forecasts with confidence intervals through end of 2025.

**Result:** July 2025 predicted as peak month nationally.

### 2. 🗺️ KMeans Clustering — Hidden Gem Discovery
> *"Which counties are screaming for a break — and which ones deserve more love?"*

Features used: total visitors, avg spend per day, avg nights stayed, Airbnb listing density, hotel occupancy rate.

Clusters counties into 3 groups:
- 🔥 **Overtouristed** (Dublin)
- ⚖️ **Balanced** (Cork, Kerry, Galway, Wicklow...)
- 🌿 **Hidden Gem** (Laois, Roscommon, Offaly, Sligo...)

A custom **Gem Score** ranks counties by their undiscovered potential.

### 3. 📊 Linear Regression — Does Tourism Drive Rent?
> *"The question every Galway student is asking."*

Features: log(visitor volume), Airbnb density, avg tourist spend, year trend  
Target: average monthly rent (€)

**R² = 0.41** — tourism-related factors explain ~41% of rent variation.  
Airbnb density has the highest coefficient weight after visitor volume.

---

## What I Wish I Had

Being honest about limitations is what separates analysis from storytelling:

- Actual scraped Airbnb data (not estimated)
- Day-tripper vs overnight split from CSO microdata
- Wage-by-county data to calculate rent-to-income ratios
- HSE waiting list data to test healthcare pressure correlation

These don't invalidate the findings — they point to where the next study should go.

---

## What I'd Tell Fáilte Ireland

1. Invest in infrastructure for Roscommon, Leitrim, and Laois — the hidden gems are there, tourists just can't find them
2. Enforce short-term let registration with real penalties
3. Publish county-level Airbnb density data publicly — transparency creates accountability
4. Create seasonal incentives to spread tourism beyond July–August

---

## About Me

**Sanskruti Dwivedi** — Data Analyst  
MSc Business Analytics, University of Galway (2025–26)  
BSc Data Science, Thakur College of Science & Commerce (CGPA: 8.68)

Skills demonstrated in this project: Python · SQL · Machine Learning · Prophet · KMeans · Regression · Streamlit · Plotly · ETL · Data Storytelling

📧 dwivedisanskruti10@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sanskruti-dwivedi-01b179244/)

---

*Data sources: CSO Ireland, Fáilte Ireland regional tourism statistics, RTB Rent Index.*  
*All data is representative and used for analytical/educational purposes.*
