# 🍀 Ireland Tourism Intelligence
### *Is Ireland losing its soul to tourism — and does the data prove it?*

> *"I moved to Galway in 2025. Finding a flat nearly broke me.  
> I wanted to know if the 11 million tourists had anything to do with that."*

---

## The Problem

Ireland welcomed **9.3 million overseas visitors** in 2024 — still near record levels post-COVID.  
The tourism industry celebrated. Economists called it a boom.

Meanwhile, average rent in Dublin crossed **€2,400/month**.  
Galway hit **€1,678**. Students were sleeping in cars.

This project asks the uncomfortable question that the tourism dashboards don't:  
**Who is paying the price for Ireland's tourism success?**

---

## Real Data Sources

| Dataset | Source | Format | License |
|---|---|---|---|
| Overseas Visits to Ireland (TMQ02) | [CSO Ireland PxStat API](https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/TMQ02/CSV/1.0/en) | CSV via REST API | CC-BY 4.0 |
| RTB Average Monthly Rent by County | [RTB/ESRI Rent Index](https://rtb.ie/data-insights/rtb-research-reports/rtb-esri-rent-index/) | Quarterly PDF reports | CC-BY 4.0 |

**No synthetic data. No made-up numbers.** The tourism figures come directly from the CSO's live PxStat API. The rent figures are compiled from official RTB/ESRI quarterly report appendices.

---

## Key Findings (Real Data)

| Finding | Detail |
|---|---|
| 📉 **COVID wiped out 80% of visitors overnight** | 9.36M in 2019 → 1.82M in 2020. Real data confirms the full scale of the collapse |
| 📈 **2025 Q3 predicted as peak quarter** | Prophet forecasts ~3.6M visits in Q3 2025 — surpassing pre-COVID highs |
| 🌍 **Great Britain's share is falling** | Down 1.6pp since 2015; US/Canada and Other Europe growing — Ireland's tourist profile is shifting |
| 📊 **R² = 0.977** | Tourist volume + time trend explain **97.7%** of national rent variation (excluding COVID years) |

---

## Tech Stack

```
Data Ingestion      CSO PxStat REST API · requests · pandas
Storage             SQLite · SQL views
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
│   ├── raw/                         # CSVs from CSO API + RTB
│   └── processed/                   # ireland_tourism_real.db (SQLite)
│
├── src/
│   ├── data_pipeline_real.py        # Pulls from CSO API + RTB sources
│   └── ml_models_real.py            # Prophet · KMeans · Linear Regression
│
├── outputs/                         # JSON model results
│   ├── forecast_results.json
│   ├── clustering_results.json
│   └── regression_results.json
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/dwivedisanskruti1/ireland-tourism-intelligence
cd ireland-tourism-intelligence

# 2. Install
pip install -r requirements.txt

# 3. Run pipeline (fetches real CSO + RTB data)
python src/data_pipeline_real.py

# 4. Run ML models
python src/ml_models_real.py
```

---

## The 3 ML Models

### 1. 📈 Prophet Time Series Forecasting
Trained on real CSO quarterly visitor data (2009–2024), including COVID as a structural break.  
**Result:** Q3 2025 predicted as peak quarter with ~3.6M visits — beating 2019.

### 2. 🗺️ KMeans Era Clustering
Groups years by visitor origin mix to identify distinct tourism eras.  
Reveals how Ireland's tourist profile has shifted away from Great Britain toward US/Canada and Europe.  
**Result:** Three clear eras — Growth (2009–12), Peak (2013–19, 2022–24), COVID (2020–21).

### 3. 📊 Linear Regression — Tourism Pressure vs Rent
Features: log(visitor volume) + year trend  
Target: national average monthly rent  
**R² = 0.977** (excluding COVID structural break years)  
**Finding:** As visits recovered post-COVID, rents accelerated — the correlation is near-perfect.

---

## What I'd Tell Fáilte Ireland

1. The Great Britain market is declining in share — diversification is already happening
2. US/Canada visitors spend more and stay longer — target them for off-season travel
3. Q3 is dangerously over-concentrated — seasonal incentives needed urgently
4. Publish granular county-level visitor data — Fáilte Ireland has it, the public doesn't

---

## Limitations

- CSO tourism data is national-level, not county-level (county data requires Fáilte Ireland access)
- RTB rent data compiled from quarterly reports — direct API access would improve reproducibility
- Regression excludes COVID years as a structural break — this is methodologically sound but reduces sample size
- Correlation ≠ causation; other factors (housing supply, interest rates, wages) also drive rent

*These limitations don't invalidate the findings. They define the next study.*

---

## About

**Sanskruti Dwivedi** — Data Analyst  
MSc Business Analytics, University of Galway (2025–26)  
BSc Data Science, Thakur College of Science & Commerce (CGPA: 8.68)

📧 dwivedisanskruti10@gmail.com | 🔗 [LinkedIn](https://www.linkedin.com/in/sanskruti-dwivedi-01b179244/)

---
*Tourism data: CSO Ireland TMQ02 via PxStat API (CC-BY 4.0)*  
*Rent data: RTB/ESRI Rent Index quarterly reports (CC-BY 4.0)*
