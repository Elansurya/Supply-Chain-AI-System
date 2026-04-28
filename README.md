# SupplyChain AI System — Demand Forecasting & Inventory Optimization

> Predicts 3-month demand across 6,284 inventory segments and flags stockout risk in real-time using XGBoost and Random Forest on 51,290 Superstore transactions — helping supply chain teams optimize reorder points, safety stock, and inventory allocation with data-driven precision.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square)

🔗 **[Live Demo → Hugging Face Spaces](https://huggingface.co/spaces/Elansurya/supply-chain-ai-system)**

---

## Problem Statement

Supply chain teams in retail and e-commerce routinely face two costly extremes — stockouts that lose sales and overstock that ties up working capital. Static reorder-point models based on fixed thresholds fail to account for seasonal demand spikes, regional variation, or category-level volatility.

This project builds an end-to-end ML pipeline that forecasts demand 3 months ahead across every product–region–market segment and computes dynamic safety stock and reorder points — enabling supply chain planners to pre-empt stockouts and eliminate overstock before it happens.

---

## Dataset

| Property | Detail |
|---|---|
| Total records | 51,290 order transactions |
| Time range | 2011 – 2015 |
| Markets | 7 (US, EU, APAC, LATAM, Africa, MEA, Canada) |
| Features | 35 variables — category, sub-category, region, market, quantity, price, discount, lead time, etc. |
| Target variable | Units sold (demand forecasting — regression) |
| Segments | 6,284 unique category × sub-category × region × market combinations |
| Imbalance handling | N/A (regression task) · SMOTE not required |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| ML Models | Random Forest, XGBoost / GradientBoosting |
| Hyperparameter tuning | Manual grid search (n_estimators, max_depth, learning_rate) |
| Feature engineering | Time features, festival flags, price signals, encoded categoricals |
| Evaluation | MAE, RMSE |
| Deployment | Hugging Face Spaces (Streamlit) |
| Visualization | Plotly (interactive charts), Seaborn, Matplotlib |

---

## Workflow

```
Raw 51,290 Superstore Transactions
        ↓
Step 3 — Data Cleaning & Preprocessing
  ├── Missing value imputation (median numeric, mode categorical)
  ├── Outlier treatment via IQR winsorization
  ├── Label encoding for category, sub-category, region, market
  ├── Festival month flag (Jan, Aug, Oct, Nov, Dec → is_festival_month = 1)
  └── Standard scaling for numerical features
        ↓
Step 4 — Exploratory Data Analysis
  ├── Revenue by category: Technology $4.7M · Furniture $4.0M · Office Supplies $3.9M
  ├── Profit margin: Technology 17.2% · Office Supplies 13.7% · Furniture 3.4%
  ├── Festival month lift: +18.4% Technology · +14.1% Office Supplies · +9.7% Furniture
  └── Loss-making sub-categories: Tables (−$17,725) · Bookcases (−$3,473)
        ↓
Step 5 — Demand Forecasting
  ├── Monthly aggregation by segment (8 aggregated columns)
  ├── 80/20 train-test split (41,032 train / 10,258 test rows)
  ├── 3 models trained: LinearRegression · RandomForest(200) · XGBoost/GBM(200, lr=0.05)
  ├── Best model: XGBoost → MAE 3.41 · RMSE 5.89 · saved to models/model.pkl
  └── Forecast generated: 6,284 segments × 3 months = 18,852 forecast rows
        ↓
Step 6 — Inventory Optimization
  ├── Average Daily Demand (ADD) = Total Units ÷ Total Days
  ├── Safety Stock = 1.65 × StdDev × √Lead Time  (95% service level)
  ├── Reorder Point (ROP) = (ADD × Lead Time) + Safety Stock
  ├── Risk flag: HIGH (below SS) · MEDIUM (below ROP) · LOW (adequate)
  └── 847 HIGH-risk segments flagged for immediate reorder
        ↓
Step 7 — Export Final Dataset
  ├── Merge: cleaned_data.csv + inventory_stats.csv + forecast_data.csv
  ├── Join key: category × sub_category × region × market
  ├── 28 output columns — demand, inventory, forecast, alert, profitability fields
  └── final_supply_chain_data.csv → Power BI ready
        ↓
Step 8 — Deployment
  └── Hugging Face Spaces — interactive 5-tab Streamlit dashboard
```

---

## Results

| Model | MAE | RMSE | Notes |
|---|---|---|---|
| Linear Regression | 12.84 | — | Baseline — underfits seasonal patterns |
| Random Forest (200 est.) | 6.17 | — | Strong baseline, misses sharp spikes |
| **XGBoost / GBM (200 est.)** | **🏆 3.41** | **🏆 5.89** | Best model — saved to model.pkl |

**Inventory Risk Output (6,284 segments):**

| Risk Flag | Segments | Share | Action |
|---|---|---|---|
| 🔴 HIGH — Below Safety Stock | 847 | 13.5% | Immediate reorder required |
| 🟡 MEDIUM — Below Reorder Point | 2,138 | 34.0% | Order within lead time |
| 🟢 LOW — Stock Adequate | 3,299 | 52.5% | Maintain current policy |

**Demand Trend (51,290 rows post-merge):**

| Trend | Rows | Share |
|---|---|---|
| Increasing | 28,412 | 55.4% |
| Decreasing | 16,819 | 32.8% |
| Stable | 6,059 | 11.8% |

---

## Feature Importance (Top 7 — Random Forest)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | avg_price | 0.312 |
| 2 | sub_category_enc | 0.241 |
| 3 | month | 0.183 |
| 4 | is_festival_month | 0.097 |
| 5 | category_enc | 0.071 |
| 6 | avg_lead_time | 0.056 |
| 7 | region_enc | 0.040 |

---

## Inventory Formulas

All three core calculations used in `step6_inventory_optimization.py`:

```
Average Daily Demand (ADD)
  ADD = Total Units Sold ÷ Total Days

Safety Stock (95% service level, Z = 1.65)
  SS = 1.65 × StdDev(demand) × √(Lead Time)

Reorder Point (ROP)
  ROP = (ADD × Lead Time) + Safety Stock
```

---

## Live Demo

🔗 **[Try Supply Chain AI on Hugging Face Spaces](https://huggingface.co/spaces/Elansurya/supply-chain-ai-system)**

The dashboard has 5 tabs:

| Tab | What it shows |
|---|---|
| **Overview** | Pipeline progress · Model comparison · Risk distribution · Active alerts |
| **EDA Insights** | Revenue share · Profit margins · Festival lift · Top sub-categories |
| **Forecasting** | Feature importance · 3-month demand forecast · Training pipeline steps |
| **Inventory** | Formula cards · Segment table with risk filter · Regional heatmap |
| **Export** | 28-column schema · Merge logic · Demand trend summary |

> **Screenshots:**
> Add these to a `/screenshots` folder:
> 1. `overview_tab.png` — Pipeline progress and KPI cards
> 2. `eda_tab.png` — Revenue and profit charts
> 3. `forecast_tab.png` — 3-month demand bar chart
> 4. `inventory_tab.png` — Segment risk table and heatmap

![Overview Tab](screenshots/overview_tab.png)
![Inventory Tab](screenshots/inventory_tab.png)
![Forecast Tab](screenshots/forecast_tab.png)

---

## Business Impact

- **Stockout prevention:** 847 HIGH-risk segments flagged before inventory runs out — directly reduces lost sales and emergency procurement costs
- **Dynamic safety stock:** Z=1.65 service level buffers replace static thresholds — adapts to each segment's actual demand variability
- **Festival readiness:** is_festival_month feature captures +14% avg seasonal lift — inventory buffers recommended 4–6 weeks ahead
- **Loss sub-category visibility:** Tables and Bookcases flagged as margin-negative — data to support discount policy review
- **Power BI ready:** final_supply_chain_data.csv exports 28 columns with alert flags — plug directly into BI dashboards for operations teams

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Elansurya/supply-chain-ai-system.git
cd supply-chain-ai-system

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run src/app.py
```

---

## Project Structure

```
supply-chain-ai-system/
├── src/
│   └── app.py                          # Streamlit dashboard (5 tabs)
├── notebooks/
│   ├── step3_preprocessing.py          # Cleaning, encoding, feature engineering
│   ├── step4_eda.py                    # EDA — charts and business insights
│   ├── step5_forecasting.py            # RF + XGBoost training + forecast generation
│   ├── step6_inventory_optimization.py # ADD, Safety Stock, ROP, risk flags
│   └── step7_export_final.py           # Merge + 28-column final export
├── data/
│   ├── cleaned_data.csv                # 51,290 rows · 35 columns
│   ├── inventory_stats.csv             # 6,284 segment rows
│   └── forecast_data.csv              # 18,852 forecast rows
├── models/
│   └── model.pkl                       # XGBoost best model + label encoders
├── output/
│   └── final_supply_chain_data.csv     # 28-column Power BI ready export
├── screenshots/
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit==1.38.0
plotly==5.24.1
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
xgboost==2.1.1
lightgbm==4.5.0
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
```

---

## Author

**Elansurya K** — Aspiring Data Scientist | ML · Python · SQL · Supply Chain Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/elansurya-karthikeyan-3b6636380)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/Elansurya)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)](https://huggingface.co/spaces/Elansurya/supply-chain-ai-system)
