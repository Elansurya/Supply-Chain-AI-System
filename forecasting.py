import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import mean_absolute_error, mean_squared_error

#  PATHS

BASE  = r"C:\project\Supply_Chain_AI_System"
DATA  = os.path.join(BASE, "data",    "cleaned_data.csv")
MDIR  = os.path.join(BASE, "models")
OUT   = os.path.join(BASE, "outputs")
os.makedirs(MDIR, exist_ok=True)
os.makedirs(OUT,  exist_ok=True)

sns.set_theme(style="whitegrid")
PAL = ["#2196F3","#FF7043","#4CAF50","#9C27B0","#FFC107"]

#  LOAD
df = pd.read_csv(DATA, parse_dates=["date","ship_date"])
print(f"✅ Loaded  →  {df.shape[0]:,} rows")

#  5.1  AGGREGATE TO MONTHLY LEVEL
#       Group by: category, sub_category, region, market, year, month

print("\nSTEP 5.1 — AGGREGATE TO MONTHLY LEVEL")
monthly = df.groupby(
    ["category","sub_category","region","market",
     "year","month","quarter","is_festival_month"]
).agg(
    units_sold    = ("units_sold",     "sum"),
    revenue       = ("revenue",        "sum"),
    avg_price     = ("unit_price",     "mean"),
    avg_discount  = ("discount",       "mean"),
    avg_shipping  = ("shipping_cost",  "mean"),
    avg_lead_time = ("lead_time_days", "mean"),
    avg_profit    = ("profit",         "mean"),
).reset_index()
print(f"✅ Monthly agg shape: {monthly.shape}")

#  5.2  ENCODE CATEGORICAL FEATURES

print("\nSTEP 5.2 — ENCODE CATEGORICAL FEATURES")
le = LabelEncoder()
enc_map = {}
for col in ["category","sub_category","region","market"]:
    monthly[f"{col}_enc"] = le.fit_transform(monthly[col].astype(str))
    enc_map[col] = dict(zip(monthly[col], monthly[f"{col}_enc"]))

FEATURES = [
    "category_enc","sub_category_enc","region_enc","market_enc",
    "year","month","quarter","is_festival_month",
    "avg_price","avg_discount","avg_shipping","avg_lead_time","avg_profit"
]
TARGET = "units_sold"

X = monthly[FEATURES]
y = monthly[TARGET]
print(f"✅ Features: {FEATURES}")

#  5.3  TRAIN / TEST SPLIT  (80% / 20%)

print("\nSTEP 5.3 — TRAIN / TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"✅ Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

#  5.4  MODEL TRAINING & EVALUATION

print("\nSTEP 5.4 — MODEL TRAINING & EVALUATION")
print("=" * 55)

results = {}

def train_evaluate(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\n  [{name}]")
    print(f"    MAE  = {mae:.2f}  units")
    print(f"    RMSE = {rmse:.2f}  units")
    results[name] = {"model": model, "preds": preds, "mae": mae, "rmse": rmse}
    return model, preds, mae, rmse

# ── Model A: Linear Regression (Simple) 
lr, lr_preds, lr_mae, lr_rmse = train_evaluate(
    "LINEAR REGRESSION (Simple)",
    LinearRegression()
)

# ── Model B: Random Forest (Better) 
rf, rf_preds, rf_mae, rf_rmse = train_evaluate(
    "RANDOM FOREST (Better)",
    RandomForestRegressor(n_estimators=200, max_depth=12,
                          random_state=42, n_jobs=-1)
)

# ── Model C: XGBoost (Advanced) — uses GBM fallback if no xgb ─
try:
    from xgboost import XGBRegressor
    xgb, xgb_preds, xgb_mae, xgb_rmse = train_evaluate(
        "XGBOOST (Advanced)",
        XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                     random_state=42, verbosity=0)
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    print("\n  [XGBoost not installed — using GradientBoostingRegressor instead]")
    xgb, xgb_preds, xgb_mae, xgb_rmse = train_evaluate(
        "GRADIENT BOOSTING (Advanced - XGBoost fallback)",
        GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.05, random_state=42)
    )

#  5.5  SELECT BEST MODEL

print("\nSTEP 5.5 — MODEL COMPARISON")
print("=" * 55)
comp = pd.DataFrame({
    "Model": list(results.keys()),
    "MAE"  : [r["mae"]  for r in results.values()],
    "RMSE" : [r["rmse"] for r in results.values()],
})
print(comp.sort_values("MAE").to_string(index=False))

best_name  = min(results, key=lambda k: results[k]["mae"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
print(f"\n🏆 Best Model → {best_name}  (MAE={results[best_name]['mae']:.2f})")
#  5.6  FEATURE IMPORTANCE  (from Random Forest)

fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nSTEP 5.6 — FEATURE IMPORTANCE (Random Forest)")
print(fi.to_string())

#  5.7  SAVE MODEL  →  model.pkl

model_path = os.path.join(MDIR, "model.pkl")
payload = {
    "model"      : best_model,
    "model_name" : best_name,
    "features"   : FEATURES,
    "mae"        : results[best_name]["mae"],
    "rmse"       : results[best_name]["rmse"],
    "enc_map"    : enc_map,
    "comparison" : comp,
}
with open(model_path, "wb") as f:
    pickle.dump(payload, f)
print(f"\n✅ Model saved → {model_path}")

#  5.8  FORECAST NEXT 3 MONTHS (90 days)

print("\nSTEP 5.8 — FORECAST NEXT 3 MONTHS")
print("-" * 45)

last_year  = monthly["year"].max()
last_month = monthly[monthly["year"] == last_year]["month"].max()

forecast_rows = []
combos = monthly[["category","sub_category","region","market",
                   "category_enc","sub_category_enc",
                   "region_enc","market_enc"]].drop_duplicates()

for delta in range(1, 4):
    fut_month = (last_month + delta - 1) % 12 + 1
    fut_year  = last_year + (last_month + delta - 1) // 12
    fut_qtr   = (fut_month - 1) // 3 + 1
    festival  = 1 if fut_month in [1, 8, 10, 11, 12] else 0

    for _, row in combos.iterrows():
        ref = monthly[
            (monthly["category"]    == row["category"]) &
            (monthly["sub_category"]== row["sub_category"]) &
            (monthly["region"]      == row["region"]) &
            (monthly["market"]      == row["market"])
        ].tail(1)
        if ref.empty:
            continue
        feat = {
            "category_enc"    : row["category_enc"],
            "sub_category_enc": row["sub_category_enc"],
            "region_enc"      : row["region_enc"],
            "market_enc"      : row["market_enc"],
            "year"            : fut_year,
            "month"           : fut_month,
            "quarter"         : fut_qtr,
            "is_festival_month": festival,
            "avg_price"       : ref["avg_price"].values[0],
            "avg_discount"    : ref["avg_discount"].values[0],
            "avg_shipping"    : ref["avg_shipping"].values[0],
            "avg_lead_time"   : ref["avg_lead_time"].values[0],
            "avg_profit"      : ref["avg_profit"].values[0],
        }
        X_fut  = pd.DataFrame([feat])[FEATURES]
        pred   = max(0, best_model.predict(X_fut)[0])
        forecast_rows.append({
            "forecast_year"     : fut_year,
            "forecast_month"    : fut_month,
            "category"          : row["category"],
            "sub_category"      : row["sub_category"],
            "region"            : row["region"],
            "market"            : row["market"],
            "forecasted_demand" : round(pred, 0),
        })

forecast_df = pd.DataFrame(forecast_rows)
fpath = os.path.join(BASE, "data", "forecast_data.csv")
forecast_df.to_csv(fpath, index=False)
print(f"✅ Forecast rows: {len(forecast_df)}")
print(forecast_df.groupby(["forecast_month","category"])["forecasted_demand"].sum().to_string())
print(f"✅ Saved → {fpath}")

#  5.9  CHARTS

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Demand Forecasting Results", fontsize=15, fontweight="bold")

# Chart A: Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, best_preds, alpha=0.5, color=PAL[0], s=22)
mn = min(y_test.min(), best_preds.min())
mx = max(y_test.max(), best_preds.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect fit")
ax.set_title(f"Actual vs Predicted\n{best_name}", fontweight="bold")
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.legend()

# Chart B: Feature Importance
ax = axes[1]
fi10 = fi.head(10)
ax.barh(fi10.index[::-1], fi10.values[::-1], color=PAL[2])
ax.set_title("Feature Importance\n(Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")

# Chart C: Forecast by Category × Month
ax = axes[2]
fc_grp = forecast_df.groupby(["forecast_month","category"])["forecasted_demand"].sum().unstack()
fc_grp.plot(kind="bar", ax=ax, color=PAL[:3], edgecolor="white")
ax.set_title("Forecasted Demand\nNext 3 Months by Category", fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Forecasted Units")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
chart_path = os.path.join(OUT, "forecasting_results.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Chart saved → {chart_path}")
print("\n🎯 Next → Run step6_inventory_optimization.py")