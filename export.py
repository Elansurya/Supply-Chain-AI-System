import pandas as pd
import numpy as np
import os

#  PATHS

BASE     = r"C:\project\Supply_Chain_AI_System"
CLEANED  = os.path.join(BASE, "data", "cleaned_data.csv")
FORECAST = os.path.join(BASE, "data", "forecast_data.csv")
INV      = os.path.join(BASE, "data", "inventory_stats.csv")
FINAL    = os.path.join(BASE, "data", "final_supply_chain_data.csv")

#  LOAD ALL FILES

print("=" * 60)
print("  STEP 7 — BUILD FINAL DATASET")
print("=" * 60)

df       = pd.read_csv(CLEANED,  parse_dates=["date"])
forecast = pd.read_csv(FORECAST)
inv      = pd.read_csv(INV)
print(f"✅ Cleaned data   : {df.shape}")
print(f"✅ Forecast data  : {forecast.shape}")
print(f"✅ Inventory stats: {inv.shape}")

#  7.1  BASE DATASET  →  Monthly aggregate (historical)

print("\nSTEP 7.1 — CREATE MONTHLY AGGREGATE (Historical)")
base = df.groupby(
    ["category","sub_category","region","market","year","month"]
).agg(
    units_sold      = ("units_sold",      "sum"),
    revenue         = ("revenue",         "sum"),
    profit          = ("profit",          "sum"),
    avg_unit_price  = ("unit_price",      "mean"),
    avg_discount    = ("discount",        "mean"),
    avg_lead_time   = ("lead_time_days",  "mean"),
    inventory_level = ("inventory_level", "sum"),
    inventory_value = ("inventory_value", "sum"),
    n_orders        = ("order_id",        "count"),
).reset_index()
print(f"✅ Monthly base: {base.shape}")

#  7.2  MERGE INVENTORY STATS  (Safety Stock, ROP, Risk Flag)

print("\nSTEP 7.2 — MERGE INVENTORY STATS")
inv_cols = ["category","sub_category","region","market",
            "avg_daily_demand","safety_stock","reorder_point",
            "risk_flag","days_of_stock","stock_cover_flag",
            "total_profit","profit_margin_pct"]
base = base.merge(inv[inv_cols], on=["category","sub_category","region","market"], how="left")
print(f"✅ After merge inventory: {base.shape}")

#  7.3  MERGE FORECAST DATA  (Forecasted Demand next 3 months)

print("\nSTEP 7.3 — MERGE FORECAST DATA")

# For each historical row, attach the matching forecast (same segment)
#  → take median forecast across 3 future months as "predicted_next_demand"
fc_agg = forecast.groupby(
    ["category","sub_category","region","market"]
)["forecasted_demand"].median().reset_index()
fc_agg.rename(columns={"forecasted_demand": "forecasted_demand_next3m"}, inplace=True)

base = base.merge(fc_agg, on=["category","sub_category","region","market"], how="left")
print(f"✅ After merge forecast: {base.shape}")

#  7.4  DERIVE FINAL RISK + ALERT COLUMNS

print("\nSTEP 7.4 — DERIVE ALERT COLUMNS")

# Low stock alert: inventory < reorder point
base["low_stock_alert"] = np.where(
    base["inventory_level"] < base["reorder_point"], "YES", "NO"
)

# Overstock alert: inventory > 3× ROP
base["overstock_alert"] = np.where(
    base["inventory_level"] > (base["reorder_point"] * 3), "YES", "NO"
)

# Demand forecast change vs historical
base["demand_forecast_delta"] = (
    base["forecasted_demand_next3m"] - base["units_sold"]
).round(2)
base["demand_trend"] = np.where(
    base["demand_forecast_delta"] > 0, "INCREASING",
    np.where(base["demand_forecast_delta"] < 0, "DECREASING", "STABLE")
)

# Fill any remaining NaN
for col in ["safety_stock","reorder_point","avg_daily_demand",
            "forecasted_demand_next3m","demand_forecast_delta"]:
    base[col] = base[col].fillna(0)
base["risk_flag"]        = base["risk_flag"].fillna("LOW")
base["stock_cover_flag"] = base["stock_cover_flag"].fillna("SAFE")
base["low_stock_alert"]  = base["low_stock_alert"].fillna("NO")
base["overstock_alert"]  = base["overstock_alert"].fillna("NO")

print("✅ Alert columns: low_stock_alert, overstock_alert, demand_trend")

#  7.5  SELECT & ORDER FINAL COLUMNS

final_cols = [
    # Identifiers
    "category","sub_category","region","market","year","month",
    # Demand (historical)
    "units_sold","revenue","profit","avg_unit_price",
    "avg_discount","avg_lead_time","n_orders",
    # Inventory
    "inventory_level","inventory_value",
    # Optimization outputs  ← KEY PROJECT OUTPUTS
    "avg_daily_demand","safety_stock","reorder_point",
    "days_of_stock","risk_flag","stock_cover_flag",
    # Forecast
    "forecasted_demand_next3m","demand_forecast_delta","demand_trend",
    # Alerts (for Power BI dashboard)
    "low_stock_alert","overstock_alert",
    # Profitability
    "profit_margin_pct",
]
final = base[final_cols].copy()


#  7.6  SAVE

final.to_csv(FINAL, index=False)

print("\n" + "=" * 60)
print("  ✅ FINAL DATASET SAVED")
print("=" * 60)
print(f"  Path  : {FINAL}")
print(f"  Shape : {final.shape[0]:,} rows  ×  {final.shape[1]} columns")

print("\n📋 COLUMNS IN FINAL FILE:")
for i, c in enumerate(final.columns, 1):
    print(f"  {i:2d}. {c}")

print("\n📊 RISK FLAG SUMMARY:")
print(final["risk_flag"].value_counts().to_string())

print("\n📊 DEMAND TREND SUMMARY:")
print(final["demand_trend"].value_counts().to_string())

print("\n📊 LOW STOCK ALERT:")
print(final["low_stock_alert"].value_counts().to_string())

print(f"\n✅ final_supply_chain_data.csv is ready for Power BI Dashboard!")
print("🎯 Next → Open step8_powerbi_guide.md and build your dashboard")