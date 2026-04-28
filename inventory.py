import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


#  PATHS

BASE  = r"C:\project\Supply_Chain_AI_System"
DATA  = os.path.join(BASE, "data",    "cleaned_data.csv")
OUT   = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid")
PAL = ["#2196F3","#FF7043","#4CAF50","#9C27B0","#FFC107"]

Z = 1.65   # Z-score for 95% service level

#  LOAD

df = pd.read_csv(DATA, parse_dates=["date"])
print(f"✅ Loaded  →  {df.shape[0]:,} rows")

#  6.1  COMPUTE PER-SEGMENT STATISTICS

print("\nSTEP 6.1 — COMPUTE SEGMENT STATISTICS")
print("-" * 50)

inv = df.groupby(["category","sub_category","region","market"]).agg(
    total_units_sold  = ("units_sold",      "sum"),
    demand_std        = ("units_sold",      "std"),
    avg_lead_time     = ("lead_time_days",  "mean"),
    avg_unit_price    = ("unit_price",      "mean"),
    avg_inventory     = ("inventory_level", "mean"),
    total_revenue     = ("revenue",         "sum"),
    total_profit      = ("profit",          "sum"),
    total_days        = ("date", lambda x: (x.max() - x.min()).days + 1),
    n_orders          = ("units_sold",      "count"),
).reset_index()

# Guard against zero / NaN
inv["demand_std"]    = inv["demand_std"].fillna(1).clip(lower=0.1)
inv["total_days"]    = inv["total_days"].clip(lower=1)
inv["avg_lead_time"] = inv["avg_lead_time"].clip(lower=1)

print(f"✅ Unique segments  :  {len(inv)}")

#  6.2  AVERAGE DAILY DEMAND (ADD)
#       Formula:  ADD = Total units sold / Number of days

print("\nSTEP 6.2 — AVERAGE DAILY DEMAND (ADD)")
inv["avg_daily_demand"] = (inv["total_units_sold"] / inv["total_days"]).round(4)
print("✅ Formula applied:  ADD = Total_Units_Sold / Total_Days")
print(inv[["category","sub_category","avg_daily_demand"]].head(8).to_string(index=False))


#  6.3  SAFETY STOCK
#       Formula:  Safety Stock = Z × Demand_StdDev × √(Lead_Time)
#       Z = 1.65  →  95% service level

print("\nSTEP 6.3 — SAFETY STOCK")
inv["safety_stock"] = (Z * inv["demand_std"] * np.sqrt(inv["avg_lead_time"])).round(2)
print("✅ Formula: Safety Stock = 1.65 × Demand_StdDev × √Lead_Time")
print(inv[["category","sub_category","demand_std","avg_lead_time","safety_stock"]].head(8).to_string(index=False))

#  6.4  REORDER POINT (ROP)
#       Formula:  ROP = (ADD × Lead_Time) + Safety_Stock

print("\nSTEP 6.4 — REORDER POINT (ROP)")
inv["reorder_point"] = (
    (inv["avg_daily_demand"] * inv["avg_lead_time"]) + inv["safety_stock"]
).round(2)
print("✅ Formula: ROP = (ADD × Lead_Time) + Safety_Stock")
print(inv[["category","sub_category","avg_daily_demand",
           "avg_lead_time","safety_stock","reorder_point"]].head(8).to_string(index=False))

#  6.5  RISK FLAG
#       🔴 HIGH   → avg_inventory < safety_stock
#       🟡 MEDIUM → avg_inventory < reorder_point
#       🟢 LOW    → avg_inventory >= reorder_point

print("\nSTEP 6.5 — RISK FLAG ASSIGNMENT")

def assign_risk(row):
    if row["avg_inventory"] < row["safety_stock"]:
        return "HIGH"
    elif row["avg_inventory"] < row["reorder_point"]:
        return "MEDIUM"
    else:
        return "LOW"

inv["risk_flag"] = inv.apply(assign_risk, axis=1)

risk_summary = inv["risk_flag"].value_counts()
print("✅ Risk Distribution:")
print(risk_summary.to_string())

print("\nSample — HIGH risk segments:")
print(inv[inv["risk_flag"] == "HIGH"][
    ["category","sub_category","region","avg_inventory",
     "safety_stock","reorder_point","risk_flag"]
].head(8).to_string(index=False))

#  6.6  ADDITIONAL KPIs

inv["days_of_stock"]     = (inv["avg_inventory"] / inv["avg_daily_demand"].replace(0, np.nan)).round(1)
inv["stock_cover_flag"]  = np.where(inv["days_of_stock"] < inv["avg_lead_time"],
                                    "CRITICAL", "SAFE")
inv["profit_margin_pct"] = (inv["total_profit"] / inv["total_revenue"].replace(0, np.nan) * 100).round(2)

#  6.7  INVENTORY OPTIMIZATION CHARTS

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Inventory Optimization Analysis", fontsize=16, fontweight="bold")

# Chart 1: Safety Stock by Category
ss = inv.groupby("category")["safety_stock"].mean().sort_values(ascending=False)
axes[0, 0].bar(ss.index, ss.values, color=PAL[:3], edgecolor="white")
axes[0, 0].set_title("🛡 Avg Safety Stock by Category", fontweight="bold")
axes[0, 0].set_ylabel("Safety Stock (Units)")

# Chart 2: Reorder Point by Category
rop = inv.groupby("category")["reorder_point"].mean().sort_values(ascending=False)
axes[0, 1].bar(rop.index, rop.values, color=[PAL[3], PAL[4], PAL[2]], edgecolor="white")
axes[0, 1].set_title("🔄 Avg Reorder Point by Category", fontweight="bold")
axes[0, 1].set_ylabel("Reorder Point (Units)")

# Chart 3: Risk Flag Pie
rc = inv["risk_flag"].value_counts()
risk_colors = {"HIGH": "#F44336", "MEDIUM": "#FFC107", "LOW": "#4CAF50"}
axes[0, 2].pie(rc.values, labels=rc.index, autopct="%1.1f%%",
               colors=[risk_colors.get(l, "gray") for l in rc.index],
               startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[0, 2].set_title("⚠️ Inventory Risk Distribution", fontweight="bold")

# Chart 4: Safety Stock vs ROP Scatter
sc = axes[1, 0].scatter(inv["safety_stock"], inv["reorder_point"],
                        c=inv["avg_lead_time"], cmap="RdYlGn_r",
                        alpha=0.6, s=35, edgecolors="white")
plt.colorbar(sc, ax=axes[1, 0], label="Avg Lead Time (days)")
axes[1, 0].set_title("🔍 Safety Stock vs Reorder Point\n(color = Lead Time)", fontweight="bold")
axes[1, 0].set_xlabel("Safety Stock"); axes[1, 0].set_ylabel("Reorder Point")

# Chart 5: Days of Stock by Category
dos = inv.groupby("category")["days_of_stock"].mean().sort_values(ascending=False)
axes[1, 1].bar(dos.index, dos.values, color=PAL[:3], edgecolor="white")
axes[1, 1].set_title("📦 Avg Days of Stock by Category", fontweight="bold")
axes[1, 1].set_ylabel("Days of Stock")
axes[1, 1].axhline(y=7, color="red", linestyle="--", linewidth=1.5, label="Min 7 days")
axes[1, 1].legend()

# Chart 6: HIGH risk — top 10 sub-categories
hi = inv[inv["risk_flag"] == "HIGH"].groupby("sub_category")["avg_inventory"].mean().sort_values().head(10)
axes[1, 2].barh(hi.index, hi.values, color="#F44336", edgecolor="white")
axes[1, 2].set_title("🚨 HIGH Risk Sub-Categories\n(Low Inventory)", fontweight="bold")
axes[1, 2].set_xlabel("Avg Inventory Level")

plt.tight_layout()
chart_path = os.path.join(OUT, "inventory_optimization.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Chart saved → {chart_path}")

#  6.8  SAVE INVENTORY STATS
inv_path = os.path.join(BASE, "data", "inventory_stats.csv")
inv.to_csv(inv_path, index=False)
print(f"✅ Inventory stats saved → {inv_path}")
print(f"   Shape: {inv.shape}")

print("\n📊 INVENTORY SUMMARY:")
print(f"  Total segments analyzed : {len(inv):,}")
print(f"  HIGH  risk segments     : {(inv['risk_flag']=='HIGH').sum():,}")
print(f"  MEDIUM risk segments    : {(inv['risk_flag']=='MEDIUM').sum():,}")
print(f"  LOW   risk segments     : {(inv['risk_flag']=='LOW').sum():,}")
print(f"  CRITICAL stock cover    : {(inv['stock_cover_flag']=='CRITICAL').sum():,}")

print("\n🎯 Next → Run step7_export_final.py")