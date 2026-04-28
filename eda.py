import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

#  PATHS

BASE  = r"C:\project\Supply_Chain_AI_System"
DATA  = os.path.join(BASE, "data",    "cleaned_data.csv")
OUT   = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

#  LOAD

df = pd.read_csv(DATA, parse_dates=["date","ship_date"])
sns.set_theme(style="whitegrid", palette="muted")
PAL = ["#2196F3","#FF7043","#4CAF50","#9C27B0","#FFC107","#00BCD4","#E91E63"]
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print(f"✅ Loaded cleaned data  →  {df.shape[0]:,} rows")

#  FIGURE 1 — DEMAND & REVENUE OVERVIEW  (2 × 3 grid)

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Supply Chain EDA — Demand & Revenue Overview",
             fontsize=17, fontweight="bold", y=1.01)

# ── 4.1  Monthly Demand Trend 
monthly = df.groupby("year_month")["units_sold"].sum().reset_index()
ax = axes[0, 0]
ax.plot(monthly["year_month"], monthly["units_sold"],
        marker="o", color=PAL[0], linewidth=2, markersize=3)
ax.fill_between(range(len(monthly)), monthly["units_sold"], alpha=0.15, color=PAL[0])
ax.set_title("📈 Monthly Demand Trend", fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Units Sold")
ax.set_xticks(range(0, len(monthly), 4))
ax.set_xticklabels(monthly["year_month"].iloc[::4], rotation=45, fontsize=8)

# ── 4.2  Category-wise Demand 
cat_d = df.groupby("category")["units_sold"].sum().sort_values(ascending=False)
ax = axes[0, 1]
bars = ax.bar(cat_d.index, cat_d.values, color=PAL[:3], edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, cat_d.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200, f"{val:,}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("📦 Category-wise Total Demand", fontweight="bold")
ax.set_ylabel("Units Sold")

# ── 4.3  Region-wise Demand 
reg_d = (df.groupby("region")["units_sold"].sum()
           .sort_values().tail(10))            # top 10 regions
ax = axes[0, 2]
cols = plt.cm.Blues(np.linspace(0.35, 0.9, len(reg_d)))
bars = ax.barh(reg_d.index, reg_d.values, color=cols, edgecolor="white")
for bar, val in zip(bars, reg_d.values):
    ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=7)
ax.set_title("🌍 Region-wise Demand (Top 10)", fontweight="bold")
ax.set_xlabel("Units Sold")

# ── 4.4  Seasonal Pattern 
season = df.groupby("month")["units_sold"].mean()
ax = axes[1, 0]
bar_cols = [PAL[1] if m in [1, 8, 10, 11, 12] else PAL[0] for m in range(1, 13)]
ax.bar(MON, season.values, color=bar_cols, edgecolor="white")
ax.set_title("🗓 Seasonal Pattern  (🟠 = Festival Month)", fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Avg Units Sold")
ax.tick_params(axis="x", rotation=30)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PAL[1], label="Festival"),
                   Patch(color=PAL[0], label="Normal")], fontsize=8)

# ── 4.5  Revenue Share by Category — Donut 
rev_cat = df.groupby("category")["revenue"].sum()
ax = axes[1, 1]
wedges, texts, autos = ax.pie(rev_cat.values, labels=rev_cat.index,
                               autopct="%1.1f%%", colors=PAL[:3],
                               wedgeprops=dict(width=0.5), startangle=140)
for a in autos: a.set_fontsize(11)
ax.set_title("💰 Revenue Share by Category", fontweight="bold")

# ── 4.6  Heatmap Category × Month 
pivot = df.pivot_table(index="category", columns="month",
                       values="units_sold", aggfunc="mean").fillna(0)
pivot.columns = [MON[c - 1] for c in pivot.columns]
ax = axes[1, 2]
sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title("🔥 Avg Demand Heatmap (Category × Month)", fontweight="bold")

plt.tight_layout()
p1 = os.path.join(OUT, "eda_overview.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved → {p1}")

#  FIGURE 2 — BUSINESS INSIGHTS  (2 × 3 grid)

fig2, axes2 = plt.subplots(2, 3, figsize=(20, 11))
fig2.suptitle("Supply Chain EDA — Business Insights",
              fontsize=17, fontweight="bold", y=1.01)

# ── 4.7  Profit Margin by Category 
pm = df.groupby("category")["profit_margin"].mean().sort_values(ascending=False)
ax = axes2[0, 0]
ax.bar(pm.index, pm.values * 100, color=PAL[2:5], edgecolor="white")
ax.set_title("📊 Avg Profit Margin % by Category", fontweight="bold")
ax.set_ylabel("Profit Margin (%)")

# ── 4.8  Discount vs Profit 
sample = df.sample(3000, random_state=1)
color_map = {"Technology": PAL[0], "Furniture": PAL[1], "Office Supplies": PAL[2]}
sc_colors = [color_map.get(c, "gray") for c in sample["category"]]
ax = axes2[0, 1]
ax.scatter(sample["discount"], sample["profit"], c=sc_colors, alpha=0.4, s=18)
ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
ax.set_title("🔍 Discount vs Profit (Loss Zone = below red)", fontweight="bold")
ax.set_xlabel("Discount Rate"); ax.set_ylabel("Profit ($)")
ax.legend(handles=[Patch(color=v, label=k) for k, v in color_map.items()], fontsize=8)

# ── 4.9  Yearly Revenue Growth 
yr = df.groupby("year")["revenue"].sum()
ax = axes2[0, 2]
ax.plot(yr.index, yr.values, marker="D", color=PAL[3], linewidth=2.5, markersize=9)
for x, y in zip(yr.index, yr.values):
    ax.annotate(f"${y/1e6:.2f}M", (x, y),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, fontweight="bold")
ax.set_title("📅 Yearly Revenue Growth", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Total Revenue ($)")

# ── 4.10  Top 10 Sub-Categories by Revenue 
top_sub = df.groupby("sub_category")["revenue"].sum().sort_values(ascending=False).head(10)
ax = axes2[1, 0]
ax.barh(top_sub.index[::-1], top_sub.values[::-1],
        color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
ax.set_title("🏆 Top 10 Sub-Categories by Revenue", fontweight="bold")
ax.set_xlabel("Revenue ($)")

# ── 4.11  Market-wise Units Sold 
mkt = df.groupby("market")["units_sold"].sum().sort_values(ascending=False)
ax = axes2[1, 1]
ax.bar(mkt.index, mkt.values, color=PAL[:len(mkt)], edgecolor="white")
ax.set_title("🌐 Market-wise Total Demand", fontweight="bold")
ax.set_ylabel("Units Sold")
ax.tick_params(axis="x", rotation=30)

# ── 4.12  Shipment Mode Distribution 
ship = df["ship_mode"].value_counts()
ax = axes2[1, 2]
ax.pie(ship.values, labels=ship.index, autopct="%1.1f%%",
       colors=PAL[:len(ship)], startangle=90,
       wedgeprops=dict(edgecolor="white", linewidth=1.5))
ax.set_title("🚚 Shipment Mode Distribution", fontweight="bold")

plt.tight_layout()
p2 = os.path.join(OUT, "eda_insights.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved → {p2}")

#  PRINT BUSINESS INSIGHTS

print("\n" + "=" * 60)
print("  📊 KEY BUSINESS INSIGHTS")
print("=" * 60)

fest   = df[df["is_festival_month"] == 1]
normal = df[df["is_festival_month"] == 0]
lift   = ((fest.groupby("category")["units_sold"].mean() -
           normal.groupby("category")["units_sold"].mean()) /
           normal.groupby("category")["units_sold"].mean() * 100).round(1)
print("\n🎉 Festival-month demand LIFT (%):")
print(lift.sort_values(ascending=False).to_string())

cv = (df.groupby("region")["units_sold"].std() /
      df.groupby("region")["units_sold"].mean()).round(3)
print("\n📌 Demand Variability CV by Region (lower = more stable):")
print(cv.sort_values().to_string())

top5 = df.groupby("sub_category")["profit"].sum().sort_values(ascending=False).head(5)
print("\n💰 Top 5 Most Profitable Sub-Categories:")
print(top5.to_string())

bot5 = df.groupby("sub_category")["profit"].sum().sort_values().head(5)
print("\n⚠️  Top 5 Loss-Making Sub-Categories:")
print(bot5.to_string())

print("\n🎯 Next → Run step5_forecasting.py")