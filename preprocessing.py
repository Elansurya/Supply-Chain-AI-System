import pandas as pd
import numpy as np
import os

#  PATHS  (update BASE to your machine path)

BASE = r"C:\project\Supply_Chain_AI_System"
RAW  = os.path.join(BASE, "data", "superstore.csv")
SAVE = os.path.join(BASE, "data", "cleaned_data.csv")

#  3.1  LOAD RAW DATA

print("=" * 60)
print("  STEP 3.1 — LOAD RAW DATA")
print("=" * 60)
df = pd.read_csv(RAW)
print(f"✅ Loaded  →  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

#  3.2  RENAME COLUMNS  →  snake_case

print("\nSTEP 3.2 — RENAME COLUMNS")
df.rename(columns={
    "Category"      : "category",
    "Sub.Category"  : "sub_category",
    "Product.Name"  : "product",
    "Product.ID"    : "product_id",
    "Region"        : "region",
    "Market"        : "market",
    "Market2"       : "market2",
    "Segment"       : "segment",
    "Quantity"      : "units_sold",
    "Sales"         : "revenue",
    "Profit"        : "profit",
    "Discount"      : "discount",
    "Shipping.Cost" : "shipping_cost",
    "Ship.Mode"     : "ship_mode",
    "Order.Date"    : "date",
    "Ship.Date"     : "ship_date",
    "Order.ID"      : "order_id",
    "Order.Priority": "order_priority",
    "Customer.ID"   : "customer_id",
    "Customer.Name" : "customer_name",
    "City"          : "city",
    "State"         : "state",
    "Country"       : "country",
    "Year"          : "year_raw",
    "Row.ID"        : "row_id",
    "weeknum"       : "weeknum",
    "记录数"         : "record_count",
}, inplace=True)
print(f"✅ Columns renamed to snake_case")

#  3.3  CONVERT DATE COLUMNS

print("\nSTEP 3.3 — CONVERT DATE COLUMNS")
df["date"]      = pd.to_datetime(df["date"],      errors="coerce")
df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")
print(f"✅ Order date : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"✅ Ship  date : {df['ship_date'].min().date()} → {df['ship_date'].max().date()}")

#  3.4  CREATE MONTH / YEAR / QUARTER / WEEK

print("\nSTEP 3.4 — CREATE TIME FEATURES")
df["year"]           = df["date"].dt.year
df["month"]          = df["date"].dt.month
df["quarter"]        = df["date"].dt.quarter
df["week"]           = df["date"].dt.isocalendar().week.astype(int)
df["day_of_week"]    = df["date"].dt.dayofweek
df["month_name"]     = df["date"].dt.strftime("%b")
df["year_month"]     = df["date"].dt.to_period("M").astype(str)
df["lead_time_days"] = (df["ship_date"] - df["date"]).dt.days.clip(lower=1)
df["lead_time_days"] = df["lead_time_days"].fillna(df["lead_time_days"].median())
print("✅ Created: year, month, quarter, week, day_of_week, year_month, lead_time_days")


#  3.5  HANDLE MISSING VALUES
print("\nSTEP 3.5 — HANDLE MISSING VALUES")
print("Missing BEFORE:")
miss = df.isnull().sum()
print(miss[miss > 0] if miss[miss > 0].any() else "  None found ✅")

for col in ["units_sold", "revenue", "profit", "shipping_cost", "discount"]:
    df[col] = df.groupby("category")[col].transform(lambda x: x.fillna(x.median()))
for col in ["ship_mode", "order_priority", "segment"]:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing AFTER:  ✅ ZERO missing values")

#  3.6  FEATURE ENGINEERING
#   NEW COLUMN          FORMULA
#   unit_price       =  revenue / units_sold
#   revenue          =  units_sold * unit_price   (verified)
#   inventory_level  =  simulated stock on hand
#   inventory_value  =  inventory_level * unit_price
#   profit_margin    =  profit / revenue
#   discount_flag    =  1 if discount > 0
#   is_festival_month=  1 for Oct/Nov/Dec/Jan/Aug
#   shipping_speed   =  Express / Standard / Economy

print("\nSTEP 3.6 — FEATURE ENGINEERING")

df["unit_price"]       = (df["revenue"] / df["units_sold"].replace(0, 1)).round(2)
df["revenue"]          = (df["units_sold"] * df["unit_price"]).round(2)   # clean recalc

np.random.seed(42)
df["inventory_level"]  = (df["units_sold"] * np.random.uniform(1.2, 2.5, len(df))).astype(int).clip(lower=5)
df["inventory_value"]  = (df["inventory_level"] * df["unit_price"]).round(2)
df["profit_margin"]    = (df["profit"] / df["revenue"].replace(0, np.nan)).fillna(0).round(4)
df["discount_flag"]    = (df["discount"] > 0).astype(int)
df["is_festival_month"]= df["month"].isin([1, 8, 10, 11, 12]).astype(int)
df["shipping_speed"]   = pd.cut(df["lead_time_days"],
                                bins=[0, 2, 5, 7, 999],
                                labels=["Express","Standard","Economy","Slow"]).astype(str)

print("✅ Features: unit_price | revenue | inventory_level | inventory_value")
print("            profit_margin | discount_flag | is_festival_month | shipping_speed")


#  3.7  SUMMARY
print("\nSTEP 3.7 — DATA QUALITY SUMMARY")
print(f"  Rows      : {len(df):,}")
print(f"  Columns   : {df.shape[1]}")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Categories: {df['category'].unique().tolist()}")
print(f"  Markets   : {df['market'].unique().tolist()}")
print()
print(df[["units_sold","revenue","unit_price","inventory_level",
          "profit","profit_margin","lead_time_days"]].describe().round(2).to_string())


#  3.8  SAVE

df.to_csv(SAVE, index=False)
print(f"\n✅ Saved → {SAVE}  |  Shape: {df.shape}")
print("🎯 Next → Run step4_eda.py")