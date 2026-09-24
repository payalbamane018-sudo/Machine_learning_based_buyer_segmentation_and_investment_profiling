"""
Synthetic Financial Feature Generation
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

*** IMPORTANT ***
The raw clients.csv has NO income, transaction value, deal size, or
property-count field. To match a reference dashboard that displays
Total Investment, Deal Size, and Portfolio Size, this script GENERATES
those fields synthetically using plausible, documented rules tied to
existing real fields (client_type, acquisition_purpose, age, loan_applied).

These are illustrative / simulated numbers, NOT real financial data.
Any report or dashboard built on this file must disclose that these
fields are synthetic. Do not present them as real client financials.

Generation rules (deterministic seed = 42 for reproducibility):
  properties_owned:
    - Company client_type            -> 3-12 properties
    - Individual + Investment purpose -> 2-7 properties
    - Individual + Home purpose        -> 1-2 properties
  avg_purchase_price:
    - base $150k-$550k, scaled up for Company and Investment buyers,
      with mild upward drift for higher satisfaction_score (proxy for
      buying power / confidence)
  total_investment = properties_owned * avg_purchase_price * noise factor
  buying_tenure_years:
    - years since an assumed first purchase, loosely tied to age
      (older clients get a longer plausible tenure window)
"""

import numpy as np
import pandas as pd

IN_PATH = "cleaned_clients.csv"
OUT_PATH = "clients_with_synthetic_financials.csv"

np.random.seed(42)
df = pd.read_csv(IN_PATH)
n = len(df)

# ---------------------------------------------------------------
# properties_owned
# ---------------------------------------------------------------
properties = np.ones(n, dtype=int)
for i, row in df.iterrows():
    if row["client_type"] == "Company":
        properties[i] = np.random.randint(3, 13)
    elif row["acquisition_purpose"] == "Investment":
        properties[i] = np.random.randint(2, 8)
    else:
        properties[i] = np.random.randint(1, 3)
df["properties_owned"] = properties

# ---------------------------------------------------------------
# avg_purchase_price
# ---------------------------------------------------------------
base_price = np.random.normal(300_000, 70_000, n)
base_price += (df["client_type"] == "Company").astype(int) * 120_000
base_price += (df["acquisition_purpose"] == "Investment").astype(int) * 60_000
base_price += (df["satisfaction_score"] - 3) * 15_000
base_price = np.clip(base_price, 120_000, 600_000)
df["avg_purchase_price"] = base_price.round(-3).astype(int)

# ---------------------------------------------------------------
# total_investment
# ---------------------------------------------------------------
noise = np.random.uniform(0.85, 1.15, n)
df["total_investment"] = (df["properties_owned"] * df["avg_purchase_price"] * noise).round(-2).astype(int)

# ---------------------------------------------------------------
# buying_tenure_years (years since assumed first purchase)
# ---------------------------------------------------------------
max_tenure = np.clip((df["age"] - 20) / 2, 1, 20)
df["buying_tenure_years"] = (np.random.uniform(0.5, 1.0, n) * max_tenure).round(1)

df.to_csv(OUT_PATH, index=False)
print(f"Saved -> {OUT_PATH}")
print(df[["properties_owned", "avg_purchase_price", "total_investment", "buying_tenure_years"]].describe())
