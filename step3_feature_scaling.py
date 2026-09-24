"""
Step 3 - Feature Scaling
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Scales numeric fields so no single variable dominates distance-based
clustering (K-Means, hierarchical, etc.) just because it has a larger
numeric range.

  StandardScaler -> mean 0, std 1. Default choice for K-Means since it
                     handles roughly-normal, unbounded variables well.
  MinMaxScaler   -> rescales to [0, 1]. Useful if you want scaled values
                     to stay in a fixed, interpretable range (e.g. next
                     to the already-0/1 one-hot columns from Step 2).

Both are produced below - use whichever the clustering step calls for.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

IN_PATH = "encoded_clients.csv"
OUT_STANDARD_PATH = "scaled_clients_standard.csv"
OUT_MINMAX_PATH = "scaled_clients_minmax.csv"

NUMERIC_COLS = ["age", "satisfaction_score"]

# ---------------------------------------------------------------
# 1. Load encoded data (output of Step 2)
# ---------------------------------------------------------------
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(df[NUMERIC_COLS].describe())

# ---------------------------------------------------------------
# 2. StandardScaler version (mean=0, std=1)
# ---------------------------------------------------------------
df_standard = df.copy()
standard_scaler = StandardScaler()
df_standard[NUMERIC_COLS] = standard_scaler.fit_transform(df_standard[NUMERIC_COLS])

print("\nStandardScaler result (should be ~mean 0, std 1):")
print(df_standard[NUMERIC_COLS].describe().loc[["mean", "std"]])

df_standard.to_csv(OUT_STANDARD_PATH, index=False)
print(f"Saved -> {OUT_STANDARD_PATH}")

# ---------------------------------------------------------------
# 3. MinMaxScaler version (range 0 to 1)
# ---------------------------------------------------------------
df_minmax = df.copy()
minmax_scaler = MinMaxScaler()
df_minmax[NUMERIC_COLS] = minmax_scaler.fit_transform(df_minmax[NUMERIC_COLS])

print("\nMinMaxScaler result (should be min 0, max 1):")
print(df_minmax[NUMERIC_COLS].describe().loc[["min", "max"]])

df_minmax.to_csv(OUT_MINMAX_PATH, index=False)
print(f"Saved -> {OUT_MINMAX_PATH}")
