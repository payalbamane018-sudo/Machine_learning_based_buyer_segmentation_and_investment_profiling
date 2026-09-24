"""
Step 2 - Feature Encoding
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Encoding strategy:
  Label Encoding  -> binary / ordinal-ish fields (2 categories)
      client_type, acquisition_purpose
  One-Hot Encoding -> nominal fields with 3+ categories, no inherent order
      region, referral_channel, country

Why the split: clustering algorithms (K-Means, etc.) treat label-encoded
integers as having distance/order between them, which is wrong for a
field like "country". One-hot avoids inventing a false ranking, so it's
used for anything with 3+ unstructured categories. Binary fields collapse
to a single 0/1 column either way, so label encoding is simpler and
equivalent there.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

IN_PATH = "cleaned_clients.csv"
OUT_PATH = "encoded_clients.csv"
MAPPING_PATH = "label_encoding_mappings.csv"

# ---------------------------------------------------------------
# 1. Load cleaned data (output of Step 1)
# ---------------------------------------------------------------
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Label Encoding (binary fields)
# ---------------------------------------------------------------
label_cols = ["client_type", "acquisition_purpose"]
label_mappings = {}

for col in label_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"{col} mapping: {label_mappings[col]}")

# ---------------------------------------------------------------
# 3. One-Hot Encoding (multi-category nominal fields)
# ---------------------------------------------------------------
onehot_cols = ["region", "referral_channel", "country"]

df = pd.get_dummies(
    df,
    columns=onehot_cols,
    prefix=onehot_cols,
    drop_first=False   # keep all categories explicit; useful for cluster profiling later
)

# One-hot columns come out as True/False - cast to 0/1 for modeling
onehot_generated = [c for c in df.columns if c.startswith(tuple(f"{p}_" for p in onehot_cols))]
df[onehot_generated] = df[onehot_generated].astype(int)

# ---------------------------------------------------------------
# 4. Drop original categorical text columns for the label-encoded fields
#    (keep the one-hot base columns already replaced by get_dummies)
# ---------------------------------------------------------------
df = df.drop(columns=label_cols)

# ---------------------------------------------------------------
# 5. Save encoded dataset + mapping reference
# ---------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)

mapping_rows = []
for col, mapping in label_mappings.items():
    for category, code in mapping.items():
        mapping_rows.append({"field": col, "category": category, "encoded_value": code})
pd.DataFrame(mapping_rows).to_csv(MAPPING_PATH, index=False)

print(f"\nSaved encoded data -> {OUT_PATH}  ({df.shape[1]} columns)")
print(f"Saved label encoding reference -> {MAPPING_PATH}")
print("\nColumns after encoding:")
print(list(df.columns))
