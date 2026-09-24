"""
Step 1 - Data Cleaning
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

What this does:
1. Load raw client data
2. Standardize date_of_birth (file has two mixed formats) -> derive age
3. Normalize categorical labels (trim spaces, consistent casing)
4. Handle missing values (robust even though none exist today)
5. Remove duplicate client entries
6. Drop PII not needed for modeling (names)
7. Save cleaned_clients.csv for Step 2 (EDA)
"""

import pandas as pd
import numpy as np

RAW_PATH = "clients.csv"
OUT_PATH = "cleaned_clients.csv"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Standardize date_of_birth -> age
#    Two formats found in the file:
#      DD-MM-YYYY  (dash separated)
#      MM/DD/YYYY  (slash separated, confirmed by day values > 12)
# ---------------------------------------------------------------
def parse_dob(value):
    if pd.isna(value):
        return pd.NaT
    value = str(value).strip()
    try:
        if "-" in value:
            return pd.to_datetime(value, format="%d-%m-%Y")
        elif "/" in value:
            return pd.to_datetime(value, format="%m/%d/%Y")
        else:
            return pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return pd.NaT

df["date_of_birth"] = df["date_of_birth"].apply(parse_dob)

# Flag any dates that failed to parse
bad_dates = df["date_of_birth"].isna().sum()
if bad_dates:
    print(f"Warning: {bad_dates} date_of_birth values could not be parsed")

today = pd.Timestamp.today()
df["age"] = ((today - df["date_of_birth"]).dt.days // 365).astype("Int64")

# Sanity-check age range; flag anything implausible instead of silently keeping it
implausible = df[(df["age"] < 18) | (df["age"] > 100)]
if len(implausible):
    print(f"Warning: {len(implausible)} clients have implausible ages (<18 or >100)")

# ---------------------------------------------------------------
# 3. Normalize categorical labels
# ---------------------------------------------------------------
categorical_cols = [
    "client_type", "gender", "country", "region",
    "acquisition_purpose", "loan_applied", "referral_channel"
]

for col in categorical_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.title()          # consistent casing, e.g. "usa" -> "Usa"; fix below for acronyms
    )

# Fix casing for known acronyms that .title() breaks (USA, UK)
acronym_fix = {"Usa": "USA", "Uk": "UK"}
df["country"] = df["country"].replace(acronym_fix)

# Normalize Yes/No fields to boolean for downstream modeling convenience
df["loan_applied"] = df["loan_applied"].map({"Yes": True, "No": False})

# ---------------------------------------------------------------
# 4. Handle missing values
#    (Dataset currently has none, but this keeps the pipeline robust
#     if the raw export changes)
# ---------------------------------------------------------------
missing_before = df.isna().sum().sum()
print(f"Missing values before fill: {missing_before}")

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

df["satisfaction_score"] = df["satisfaction_score"].fillna(
    df["satisfaction_score"].median()
)

# ---------------------------------------------------------------
# 5. Remove duplicate client entries
# ---------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["client_id"], keep="first")
df = df.drop_duplicates(
    subset=[c for c in df.columns if c != "client_id"], keep="first"
)
after = len(df)
print(f"Removed {before - after} duplicate rows")

# ---------------------------------------------------------------
# 6. Drop PII not needed for clustering/modeling
# ---------------------------------------------------------------
df = df.drop(columns=["first_name", "last_name"])

# ---------------------------------------------------------------
# 7. Save cleaned dataset
# ---------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print(f"Saved cleaned data -> {OUT_PATH}")
print(df.head())
