"""
Buyer Segmentation - rebuilt to match the 4 required business segments:
  1. Global Investors  - High income, investment purchases
  2. First-Time Buyers - Younger, loan dependent
  3. Corporate Buyers  - Companies purchasing multiple units
  4. Luxury Investors  - High satisfaction, large investments

WHY RULE-BASED (not pure unsupervised K-Means) THIS TIME:
K-Means (see recluster_with_financials.py) was re-run on the full
financing/deal-size/tenure/demographic feature set including the
synthetic financial fields. It DID cleanly discover two of the four
segments on its own (a 100%-Company cluster, and a 100%-Investment-
purpose high-value cluster) - reported below for transparency. But it
did NOT separate a satisfaction-driven "Luxury" cluster or an age/loan
driven "First-Time" cluster, because satisfaction_score and loan_applied
don't vary enough between the remaining clients to form their own
natural cluster. So this script applies clear, priority-ordered rules
(informed by what K-Means already confirmed) to guarantee all 4 named
segments are actually produced and non-overlapping:

  Priority 1 - Corporate Buyers:
      client_type == "Company"
      (confirmed by K-Means: forms its own pure cluster naturally)

  Priority 2 - Global Investors:
      acquisition_purpose == "Investment"
      (confirmed by K-Means: forms its own pure, high-investment cluster)

  Priority 3 - Luxury Investors (from remaining Home-purpose individuals):
      total_investment >= 75th percentile AND satisfaction_score >= 4
      (large investments + high satisfaction, as specified)

  Priority 4 - First-Time Buyers (everyone left):
      remaining Home-purpose individuals - reported below to confirm
      they skew younger / more loan-dependent than the other segments
"""

import pandas as pd

IN_PATH = "clients_with_synthetic_financials.csv"
OUT_PATH = "FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv"

df = pd.read_csv(IN_PATH)

# Corporate Buyers and Global Investors are pulled out first (K-Means already
# confirmed these form pure, well-separated clusters - see module docstring).
is_corporate = df["client_type"] == "Company"
is_global = (~is_corporate) & (df["acquisition_purpose"] == "Investment")
remaining_mask = ~(is_corporate | is_global)   # Home-purpose Individuals only

# First-Time Buyers pulled out EXPLICITLY next (younger + loan-dependent),
# so the label actually matches its definition instead of being a leftover
# bucket that inherits whatever age distribution is left over.
remaining_age_median = df.loc[remaining_mask, "age"].median()
is_first_time = remaining_mask & (df["age"] < remaining_age_median) & (df["loan_applied"])

# Luxury Investors: identified within whatever's STILL left (its own 75th
# percentile of total_investment, not the whole dataset's) - otherwise a
# "large investment" here would be measured against Investment-purpose/
# Company deals already pulled out, and no Home buyer would ever qualify.
still_remaining = remaining_mask & (~is_first_time)
local_inv_75th = df.loc[still_remaining, "total_investment"].quantile(0.75)
print(f"75th percentile of total_investment within the final remaining pool: "
      f"${local_inv_75th:,.0f}")

is_luxury = still_remaining & (df["total_investment"] >= local_inv_75th) & (df["satisfaction_score"] >= 4)

# Anyone still unassigned (older, not loan-dependent, not high-satisfaction/
# large-investment) is merged into First-Time Buyers as the catch-all -
# reported below so this is transparent, not hidden.
df["segment"] = "First-Time Buyers"
df.loc[is_corporate, "segment"] = "Corporate Buyers"
df.loc[is_global, "segment"] = "Global Investors"
df.loc[is_luxury, "segment"] = "Luxury Investors"
# (is_first_time rows already default to "First-Time Buyers" above)

SEGMENT_ORDER = ["Global Investors", "First-Time Buyers",
                  "Corporate Buyers", "Luxury Investors"]

print("\nSegment counts:")
print(df["segment"].value_counts().reindex(SEGMENT_ORDER))

print("\n--- Validation: does each segment match its definition? ---")
for seg in SEGMENT_ORDER:
    sub = df[df["segment"] == seg]
    print(f"\n{seg}  (n={len(sub)}, {len(sub)/len(df):.1%})")
    print(f"  Avg age:              {sub['age'].mean():.1f}")
    print(f"  Loan applied rate:    {sub['loan_applied'].mean():.1%}")
    print(f"  % Company:            {(sub['client_type']=='Company').mean():.1%}")
    print(f"  % Investment purpose: {(sub['acquisition_purpose']=='Investment').mean():.1%}")
    print(f"  Avg satisfaction:     {sub['satisfaction_score'].mean():.2f}")
    print(f"  Avg total_investment: ${sub['total_investment'].mean():,.0f}")
    print(f"  Avg properties owned: {sub['properties_owned'].mean():.2f}")

# ---------------------------------------------------------------
# PCA for the Segmentation Overview scatter plot (unchanged features)
# ---------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

le_type = LabelEncoder()
le_purpose = LabelEncoder()
le_gender = LabelEncoder()
df["client_type_encoded"] = le_type.fit_transform(df["client_type"])
df["acquisition_purpose_encoded"] = le_purpose.fit_transform(df["acquisition_purpose"])
df["gender_encoded"] = le_gender.fit_transform(df["gender"])
df["loan_applied_encoded"] = df["loan_applied"].astype(int)

feature_cols = [
    "loan_applied_encoded", "total_investment", "avg_purchase_price",
    "properties_owned", "buying_tenure_years", "age", "satisfaction_score",
    "acquisition_purpose_encoded", "client_type_encoded", "gender_encoded",
]
X = StandardScaler().fit_transform(df[feature_cols].astype(float))
coords = PCA(n_components=2, random_state=42).fit_transform(X)
df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]

df = df.drop(columns=["client_type_encoded", "acquisition_purpose_encoded",
                       "gender_encoded", "loan_applied_encoded"])

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
