"""
Step 6 - Cluster Interpretation
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Analyzes each K-Means cluster (k=4, from Step 4/5) along four dimensions:
  1. Investment purpose   -> acquisition_purpose (Home vs Investment)
  2. Geographic distribution -> country, region (top values per cluster)
  3. Loan behavior         -> loan_applied rate
  4. Customer demographics -> age, gender, client_type
"""

import pandas as pd

CLUSTERED_PATH = "clustered_clients.csv"
CLEANED_PATH = "cleaned_clients.csv"

clustered = pd.read_csv(CLUSTERED_PATH)[["client_id", "kmeans_cluster"]]
raw = pd.read_csv(CLEANED_PATH)
df = raw.merge(clustered, on="client_id")

clusters = sorted(df["kmeans_cluster"].unique())
summary_rows = []

print("=" * 70)
for c in clusters:
    sub = df[df["kmeans_cluster"] == c]
    n = len(sub)
    print(f"\nCLUSTER {c}  (n={n}, {n/len(df):.1%} of clients)")

    # 1. Investment purpose
    purpose_pct = sub["acquisition_purpose"].value_counts(normalize=True).round(3)
    print(f"  Investment purpose : {purpose_pct.to_dict()}")

    # 2. Geographic distribution (top 3 countries, top 3 regions)
    top_countries = sub["country"].value_counts(normalize=True).head(3).round(3)
    top_regions = sub["region"].value_counts(normalize=True).head(3).round(3)
    print(f"  Top countries       : {top_countries.to_dict()}")
    print(f"  Top regions         : {top_regions.to_dict()}")

    # 3. Loan behavior
    loan_rate = sub["loan_applied"].mean()
    print(f"  Loan applied rate   : {loan_rate:.1%}")

    # 4. Demographics
    avg_age = sub["age"].mean()
    gender_pct = sub["gender"].value_counts(normalize=True).round(3)
    client_type_pct = sub["client_type"].value_counts(normalize=True).round(3)
    avg_satisfaction = sub["satisfaction_score"].mean()
    print(f"  Avg age             : {avg_age:.1f}")
    print(f"  Gender split        : {gender_pct.to_dict()}")
    print(f"  Client type split   : {client_type_pct.to_dict()}")
    print(f"  Avg satisfaction    : {avg_satisfaction:.2f}")

    summary_rows.append({
        "cluster": c,
        "n_clients": n,
        "pct_of_total": round(n / len(df), 3),
        "pct_investment_purpose": purpose_pct.get("Investment", 0),
        "pct_home_purpose": purpose_pct.get("Home", 0),
        "top_country": top_countries.index[0],
        "top_country_pct": top_countries.iloc[0],
        "top_region": top_regions.index[0],
        "top_region_pct": top_regions.iloc[0],
        "loan_applied_rate": round(loan_rate, 3),
        "avg_age": round(avg_age, 1),
        "pct_female": gender_pct.get("F", 0),
        "pct_male": gender_pct.get("M", 0),
        "pct_company": client_type_pct.get("Company", 0),
        "pct_individual": client_type_pct.get("Individual", 0),
        "avg_satisfaction": round(avg_satisfaction, 2),
    })

print("\n" + "=" * 70)

# ---------------------------------------------------------------
# Save a single comparison table - the core Step 6 deliverable
# ---------------------------------------------------------------
summary = pd.DataFrame(summary_rows)
summary.to_csv("cluster_interpretation_summary.csv", index=False)
print("\nSaved -> cluster_interpretation_summary.csv")
print(summary.to_string(index=False))

# ---------------------------------------------------------------
# Also save full crosstabs per dimension for appendix-level detail
# ---------------------------------------------------------------
pd.crosstab(df["kmeans_cluster"], df["acquisition_purpose"], normalize="index").round(3).to_csv(
    "crosstab_investment_purpose.csv")
pd.crosstab(df["kmeans_cluster"], df["country"], normalize="index").round(3).to_csv(
    "crosstab_geography.csv")
pd.crosstab(df["kmeans_cluster"], df["loan_applied"], normalize="index").round(3).to_csv(
    "crosstab_loan_behavior.csv")
pd.crosstab(df["kmeans_cluster"], df["gender"], normalize="index").round(3).to_csv(
    "crosstab_demographics_gender.csv")
pd.crosstab(df["kmeans_cluster"], df["client_type"], normalize="index").round(3).to_csv(
    "crosstab_demographics_client_type.csv")

print("\nSaved detailed crosstabs: investment purpose, geography, loan behavior, demographics")
