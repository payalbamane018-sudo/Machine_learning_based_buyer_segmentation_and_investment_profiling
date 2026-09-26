"""
Step 6 - Cluster Interpretation (REBUILT on the 4 business segments)
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Analyzes each of the 4 named segments (Global Investors, First-Time
Buyers, Corporate Buyers, Luxury Investors - see
final_segmentation_rulebased.py) along four dimensions:
  1. Investment purpose      -> acquisition_purpose (Home vs Investment)
  2. Geographic distribution -> country, region (top values per segment)
  3. Loan behavior           -> loan_applied rate
  4. Customer demographics   -> age, gender, client_type

Also profiles the financial fields (total_investment, avg_purchase_price,
properties_owned) that actually define this segmentation, since those -
not geography or raw demographics - are what separate these 4 segments.
"""

import pandas as pd

IN_PATH = "FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv"
SEGMENT_ORDER = ["Global Investors", "First-Time Buyers",
                  "Corporate Buyers", "Luxury Investors"]

df = pd.read_csv(IN_PATH)
summary_rows = []

print("=" * 70)
for seg in SEGMENT_ORDER:
    sub = df[df["segment"] == seg]
    n = len(sub)
    print(f"\nSEGMENT: {seg}  (n={n}, {n/len(df):.1%} of clients)")

    # 1. Investment purpose
    purpose_pct = sub["acquisition_purpose"].value_counts(normalize=True).round(3)
    print(f"  Investment purpose   : {purpose_pct.to_dict()}")

    # 2. Geographic distribution (top 3 countries, top 3 regions)
    top_countries = sub["country"].value_counts(normalize=True).head(3).round(3)
    top_regions = sub["region"].value_counts(normalize=True).head(3).round(3)
    print(f"  Top countries        : {top_countries.to_dict()}")
    print(f"  Top regions          : {top_regions.to_dict()}")

    # 3. Loan behavior
    loan_rate = sub["loan_applied"].mean()
    print(f"  Loan applied rate    : {loan_rate:.1%}")

    # 4. Demographics
    avg_age = sub["age"].mean()
    gender_pct = sub["gender"].value_counts(normalize=True).round(3)
    client_type_pct = sub["client_type"].value_counts(normalize=True).round(3)
    avg_satisfaction = sub["satisfaction_score"].mean()
    print(f"  Avg age              : {avg_age:.1f}")
    print(f"  Gender split         : {gender_pct.to_dict()}")
    print(f"  Client type split    : {client_type_pct.to_dict()}")
    print(f"  Avg satisfaction     : {avg_satisfaction:.2f}")

    # 5. What actually defines this segment (financial profile)
    avg_investment = sub["total_investment"].mean()
    avg_price = sub["avg_purchase_price"].mean()
    avg_properties = sub["properties_owned"].mean()
    avg_tenure = sub["buying_tenure_years"].mean()
    print(f"  Avg total investment : ${avg_investment:,.0f}")
    print(f"  Avg purchase price   : ${avg_price:,.0f}")
    print(f"  Avg properties owned : {avg_properties:.2f}")
    print(f"  Avg buying tenure    : {avg_tenure:.1f} years")

    summary_rows.append({
        "segment": seg,
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
        "avg_total_investment": round(avg_investment, 0),
        "avg_purchase_price": round(avg_price, 0),
        "avg_properties_owned": round(avg_properties, 2),
        "avg_buying_tenure_years": round(avg_tenure, 1),
    })

print("\n" + "=" * 70)

# ---------------------------------------------------------------
# Save the core Step 6 deliverable
# ---------------------------------------------------------------
summary = pd.DataFrame(summary_rows)
summary.to_csv("cluster_interpretation_summary.csv", index=False)
print("\nSaved -> cluster_interpretation_summary.csv")
print(summary.to_string(index=False))

# ---------------------------------------------------------------
# Full crosstabs per dimension for appendix-level detail
# ---------------------------------------------------------------
pd.crosstab(df["segment"], df["acquisition_purpose"], normalize="index").round(3).to_csv(
    "crosstab_investment_purpose.csv")
pd.crosstab(df["segment"], df["country"], normalize="index").round(3).to_csv(
    "crosstab_geography.csv")
pd.crosstab(df["segment"], df["loan_applied"], normalize="index").round(3).to_csv(
    "crosstab_loan_behavior.csv")
pd.crosstab(df["segment"], df["gender"], normalize="index").round(3).to_csv(
    "crosstab_demographics_gender.csv")
pd.crosstab(df["segment"], df["client_type"], normalize="index").round(3).to_csv(
    "crosstab_demographics_client_type.csv")

print("\nSaved detailed crosstabs: investment purpose, geography, loan behavior, demographics")

# ---------------------------------------------------------------
# Interpretation note
# ---------------------------------------------------------------
print("\n--- Interpretation ---")
print("Investment purpose, geography, loan behavior and raw demographics "
      "(age/gender) are the dimensions the project brief asks to profile, "
      "but for THIS segmentation they mostly just reflect how each segment "
      "was defined (e.g. Global Investors = 100% Investment purpose by "
      "construction) rather than revealing new patterns. The genuinely "
      "differentiating dimensions are the financial ones - total "
      "investment, purchase price, properties owned, and buying tenure - "
      "which is why they're profiled here too, alongside the required four.")
