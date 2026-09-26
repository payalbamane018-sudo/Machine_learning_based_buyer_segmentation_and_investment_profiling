"""
Step 4 - Clustering Model Selection (REBUILT on the enriched feature set)
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Two clustering approaches, as required:

  K-Means Clustering
    Advantages: efficient (scales well, O(n) per iteration), easy to
    interpret (each client belongs to exactly one cluster, centroids are
    directly readable).

  Hierarchical Clustering (Agglomerative, Ward linkage)
    Advantages: reveals nested cluster relationships via the dendrogram,
    helps validate K-Means results (agreement measured via Adjusted Rand
    Index - no need to pre-specify k to see the merge structure).

Feature set (financing, deal-size, tenure, demographic - matches the
synthetic financial fields generated in gen_synthetic_financials.py):
  loan_applied, total_investment, avg_purchase_price, properties_owned,
  buying_tenure_years, age, satisfaction_score, acquisition_purpose,
  client_type, gender

Recommended Buyer Segments (target business taxonomy):
  C1  Global Investors    - High income, investment purchases
  C2  First-Time Buyers   - Younger, loan dependent
  C3  Corporate Buyers    - Companies purchasing multiple units
  C4  Luxury Investors    - High satisfaction, large investments
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

IN_PATH = "clients_with_synthetic_financials.csv"
OUT_PATH = "clustered_clients.csv"
K = 4

# ---------------------------------------------------------------
# 1. Load data and build the clustering feature set
# ---------------------------------------------------------------
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

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
print(f"Clustering on {len(feature_cols)} features: {feature_cols}")

# ---------------------------------------------------------------
# 2. K-Means (k=4) - efficient, easy to interpret
# ---------------------------------------------------------------
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X)
sil = silhouette_score(X, df["kmeans_cluster"])
print(f"\nK-Means (k={K}) silhouette score: {sil:.4f}")
print(df["kmeans_cluster"].value_counts().sort_index())

# ---------------------------------------------------------------
# 3. Hierarchical clustering - validate + reveal nested structure
# ---------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
df["hierarchical_cluster"] = agg.fit_predict(X)

ari = adjusted_rand_score(df["kmeans_cluster"], df["hierarchical_cluster"])
print(f"Adjusted Rand Index (K-Means vs Hierarchical agreement): {ari:.4f}")
print("(closer to 1.0 = the two methods agree on cluster structure -> validates K-Means)")

# Dendrogram on a sample (full 2000-row linkage matrix is expensive to view)
sample_idx = pd.Series(range(len(X))).sample(n=min(200, len(X)), random_state=42).index
Z = linkage(X[sample_idx], method="ward")

plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (sample of 200 clients, truncated)")
plt.xlabel("Client cluster groupings")
plt.ylabel("Ward distance")
plt.tight_layout()
plt.savefig("hierarchical_dendrogram.png", dpi=150)
print("Saved -> hierarchical_dendrogram.png")

# ---------------------------------------------------------------
# 4. Profile each K-Means cluster on the raw (unscaled) fields
# ---------------------------------------------------------------
print("\n--- K-Means cluster profiles ---")
for c in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == c]
    print(f"\nCluster {c}  (n={len(sub)}, {len(sub)/len(df):.1%})")
    print(f"  Avg age:              {sub['age'].mean():.1f}")
    print(f"  Loan applied rate:    {sub['loan_applied'].mean():.1%}")
    print(f"  % Company:            {(sub['client_type']=='Company').mean():.1%}")
    print(f"  % Investment purpose: {(sub['acquisition_purpose']=='Investment').mean():.1%}")
    print(f"  Avg satisfaction:     {sub['satisfaction_score'].mean():.2f}")
    print(f"  Avg total_investment: ${sub['total_investment'].mean():,.0f}")
    print(f"  Avg properties owned: {sub['properties_owned'].mean():.2f}")

# ---------------------------------------------------------------
# 5. Map K-Means clusters onto the 4 target business segments
#    (this is exploratory - see final_segmentation_rulebased.py for
#    the production rule-based assignment that guarantees a clean fit)
# ---------------------------------------------------------------
print("\n--- Mapping clusters to target segments ---")
mapping_notes = []
for c in sorted(df["kmeans_cluster"].unique()):
    sub = df[df["kmeans_cluster"] == c]
    pct_company = (sub["client_type"] == "Company").mean()
    pct_investment = (sub["acquisition_purpose"] == "Investment").mean()
    avg_satisfaction = sub["satisfaction_score"].mean()
    avg_investment = sub["total_investment"].mean()

    if pct_company > 0.9:
        label = "Corporate Buyers (companies purchasing multiple units)"
    elif pct_investment > 0.9:
        label = "Global Investors (high income, investment purchases)"
    elif avg_satisfaction >= 4.0:
        label = "Luxury Investors (high satisfaction, large investments)"
    else:
        label = "First-Time Buyers (closest natural fit - see caveat below)"
    mapping_notes.append((c, label, len(sub)))
    print(f"  Cluster {c} -> {label}  (n={len(sub)})")

print("\nCAVEAT: K-Means naturally isolates Corporate Buyers and Global "
      "Investors cleanly (pure clusters). It does NOT naturally isolate a "
      "satisfaction-driven 'Luxury' cluster or an age/loan-driven "
      "'First-Time' cluster in this dataset, because satisfaction_score "
      "and loan_applied don't vary enough within the remaining clients to "
      "form their own natural cluster. The production pipeline "
      "(final_segmentation_rulebased.py) handles this with explicit rules "
      "layered on top of these K-Means results - see that script and "
      "Step 6 for the full methodology and validation.")

# ---------------------------------------------------------------
# 6. Save cluster assignments
# ---------------------------------------------------------------
df = df.drop(columns=["client_type_encoded", "acquisition_purpose_encoded",
                       "gender_encoded", "loan_applied_encoded"])
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
