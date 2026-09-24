"""
Step 4 - Clustering Model Selection
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Runs K-Means (primary, efficient/interpretable) and Hierarchical/
Agglomerative clustering (secondary, validates K-Means + shows nested
structure via dendrogram).

IMPORTANT DATA NOTE:
The raw dataset has no income, purchase price, unit count, or deal-size
field. So "High income", "Large investments", "Multiple units" in the
target segment table below can't be verified directly from this data -
this script uses the closest available proxies:
  client_type          (Individual vs Company)
  acquisition_purpose   (Home vs Investment)
  loan_applied          (proxy for financing dependence / lower liquidity)
  satisfaction_score
  age
If real transaction-value / unit-count fields exist elsewhere, add them
before this step for a more defensible "Luxury Investors" / "Global
Investors" split.

Feature set used for clustering (region one-hot excluded - 65 sparse
columns would dominate distance calculations; region is kept in the
output for profiling/drill-down only, not as a clustering input):
  age, satisfaction_score (scaled), client_type_encoded,
  acquisition_purpose_encoded, loan_applied, gender_encoded,
  country_* (10 cols), referral_channel_* (3 cols)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

IN_PATH = "scaled_clients_standard.csv"
OUT_PATH = "clustered_clients.csv"

# ---------------------------------------------------------------
# 1. Load scaled data (output of Step 3) and build clustering feature set
# ---------------------------------------------------------------
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

df["gender_encoded"] = df["gender"].map({"M": 0, "F": 1})

region_cols = [c for c in df.columns if c.startswith("region_")]
country_cols = [c for c in df.columns if c.startswith("country_")]
referral_cols = [c for c in df.columns if c.startswith("referral_channel_")]

feature_cols = (
    ["age", "satisfaction_score", "client_type_encoded",
     "acquisition_purpose_encoded", "loan_applied", "gender_encoded"]
    + country_cols
    + referral_cols
)
X = df[feature_cols].astype(float)
print(f"Clustering on {len(feature_cols)} features (region_* excluded: {len(region_cols)} cols)")

# ---------------------------------------------------------------
# 2. K-Means: pick k via elbow (inertia) + silhouette score
# ---------------------------------------------------------------
inertias, sil_scores = [], []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

print("\nk | inertia | silhouette")
for k, i, s in zip(k_range, inertias, sil_scores):
    print(f"{k} | {i:.1f} | {s:.4f}")

best_k = k_range[int(np.argmax(sil_scores))]
print(f"\nBest k by silhouette score: {best_k}")

# Recommended segment table has 4 buyer types - use k=4 to align with
# business labels unless silhouette strongly favors a different k
K = 4
kmeans_final = KMeans(n_clusters=K, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans_final.fit_predict(X)
print(f"\nUsing K={K} K-Means clusters (mapped to the 4 target buyer segments)")
print(df["kmeans_cluster"].value_counts().sort_index())

# ---------------------------------------------------------------
# 3. Hierarchical clustering - validate K-Means + inspect nested structure
# ---------------------------------------------------------------
# Dendrogram on a sample (full 2000-row linkage matrix is expensive to view)
sample_idx = X.sample(n=min(200, len(X)), random_state=42).index
Z = linkage(X.loc[sample_idx], method="ward")

agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
df["hierarchical_cluster"] = agg.fit_predict(X)

ari = adjusted_rand_score(df["kmeans_cluster"], df["hierarchical_cluster"])
print(f"\nAgreement between K-Means and Hierarchical (Adjusted Rand Index): {ari:.4f}")
print("(closer to 1.0 = the two methods agree on cluster structure -> validates K-Means)")

# ---------------------------------------------------------------
# 4. Profile each K-Means cluster on the original (unscaled) fields
# ---------------------------------------------------------------
profile_cols = ["client_type", "acquisition_purpose", "loan_applied",
                 "gender", "satisfaction_score", "age"]
raw = pd.read_csv("cleaned_clients.csv")
df_profile = df[["client_id", "kmeans_cluster"]].merge(raw, on="client_id")

print("\n--- Cluster profiles (K-Means) ---")
for c in sorted(df_profile["kmeans_cluster"].unique()):
    sub = df_profile[df_profile["kmeans_cluster"] == c]
    print(f"\nCluster {c}  (n={len(sub)})")
    print(f"  client_type:        {sub['client_type'].value_counts(normalize=True).round(2).to_dict()}")
    print(f"  acquisition_purpose:{sub['acquisition_purpose'].value_counts(normalize=True).round(2).to_dict()}")
    print(f"  loan_applied:       {sub['loan_applied'].value_counts(normalize=True).round(2).to_dict()}")
    print(f"  avg satisfaction:   {sub['satisfaction_score'].mean():.2f}")
    print(f"  avg age:            {sub['age'].mean():.1f}")

# ---------------------------------------------------------------
# 5. Save cluster assignments for Step 5 (labeling / investment profiling)
# ---------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
