"""
Step 5 - Optimal Cluster Selection (REBUILT on the enriched feature set)
Project: ML-based Buyer Segmentation and Investment Profiling (Parcl)

Evaluation methods:
  Elbow Method     -> plots inertia (within-cluster sum of squares) vs k;
                       optimal k is where the curve "bends" and additional
                       clusters stop reducing inertia much.
  Silhouette Score -> measures how well-separated clusters are (-1 to 1,
                       higher = better separated/more cohesive clusters).

Uses the same feature set as the rebuilt Step 4 (financing, deal-size,
tenure, demographic - including the synthetic financial fields).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IN_PATH = "clients_with_synthetic_financials.csv"
K_RANGE = range(2, 11)

# ---------------------------------------------------------------
# 1. Load data and rebuild the Step 4 feature set
# ---------------------------------------------------------------
df = pd.read_csv(IN_PATH)

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
print(f"Evaluating k = {list(K_RANGE)} on {len(feature_cols)} features, n={len(X)} rows")

# ---------------------------------------------------------------
# 2. Compute inertia (elbow) and silhouette score for each k
# ---------------------------------------------------------------
inertias, sil_scores = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

results = pd.DataFrame({
    "k": list(K_RANGE),
    "inertia": inertias,
    "silhouette_score": sil_scores
})
print("\n", results.to_string(index=False))

# ---------------------------------------------------------------
# 3. Elbow detection: largest drop in the rate of inertia decrease
#    (simple "knee" heuristic - biggest second-derivative change)
# ---------------------------------------------------------------
deltas = np.diff(inertias)
delta_of_deltas = np.diff(deltas)
elbow_k = list(K_RANGE)[int(np.argmax(delta_of_deltas)) + 1]
print(f"\nElbow method suggests k = {elbow_k}")

best_sil_k = results.loc[results["silhouette_score"].idxmax(), "k"]
print(f"Silhouette score suggests k = {int(best_sil_k)} "
      f"(score = {results['silhouette_score'].max():.4f})")

# ---------------------------------------------------------------
# 4. Plot both diagnostics
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(list(K_RANGE), inertias, marker="o")
axes[0].axvline(elbow_k, color="red", linestyle="--", alpha=0.6, label=f"elbow k={elbow_k}")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_title("Elbow Method")
axes[0].legend()

axes[1].plot(list(K_RANGE), sil_scores, marker="o", color="green")
axes[1].axvline(best_sil_k, color="red", linestyle="--", alpha=0.6, label=f"best k={int(best_sil_k)}")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("cluster_evaluation_elbow_silhouette.png", dpi=150)
print("\nSaved plot -> cluster_evaluation_elbow_silhouette.png")

results.to_csv("cluster_evaluation_results.csv", index=False)
print("Saved metrics -> cluster_evaluation_results.csv")

# ---------------------------------------------------------------
# 5. Reconcile with business requirement (4 named segments)
# ---------------------------------------------------------------
k4_sil = results.loc[results["k"] == 4, "silhouette_score"].values[0]
print(f"\nk=4 silhouette score: {k4_sil:.4f}")
print(f"Statistically 'best' k by silhouette: {int(best_sil_k)} (score {results['silhouette_score'].max():.4f})")
print("k=4 was still used in Step 4 to align with Parcl's 4-segment business "
      "taxonomy (Global Investors, First-Time Buyers, Corporate Buyers, "
      "Luxury Investors), with the gap vs the statistical optimum documented "
      "here for transparency.")
