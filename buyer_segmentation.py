import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load CSV
df = pd.read_csv("buyer_segmentation_sample.csv")

# 2. Features used for BEHAVIOURAL buyer segmentation
features = [
    "age",
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "satisfaction_score",
    "purchase_count",
    "total_spend",
    "referral_channel"
]

X = df[features].copy()

# 3. Separate numeric and categorical columns
numeric_features = [
    "age",
    "satisfaction_score",
    "purchase_count",
    "total_spend"
]

categorical_features = [
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 5. K-Means model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10))
    ]
)

# 6. Train model and create clusters
df["kmeans_cluster"] = model.fit_predict(X)

# 7. Evaluate clustering
X_transformed = model.named_steps["preprocessor"].transform(X)
score = silhouette_score(X_transformed, df["kmeans_cluster"])

print("Silhouette Score:", round(score, 3))
print("\nCluster Counts:")
print(df["kmeans_cluster"].value_counts().sort_index())

# 8. Create simple business labels AFTER examining cluster profiles
cluster_means = df.groupby("kmeans_cluster")[[
    "age", "satisfaction_score", "purchase_count", "total_spend"
]].mean().round(2)

print("\nCluster Profile:")
print(cluster_means)

# Example labels for this sample data.
# For your real dataset, verify these labels from the cluster profile first.
label_map = {
    0: "Buyer Segment 1",
    1: "Buyer Segment 2",
    2: "Buyer Segment 3",
    3: "Buyer Segment 4"
}

df["buyer_segment"] = df["kmeans_cluster"].map(label_map)

# 9. Save results
df.to_csv("buyer_segmentation_results.csv", index=False)

print("\nResults saved as buyer_segmentation_results.csv")
