# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

An unsupervised machine learning pipeline that segments Parcl's real estate buyers into behavioral clusters, revealing patterns in investment motivation, geography, financing, and demographics that traditional analytics can't detect.

## Background

Real estate companies serve highly diverse buyer types — individual home buyers, institutional investors, international buyers, high-net-worth investors, and first-time buyers — but Parcl currently has no data-driven way to distinguish between them. This leads to inefficient marketing spend, generic property recommendations, and poor investor targeting.

## Problem Statement

Parcl lacks a data-driven understanding of:
- Different types of property buyers
- Investment motivations across demographics
- Geographic differences in investment behavior
- Customer financing patterns

## Approach

This project applies clustering algorithms (K-Means and Hierarchical Clustering) to client data to uncover hidden buyer segments, then evaluates whether those data-driven segments can be mapped to Parcl's four target business segments: **Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors**.

## Pipeline

| Step | Description | Script |
|---|---|---|
| 1 | Data Cleaning — parsed mixed date formats, derived age, normalized categorical labels, removed duplicates | `step1_data_cleaning.py` |
| 2 | Feature Encoding — Label Encoding (binary fields) + One-Hot Encoding (nominal fields) | `step2_feature_encoding.py` |
| 3 | Feature Scaling — StandardScaler and MinMaxScaler on numeric fields | `step3_feature_scaling.py` |
| 4 | Clustering — K-Means (primary) + Hierarchical Clustering (validation via Adjusted Rand Index) | `step4_clustering.py` |
| 5 | Optimal Cluster Selection — Elbow Method + Silhouette Score across k=2–10 | `step5_optimal_cluster_selection.py` |
| 6 | Cluster Interpretation — profiled clusters by investment purpose, geography, loan behavior, demographics | `step6_cluster_interpretation.py` |
| — | Two Segmentation Approaches — data-driven cluster labels vs. rule-based mapping to business segment names | `step5_two_segmentation_approaches.py` |

## Tech Stack

- Python 3
- pandas, numpy — data processing
- scikit-learn — encoding, scaling, K-Means, Agglomerative Clustering, evaluation metrics
- scipy — hierarchical linkage
- matplotlib — elbow/silhouette visualization

## Dataset

`clients.csv` — 2,000 client records with: client type, demographics (age, gender), country/region, acquisition purpose, satisfaction score, loan application status, and referral channel.

**Known limitation:** the dataset has no income, transaction value, unit count, or purchase-history field. Segments like "high income" or "large investments" are approximated using proxy fields (client type, acquisition purpose, loan status, satisfaction) rather than direct financial data — see Findings below.

## Key Findings

- Statistical evaluation (elbow method, silhouette score) indicates the data naturally supports **k=2–3 clusters**, not the 4 segments in Parcl's target taxonomy. k=4 was used anyway to align with business requirements, at some cost to cluster cohesion (silhouette 0.162 at k=4 vs. 0.184 at k=2).
- K-Means and Hierarchical Clustering show moderate agreement (Adjusted Rand Index ≈ 0.44).
- Clusters are driven almost entirely by **age and satisfaction score**. Investment purpose (~27–34% Investment in every cluster), geography (USA/California dominant everywhere), and loan behavior (33–40% everywhere) barely vary between clusters.
- A rule-based mapping to the four named business segments (Corporate Buyers, Luxury Investors, Global Investors, First-Time Buyers) is possible as a proxy, but is not a statistically discovered pattern — it should be treated as a hypothesis pending richer transaction data.

## Recommendations

To produce segments that genuinely reflect investment behavior (rather than age/satisfaction), future data collection should include:
- Transaction/purchase value
- Number of units purchased
- Income bracket or credit profile
- Repeat-purchase / transaction history (to validate "first-time" vs. repeat buyer status)

## Repository Structure

```
├── clients.csv                              # raw input data
├── step1_data_cleaning.py
├── cleaned_clients.csv
├── step2_feature_encoding.py
├── encoded_clients.csv
├── label_encoding_mappings.csv
├── step3_feature_scaling.py
├── scaled_clients_standard.csv
├── scaled_clients_minmax.csv
├── step4_clustering.py
├── clustered_clients.csv
├── step5_optimal_cluster_selection.py
├── cluster_evaluation_results.csv
├── cluster_evaluation_elbow_silhouette.png
├── step5_two_segmentation_approaches.py
├── segments_approach1_datadriven.csv
├── segments_approach2_rulebased.csv
├── step6_cluster_interpretation.py
├── cluster_interpretation_summary.csv
├── crosstab_investment_purpose.csv
├── crosstab_geography.csv
├── crosstab_loan_behavior.csv
├── crosstab_demographics_gender.csv
├── crosstab_demographics_client_type.csv
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn scipy matplotlib

python step1_data_cleaning.py
python step2_feature_encoding.py
python step3_feature_scaling.py
python step4_clustering.py
python step5_optimal_cluster_selection.py
python step5_two_segmentation_approaches.py
python step6_cluster_interpretation.py
```

Each script reads the output of the previous step and writes its own output CSV(s) to the working directory.

## Author

[Your name]
