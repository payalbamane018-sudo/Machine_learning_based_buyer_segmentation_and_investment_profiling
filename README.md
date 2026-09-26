# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

An end-to-end ML pipeline and interactive Streamlit dashboard that segments Parcl's real estate buyers into four business-defined segments, revealing investment behavior, financing patterns, and geographic distribution that traditional analytics can't detect.

## Background

Real estate companies serve highly diverse buyer types — individual home buyers, institutional investors, international buyers, high-net-worth investors, and first-time buyers — but Parcl currently has no data-driven way to distinguish between them. This leads to inefficient marketing spend, generic property recommendations, and poor investor targeting.

## Problem Statement

Parcl lacks a data-driven understanding of:
- Different types of property buyers
- Investment motivations across demographics
- Geographic differences in investment behavior
- Customer financing patterns

## Target Buyer Segments

| Segment | Definition |
|---|---|
| **Global Investors** | High income, investment purchases |
| **First-Time Buyers** | Younger, loan dependent |
| **Corporate Buyers** | Companies purchasing multiple units |
| **Luxury Investors** | High satisfaction, large investments |

## Data Disclosure

The source dataset (`clients.csv`) has no income, transaction value, unit count, or purchase-history field. To enable investment profiling, `total_investment`, `avg_purchase_price`, `properties_owned`, and `buying_tenure_years` were **generated synthetically** using documented rules tied to real fields (client type, acquisition purpose, age, satisfaction score) — see `gen_synthetic_financials.py`. These are illustrative figures, not real client financials, and are disclosed in the dashboard sidebar and the research report.

## Pipeline

| Step | Description | Script |
|---|---|---|
| 1 | Data Cleaning — mixed date formats, derived age, normalized labels, duplicate check | `step1_data_cleaning.py` |
| 2 | Feature Encoding — Label Encoding (binary) + One-Hot Encoding (nominal) | `step2_feature_encoding.py` |
| 3 | Feature Scaling — StandardScaler and MinMaxScaler | `step3_feature_scaling.py` |
| — | Synthetic Financial Fields — total investment, deal size, properties owned, buying tenure | `gen_synthetic_financials.py` |
| 4 | Clustering — K-Means (k=4) validated against Hierarchical (Ward), on the enriched feature set | `step4_clustering.py` |
| 5 | Optimal Cluster Selection — Elbow Method + Silhouette Score, k=2–10 | `step5_optimal_cluster_selection.py` |
| 6 | Cluster Interpretation — profiled by investment purpose, geography, loan behavior, demographics | `step6_cluster_interpretation.py` |
| — | Final Segment Assignment — rule-based mapping to the 4 required business segments | `final_segmentation_rulebased.py` |
| — | Supplementary Visuals — correlation heatmap, EDA distributions, standalone PCA/silhouette/value-share plots | `generate_missing_outputs.py` |

### How segments are actually assigned

K-Means clustering (k=4) on financing, deal-size, tenure and demographic features naturally and cleanly isolated two of the four segments with no rules needed:
- **Corporate Buyers** — 100% Company `client_type`
- **Global Investors** — 100% Investment `acquisition_purpose`

The other two required explicit rules on top, because `satisfaction_score` and `loan_applied` don't vary enough among the remaining clients to form natural clusters on their own:
- **Luxury Investors** — remaining Home-purpose individuals with `total_investment` in the top quartile AND `satisfaction_score` ≥ 4
- **First-Time Buyers** — everyone else (includes an explicit younger + loan-dependent subgroup, plus a residual catch-all)

This trade-off is documented transparently in `final_segmentation_rulebased.py`, `clustering_summary.txt`, and the research report — it is not hidden.

## Key Findings

- Elbow Method confirms k=4 on the enriched feature set; Silhouette Score favors a simpler k=2 (0.271 vs 0.210 at k=4) — k=4 was kept to align with Parcl's business taxonomy
- K-Means vs Hierarchical agreement (Adjusted Rand Index): 0.80 — strong validation
- Geography and raw demographics barely differ between segments (USA/California dominates all four, 76–79%); **financing behavior, deal size, portfolio depth, and buying tenure are what actually separate the segments**
- Luxury Investors stands out with a notably lower loan rate (17.4% vs 38–42% elsewhere), consistent with self-financed, high-satisfaction purchases
- First-Time Buyers is the weakest-fitting segment: only a minority genuinely matches "younger, loan-dependent" — the rest is a residual catch-all, flagged directly in the dashboard's Key Finding panel

## Streamlit Dashboard (`app.py`)

| Requirement | Tab |
|---|---|
| Buyer Segmentation Overview — cluster distribution | Donut chart, PCA scatter, segment size vs. value contribution |
| Investor Behavior Dashboard — investment patterns by segment | Investment/purchase-price boxplots, loan rate, purpose mix, deal-size scatter |
| Geographic Buyer Analysis — segments mapped by region | Country bar chart, investment-by-country, regional table, treemap |
| Segment Insights Panel — descriptive statistics per segment | Stats table, key finding, segment playbook, client-level explorer |

**Sidebar filters:** Country (multi-select), Region (multi-select, narrows to selected countries), Segment (multi-select), plus a live "Clients in view" counter. All four tabs respond to these filters together.

## Repository Structure

```
├── clients.csv                                # raw input data
├── step1_data_cleaning.py
├── cleaned_clients.csv
├── step2_feature_encoding.py
├── encoded_clients.csv
├── label_encoding_mappings.csv
├── step3_feature_scaling.py
├── scaled_clients_standard.csv
├── scaled_clients_minmax.csv
├── gen_synthetic_financials.py
├── clients_with_synthetic_financials.csv
├── step4_clustering.py
├── clustered_clients.csv
├── hierarchical_dendrogram.png
├── step5_optimal_cluster_selection.py
├── cluster_evaluation_results.csv
├── cluster_evaluation_elbow_silhouette.png
├── final_segmentation_rulebased.py
├── FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv    # ← used by app.py
├── step6_cluster_interpretation.py
├── cluster_interpretation_summary.csv
├── cluster_profile.csv
├── crosstab_investment_purpose.csv
├── crosstab_geography.csv
├── crosstab_loan_behavior.csv
├── crosstab_demographics_gender.csv
├── crosstab_demographics_client_type.csv
├── generate_missing_outputs.py
├── correlation_heatmap.png
├── eda_distributions.png
├── pca_clusters.png
├── segment_value_share.png
├── silhouette_plot.png
├── clustering_summary.txt
├── Buyer_Segmentation_Research_Report.docx
├── app.py                                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to Run the Pipeline

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn

python step1_data_cleaning.py
python step2_feature_encoding.py
python step3_feature_scaling.py
python gen_synthetic_financials.py
python step4_clustering.py
python step5_optimal_cluster_selection.py
python final_segmentation_rulebased.py
python step6_cluster_interpretation.py
python generate_missing_outputs.py
```

Each script reads the output of the previous step and writes its own output file(s) to the working directory.

## How to Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

`app.py` reads `FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv` from the same folder — make sure both are in the repo root together (never update one without the other, since the app depends on that file's exact column names).

For Streamlit Community Cloud: point the deploy form's **Main file path** to `app.py` on the `main` branch, and reboot the app after any update to keep dependencies and data in sync.

## Recommendations for Parcl

- Collect real transaction-level data (purchase price, deal size, unit count, financing terms) to replace the synthetic financial fields and validate these segments against ground truth
- Prioritize relationship management and exclusive listings for Luxury Investors (highest satisfaction, lowest loan dependency)
- Target Corporate Buyers with bulk-purchase agreements given their high average portfolio depth (7.33 properties)
- Revisit the First-Time Buyers segment once real data is available, since only a minority currently matches its intended definition
