"""
Streamlit dashboard - Buyer Segmentation and Investment Profiling (Parcl)

Reads the CSVs produced by the Steps 1-6 pipeline (expects them in the
same folder as this script, e.g. your GitHub repo root):
  cleaned_clients.csv
  clustered_clients.csv
  segments_approach1_datadriven.csv
  segments_approach2_rulebased.csv

Run with:
  streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Parcl Buyer Segmentation Dashboard", layout="wide")

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    cleaned = pd.read_csv("cleaned_clients.csv")
    clustered = pd.read_csv("clustered_clients.csv")[["client_id", "kmeans_cluster"]]
    approach1 = pd.read_csv("segments_approach1_datadriven.csv")[["client_id", "buyer_segment"]]
    approach2 = pd.read_csv("segments_approach2_rulebased.csv")[["client_id", "buyer_segment"]]

    df = cleaned.merge(clustered, on="client_id")
    df = df.merge(approach1.rename(columns={"buyer_segment": "segment_datadriven"}), on="client_id")
    df = df.merge(approach2.rename(columns={"buyer_segment": "segment_rulebased"}), on="client_id")
    return df

df = load_data()

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.title("Filters")

approach = st.sidebar.radio(
    "Segmentation approach",
    ["Data-Driven (K-Means)", "Rule-Based (Business Segments)"],
    help="Data-Driven uses the actual K-Means clusters (named by age/satisfaction). "
         "Rule-Based maps clients to Parcl's 4 target segment names using proxy rules.",
)
segment_col = "segment_datadriven" if approach == "Data-Driven (K-Means)" else "segment_rulebased"

all_segments = sorted(df[segment_col].unique())
selected_segments = st.sidebar.multiselect(
    "Segments to include", all_segments, default=all_segments
)

filtered = df[df[segment_col].isin(selected_segments)]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Note: the raw dataset has no income, deal-size, or unit-count field. "
    "Rule-Based segment labels are proxies, not statistically discovered patterns "
    "-- see README for full caveats."
)

st.title("Parcl Buyer Segmentation & Investment Profiling")
st.caption(f"{len(filtered):,} of {len(df):,} clients shown | Segmentation: {approach}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Buyer Segmentation Overview",
    "Investor Behavior Dashboard",
    "Geographic Buyer Analysis",
    "Segment Insights Panel",
])

# ---------------------------------------------------------------
# Tab 1 - Buyer Segmentation Overview
# ---------------------------------------------------------------
with tab1:
    st.subheader("Cluster / Segment Distribution")
    col1, col2 = st.columns(2)

    counts = filtered[segment_col].value_counts().reset_index()
    counts.columns = ["segment", "count"]

    with col1:
        fig_bar = px.bar(
            counts, x="segment", y="count", color="segment",
            title="Client Count by Segment", text="count"
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Clients")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            counts, names="segment", values="count",
            title="Segment Share of Total Buyers"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.dataframe(counts.assign(pct=(counts["count"] / counts["count"].sum() * 100).round(1)),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# Tab 2 - Investor Behavior Dashboard
# ---------------------------------------------------------------
with tab2:
    st.subheader("Investment Patterns by Segment")
    col1, col2 = st.columns(2)

    with col1:
        purpose_pct = (
            filtered.groupby(segment_col)["acquisition_purpose"]
            .value_counts(normalize=True).mul(100).rename("pct").reset_index()
        )
        fig_purpose = px.bar(
            purpose_pct, x=segment_col, y="pct", color="acquisition_purpose",
            barmode="group", title="Acquisition Purpose (%) by Segment"
        )
        fig_purpose.update_layout(xaxis_title="", yaxis_title="% of segment")
        st.plotly_chart(fig_purpose, use_container_width=True)

    with col2:
        loan_pct = (
            filtered.groupby(segment_col)["loan_applied"]
            .mean().mul(100).round(1).rename("loan_applied_pct").reset_index()
        )
        fig_loan = px.bar(
            loan_pct, x=segment_col, y="loan_applied_pct", color=segment_col,
            title="Loan Applied Rate (%) by Segment", text="loan_applied_pct"
        )
        fig_loan.update_layout(showlegend=False, xaxis_title="", yaxis_title="% loan applied")
        st.plotly_chart(fig_loan, use_container_width=True)

    st.subheader("Client Type Mix by Segment")
    type_pct = (
        filtered.groupby(segment_col)["client_type"]
        .value_counts(normalize=True).mul(100).rename("pct").reset_index()
    )
    fig_type = px.bar(
        type_pct, x=segment_col, y="pct", color="client_type",
        barmode="stack", title="Individual vs Company (%) by Segment"
    )
    fig_type.update_layout(xaxis_title="", yaxis_title="% of segment")
    st.plotly_chart(fig_type, use_container_width=True)

# ---------------------------------------------------------------
# Tab 3 - Geographic Buyer Analysis
# ---------------------------------------------------------------
with tab3:
    st.subheader("Where Each Segment's Buyers Are Located")

    country_counts = (
        filtered.groupby(["country", segment_col]).size().reset_index(name="count")
    )
    fig_map = px.choropleth(
        country_counts, locations="country", locationmode="country names",
        color="count", hover_name="country", facet_col=segment_col, facet_col_wrap=2,
        color_continuous_scale="Blues", title="Buyer Count by Country, per Segment"
    )
    fig_map.update_layout(height=700)
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top Regions by Segment")
    top_regions = (
        filtered.groupby([segment_col, "region"]).size().reset_index(name="count")
        .sort_values(["segment_datadriven" if segment_col == "segment_datadriven" else "segment_rulebased", "count"],
                     ascending=[True, False])
    )
    top_n = (
        top_regions.groupby(segment_col, group_keys=False)
        .apply(lambda g: g.nlargest(5, "count"))
    )
    fig_regions = px.bar(
        top_n, x="count", y="region", color=segment_col, orientation="h",
        title="Top 5 Regions per Segment", barmode="group"
    )
    st.plotly_chart(fig_regions, use_container_width=True)

# ---------------------------------------------------------------
# Tab 4 - Segment Insights Panel
# ---------------------------------------------------------------
with tab4:
    st.subheader("Descriptive Statistics per Segment")

    focus_segment = st.selectbox("Choose a segment to inspect", all_segments)
    sub = df[df[segment_col] == focus_segment]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clients", f"{len(sub):,}", f"{len(sub) / len(df):.1%} of total")
    m2.metric("Avg age", f"{sub['age'].mean():.1f}")
    m3.metric("Avg satisfaction", f"{sub['satisfaction_score'].mean():.2f} / 5")
    m4.metric("Loan applied rate", f"{sub['loan_applied'].mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Acquisition purpose**")
        st.dataframe(
            sub["acquisition_purpose"].value_counts(normalize=True).mul(100).round(1)
            .rename("pct").reset_index(), hide_index=True, use_container_width=True
        )
        st.markdown("**Client type**")
        st.dataframe(
            sub["client_type"].value_counts(normalize=True).mul(100).round(1)
            .rename("pct").reset_index(), hide_index=True, use_container_width=True
        )
    with col2:
        st.markdown("**Gender split**")
        st.dataframe(
            sub["gender"].value_counts(normalize=True).mul(100).round(1)
            .rename("pct").reset_index(), hide_index=True, use_container_width=True
        )
        st.markdown("**Referral channel**")
        st.dataframe(
            sub["referral_channel"].value_counts(normalize=True).mul(100).round(1)
            .rename("pct").reset_index(), hide_index=True, use_container_width=True
        )

    st.markdown("**Age distribution**")
    fig_age = px.histogram(sub, x="age", nbins=20, title=f"Age Distribution - {focus_segment}")
    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("**Full numeric summary**")
    st.dataframe(sub[["age", "satisfaction_score"]].describe().round(2), use_container_width=True)
