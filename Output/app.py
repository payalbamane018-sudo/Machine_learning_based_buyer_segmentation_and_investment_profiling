"""
Parcl Buyer Segmentation & Investment Profiling — Streamlit Dashboard
Run with: streamlit run app.py
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Parcl Buyer Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clients_clustered.csv")
    profile = pd.read_csv("outputs/cluster_profile.csv")
    return df, profile

df, profile = load_data()

SEGMENT_COLORS = {
    "High-Net-Worth Investors": "#7C3AED",
    "Mainstream Buyers": "#F97316",
    "Premium / Global Investors": "#2563EB",
    "First-Time Buyers": "#EC4899",
}

# ------------------------------------------------------------------
# SIDEBAR — USER CONTROLS (filter by country, region, acquisition purpose, client type)
# ------------------------------------------------------------------
st.sidebar.title("🏢 Parcl Buyer Intelligence")
st.sidebar.caption("ML-based Buyer Segmentation & Investment Profiling")
st.sidebar.divider()
st.sidebar.header("Filters")

countries = st.sidebar.multiselect("Country", sorted(df["country"].unique()), default=[])
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=[])
purposes = st.sidebar.multiselect("Acquisition Purpose", sorted(df["acquisition_purpose"].unique()), default=[])
client_types = st.sidebar.multiselect("Client Type", sorted(df["client_type"].unique()), default=[])
segments = st.sidebar.multiselect("Buyer Segment", sorted(df["segment"].unique()), default=[])

fdf = df.copy()
if countries:
    fdf = fdf[fdf["country"].isin(countries)]
if regions:
    fdf = fdf[fdf["region"].isin(regions)]
if purposes:
    fdf = fdf[fdf["acquisition_purpose"].isin(purposes)]
if client_types:
    fdf = fdf[fdf["client_type"].isin(client_types)]
if segments:
    fdf = fdf[fdf["segment"].isin(segments)]

st.sidebar.divider()
st.sidebar.metric("Clients in view", f"{len(fdf):,} / {len(df):,}")

if fdf.empty:
    st.warning("No clients match the current filters. Adjust filters in the sidebar.")
    st.stop()

# ------------------------------------------------------------------
# HEADER KPIs
# ------------------------------------------------------------------
st.title("Buyer Segmentation & Investment Profiling")
st.caption("K-Means clustering (k=4) over financing, deal-size, tenure and demographic features · validated against Hierarchical clustering")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients", f"{len(fdf):,}")
k2.metric("Total Investment", f"${fdf['total_investment'].sum()/1e6:,.1f}M")
k3.metric("Avg Deal Size", f"${fdf['avg_purchase_price'].mean():,.0f}")
k4.metric("Avg Satisfaction", f"{fdf['satisfaction_score'].mean():.2f} / 5")
k5.metric("Loan-Financed", f"{fdf['loan_applied_flag'].mean()*100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segmentation Overview",
    "💰 Investor Behavior",
    "🌍 Geographic Analysis",
    "🔍 Segment Insights",
])

# ------------------------------------------------------------------
# TAB 1 — Buyer Segmentation Overview
# ------------------------------------------------------------------
with tab1:
    st.subheader("Cluster Distribution")
    c1, c2 = st.columns([1, 1.4])

    with c1:
        dist = fdf["segment"].value_counts().reset_index()
        dist.columns = ["segment", "count"]
        fig = px.pie(dist, names="segment", values="count", hole=0.5,
                     color="segment", color_discrete_map=SEGMENT_COLORS)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pca_fig = px.scatter(
            fdf, x="pca1", y="pca2", color="segment",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=["client_id", "total_investment", "property_count"],
            title="Client Segments in PCA Space (2D projection of the clustering features)",
            labels={"pca1": "Principal Component 1", "pca2": "Principal Component 2"},
        )
        pca_fig.update_traces(marker=dict(size=6, opacity=0.7))
        pca_fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(pca_fig, use_container_width=True)

    st.subheader("Segment Sizes vs. Value Contribution")
    seg_val = fdf.groupby("segment").agg(
        clients=("client_id", "count"),
        total_investment=("total_investment", "sum"),
    ).reset_index()
    seg_val["share_of_clients_%"] = (seg_val["clients"] / seg_val["clients"].sum() * 100).round(1)
    seg_val["share_of_investment_%"] = (seg_val["total_investment"] / seg_val["total_investment"].sum() * 100).round(1)

    fig2 = go.Figure()
    fig2.add_bar(name="% of Clients", x=seg_val["segment"], y=seg_val["share_of_clients_%"], marker_color="#94A3B8")
    fig2.add_bar(name="% of Total Investment", x=seg_val["segment"], y=seg_val["share_of_investment_%"], marker_color="#2563EB")
    fig2.update_layout(barmode="group", margin=dict(t=10, b=10, l=10, r=10),
                        yaxis_title="Share (%)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("A segment punching above its client-count share in investment value is a high-priority relationship-management target.")

# ------------------------------------------------------------------
# TAB 2 — Investor Behavior Dashboard
# ------------------------------------------------------------------
with tab2:
    st.subheader("Investment Patterns by Segment")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(fdf, x="segment", y="total_investment", color="segment",
                     color_discrete_map=SEGMENT_COLORS, points=False,
                     title="Total Investment per Client")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(fdf, x="segment", y="avg_purchase_price", color="segment",
                     color_discrete_map=SEGMENT_COLORS, points=False,
                     title="Average Purchase Price per Deal")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        loan_by_seg = fdf.groupby("segment")["loan_applied_flag"].mean().reset_index()
        loan_by_seg["loan_applied_flag"] *= 100
        fig = px.bar(loan_by_seg, x="segment", y="loan_applied_flag", color="segment",
                     color_discrete_map=SEGMENT_COLORS, title="Financing (Loan) Dependency by Segment",
                     labels={"loan_applied_flag": "% who used a loan"})
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        purpose_seg = fdf.groupby(["segment", "acquisition_purpose"]).size().reset_index(name="count")
        fig = px.bar(purpose_seg, x="segment", y="count", color="acquisition_purpose",
                     barmode="stack", title="Acquisition Purpose Mix by Segment")
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Deal Size vs. Property Portfolio Size")
    fig = px.scatter(fdf, x="property_count", y="total_investment", color="segment",
                      color_discrete_map=SEGMENT_COLORS, size="avg_purchase_price",
                      hover_data=["client_id", "age", "referral_channel"],
                      labels={"property_count": "Number of Properties Owned",
                              "total_investment": "Total Investment ($)"})
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Referral Channel Effectiveness by Segment")
    ref = fdf.groupby(["segment", "referral_channel"]).size().reset_index(name="count")
    fig = px.bar(ref, x="referral_channel", y="count", color="segment",
                 color_discrete_map=SEGMENT_COLORS, barmode="group")
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# TAB 3 — Geographic Buyer Analysis
# ------------------------------------------------------------------
with tab3:
    st.subheader("Buyer Segments by Region")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        country_seg = fdf.groupby(["country", "segment"]).size().reset_index(name="count")
        fig = px.bar(country_seg, x="country", y="count", color="segment",
                     color_discrete_map=SEGMENT_COLORS, barmode="stack",
                     title="Segment Composition by Country")
        fig.update_layout(margin=dict(t=40, b=10), xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        country_inv = fdf.groupby("country")["total_investment"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(country_inv, x="total_investment", y="country", orientation="h",
                     title="Total Investment by Country", color_discrete_sequence=["#2563EB"])
        fig.update_layout(margin=dict(t=40, b=10), xaxis_title="Total Investment ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Detail")
    region_country = fdf.groupby(["country", "region"]).agg(
        clients=("client_id", "count"),
        total_investment=("total_investment", "sum"),
        avg_satisfaction=("satisfaction_score", "mean"),
    ).reset_index().sort_values("total_investment", ascending=False)
    region_country["total_investment"] = region_country["total_investment"].map(lambda x: f"${x:,.0f}")
    region_country["avg_satisfaction"] = region_country["avg_satisfaction"].round(2)
    st.dataframe(region_country, use_container_width=True, hide_index=True)

    st.subheader("Regional Treemap")
    fig = px.treemap(fdf, path=["country", "region", "segment"], values="total_investment",
                      color="segment", color_discrete_map=SEGMENT_COLORS)
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# TAB 4 — Segment Insights Panel
# ------------------------------------------------------------------
with tab4:
    st.subheader("Descriptive Statistics per Segment")

    display_cols = {
        "n_clients": "Clients",
        "avg_age": "Avg Age",
        "pct_investment_purpose": "% Investment Purpose",
        "pct_corporate": "% Corporate",
        "pct_loan": "% Loan Financed",
        "avg_satisfaction": "Avg Satisfaction",
        "avg_property_count": "Avg Properties Owned",
        "avg_total_investment": "Avg Total Investment ($)",
        "avg_purchase_price": "Avg Purchase Price ($)",
        "avg_floor_area": "Avg Floor Area (sqft)",
        "avg_purchase_span_days": "Avg Buying-Window (days)",
    }
    prof_display = profile.set_index("segment_name")[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(
        prof_display.style.format({
            "Avg Total Investment ($)": "${:,.0f}",
            "Avg Purchase Price ($)": "${:,.0f}",
            "Avg Age": "{:.1f}",
            "% Investment Purpose": "{:.1f}%",
            "% Corporate": "{:.1f}%",
            "% Loan Financed": "{:.1f}%",
            "Avg Satisfaction": "{:.2f}",
            "Avg Properties Owned": "{:.2f}",
            "Avg Floor Area (sqft)": "{:.0f}",
            "Avg Buying-Window (days)": "{:.1f}",
        }),
        use_container_width=True,
    )

    st.info(
        "**Key finding:** corporate-client share (~5%) and country mix are nearly identical "
        "across all four segments. Demographics do not drive this segmentation — "
        "**financing behavior, deal size, portfolio depth, and buying tenure do.**"
    )

    st.subheader("Segment Playbook")
    playbooks = {
        "High-Net-Worth Investors": "Smallest group (~4% of clients) but the single richest portfolio per head — oldest, highest total spend, most repeat purchases, rarely finance. **Action:** white-glove relationship management, off-market/priority inventory access, no loan-product push needed.",
        "Premium / Global Investors": "Buy fewer units but the priciest, largest-footprint ones (offices skew here). **Action:** lead with premium/office listings and investment-grade ROI narratives; cross-sell portfolio diversification.",
        "Mainstream Buyers": "The largest segment by far, average deal size, and the **lowest satisfaction score** of any segment. **Action:** priority for CX investment — this group is the biggest volume driver and the most at churn/dissatisfaction risk.",
        "First-Time Buyers": "Youngest, most loan-dependent, and their entire (small) portfolio was purchased in a single ~6-day window — true new entrants. **Action:** financing partnerships, first-time-buyer incentives, and nurture campaigns to convert them into repeat buyers.",
    }
    for seg, text in playbooks.items():
        with st.expander(f"🎯 {seg}"):
            st.markdown(text)

    st.subheader("Client-Level Explorer")
    st.dataframe(
        fdf[["client_id", "first_name", "last_name", "segment", "country", "region",
             "client_type", "acquisition_purpose", "age", "property_count",
             "total_investment", "satisfaction_score", "loan_applied"]].sort_values(
            "total_investment", ascending=False),
        use_container_width=True, hide_index=True,
    )

st.divider()
st.caption("Parcl Co. Limited × Unified Mentor — Machine Learning based Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence")
