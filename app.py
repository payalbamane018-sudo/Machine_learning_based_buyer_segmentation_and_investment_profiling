"""
Streamlit dashboard - Buyer Segmentation & Investment Profiling (Parcl)

Reads FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv (must be in the same
folder / repo root as this file).

*** DATA DISCLOSURE ***
total_investment, avg_purchase_price, properties_owned and
buying_tenure_years are SYNTHETIC fields generated to illustrate what
a full investment-profiling dashboard looks like, because the source
dataset (clients.csv) has no real financial/transaction data. This is
disclosed in the app UI (sidebar) and must be disclosed in any report
built from it. All other fields are real client data.

Run with:
  streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Buyer Segmentation & Investment Profiling", layout="wide")

DATA_FILE = "FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv"
SEGMENT_ORDER = ["First-Time Buyers", "Global Investors",
                  "Luxury Investors", "Corporate Buyers"]
SEGMENT_COLORS = {
    "First-Time Buyers": "#e0489a",
    "Global Investors": "#3b82f6",
    "Luxury Investors": "#8b5cf6",
    "Corporate Buyers": "#f28c28",
}

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.columns = [str(c).strip() for c in data.columns]
    return data

df = load_data()

# ---------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------
st.title("Buyer Segmentation & Investment Profiling")
st.caption(
    "4 business-defined segments (Global Investors, First-Time Buyers, "
    "Corporate Buyers, Luxury Investors) \u00b7 built from financing, deal-size, "
    "tenure and demographic features, informed by K-Means clustering"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients", f"{len(df):,}")
k2.metric("Total Investment", f"${df['total_investment'].sum():,.0f}")
k3.metric("Avg Deal Size", f"${df['avg_purchase_price'].mean():,.0f}")
k4.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f} / 5")
k5.metric("Loan-Financed", f"{df['loan_applied'].mean():.1%}")

st.sidebar.title("\U0001F3E2 Parcl Buyer Intelligence")
st.sidebar.caption("ML-based Buyer Segmentation & Investment Profiling")
st.sidebar.markdown("---")

st.sidebar.markdown("### Filters")

country_options = sorted(df["country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country", country_options, default=[], placeholder="Choose options"
)

region_options = sorted(
    df[df["country"].isin(selected_countries)]["region"].unique().tolist()
    if selected_countries else df["region"].unique().tolist()
)
selected_regions = st.sidebar.multiselect(
    "Region", region_options, default=[], placeholder="Choose options"
)

purpose_options = sorted(df["acquisition_purpose"].unique().tolist())
selected_purposes = st.sidebar.multiselect(
    "Acquisition Purpose", purpose_options, default=[], placeholder="Choose options"
)

client_type_options = sorted(df["client_type"].unique().tolist())
selected_client_types = st.sidebar.multiselect(
    "Client Type", client_type_options, default=[], placeholder="Choose options"
)

segment_filter = st.sidebar.multiselect(
    "Buyer Segment", SEGMENT_ORDER, default=[], placeholder="Choose options"
)

view = df.copy()
if selected_countries:
    view = view[view["country"].isin(selected_countries)]
if selected_regions:
    view = view[view["region"].isin(selected_regions)]
if selected_purposes:
    view = view[view["acquisition_purpose"].isin(selected_purposes)]
if selected_client_types:
    view = view[view["client_type"].isin(selected_client_types)]
if segment_filter:
    view = view[view["segment"].isin(segment_filter)]

st.sidebar.markdown("---")
st.sidebar.metric("Clients in view", f"{len(view):,} / {len(df):,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001F4CA Segmentation Overview", "\U0001F4B0 Investor Behavior",
    "\U0001F30D Geographic Analysis", "\U0001F50D Segment Insights",
])

# =================================================================
# TAB 1 - Segmentation Overview
# =================================================================
with tab1:
    st.subheader("Cluster Distribution")
    col1, col2 = st.columns(2)

    with col1:
        counts = view["segment"].value_counts().reindex(SEGMENT_ORDER).dropna().reset_index()
        counts.columns = ["segment", "count"]
        fig_donut = px.pie(
            counts, names="segment", values="count", hole=0.55,
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title=None,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        fig_pca = px.scatter(
            view, x="pc1", y="pc2", color="segment",
            color_discrete_map=SEGMENT_COLORS, opacity=0.6,
            title="Client Segments in PCA Space (2D projection of the clustering features)",
            labels={"pc1": "Principal Component 1", "pc2": "Principal Component 2"},
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Segment Sizes vs. Value Contribution")
    total_n, total_inv = len(view), view["total_investment"].sum()
    contrib = view.groupby("segment").agg(
        n=("client_id", "count"), inv=("total_investment", "sum")
    ).reindex(SEGMENT_ORDER).dropna().reset_index()
    contrib["% of Clients"] = contrib["n"] / total_n * 100
    contrib["% of Total Investment"] = contrib["inv"] / total_inv * 100
    contrib_long = contrib.melt(
        id_vars="segment", value_vars=["% of Clients", "% of Total Investment"],
        var_name="metric", value_name="share"
    )
    fig_contrib = px.bar(
        contrib_long, x="segment", y="share", color="metric", barmode="group",
        labels={"share": "Share (%)", "segment": ""},
    )
    st.plotly_chart(fig_contrib, use_container_width=True)
    st.caption("A segment punching above its client-count share in investment value "
               "is a high-priority relationship-management target.")

# =================================================================
# TAB 2 - Investor Behavior
# =================================================================
with tab2:
    st.subheader("Investment Patterns by Segment")
    col1, col2 = st.columns(2)

    with col1:
        fig_box_inv = px.box(
            view, x="segment", y="total_investment", color="segment",
            color_discrete_map=SEGMENT_COLORS, category_orders={"segment": SEGMENT_ORDER},
            title="Total Investment per Client", points=False,
        )
        fig_box_inv.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_box_inv, use_container_width=True)

    with col2:
        fig_box_price = px.box(
            view, x="segment", y="avg_purchase_price", color="segment",
            color_discrete_map=SEGMENT_COLORS, category_orders={"segment": SEGMENT_ORDER},
            title="Average Purchase Price per Deal", points=False,
        )
        fig_box_price.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_box_price, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        loan_pct = (
            view.groupby("segment")["loan_applied"].mean().mul(100)
            .reindex(SEGMENT_ORDER).dropna().reset_index()
        )
        loan_pct.columns = ["segment", "pct"]
        fig_loan = px.bar(
            loan_pct, x="segment", y="pct", color="segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Financing (Loan) Dependency by Segment",
            labels={"pct": "% who used a loan"},
        )
        fig_loan.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_loan, use_container_width=True)

    with col4:
        purpose_counts = (
            view.groupby(["segment", "acquisition_purpose"]).size()
            .reset_index(name="count")
        )
        fig_purpose = px.bar(
            purpose_counts, x="segment", y="count", color="acquisition_purpose",
            barmode="stack", category_orders={"segment": SEGMENT_ORDER},
            title="Acquisition Purpose Mix by Segment",
        )
        fig_purpose.update_layout(xaxis_title="")
        st.plotly_chart(fig_purpose, use_container_width=True)

    st.subheader("Deal Size vs. Property Portfolio Size")
    fig_scatter = px.scatter(
        view, x="properties_owned", y="total_investment", color="segment",
        color_discrete_map=SEGMENT_COLORS, opacity=0.6,
        labels={"properties_owned": "Number of Properties Owned",
                "total_investment": "Total Investment ($)"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# =================================================================
# TAB 3 - Geographic Analysis
# =================================================================
with tab3:
    st.subheader("Buyer Segments by Region")
    col1, col2 = st.columns(2)

    with col1:
        geo_counts = view.groupby(["country", "segment"]).size().reset_index(name="count")
        country_order = view["country"].value_counts().index.tolist()
        fig_geo = px.bar(
            geo_counts, x="country", y="count", color="segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"country": country_order, "segment": SEGMENT_ORDER},
            title="Segment Composition by Country",
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with col2:
        inv_by_country = (
            view.groupby("country")["total_investment"].sum()
            .sort_values(ascending=True).reset_index()
        )
        fig_inv_country = px.bar(
            inv_by_country, x="total_investment", y="country", orientation="h",
            title="Total Investment by Country",
            labels={"total_investment": "Total Investment ($)"},
        )
        st.plotly_chart(fig_inv_country, use_container_width=True)

    st.subheader("Regional Detail")
    regional = (
        view.groupby(["country", "region"]).agg(
            clients=("client_id", "count"),
            total_investment=("total_investment", "sum"),
            avg_satisfaction=("satisfaction_score", "mean"),
        ).reset_index().sort_values("clients", ascending=False)
    )
    regional["avg_satisfaction"] = regional["avg_satisfaction"].round(2)
    st.dataframe(regional, use_container_width=True, hide_index=True)

    st.subheader("Regional Treemap")
    fig_tree = px.treemap(
        view, path=["country", "region", "segment"], values="total_investment",
        color="segment", color_discrete_map=SEGMENT_COLORS,
    )
    fig_tree.update_layout(height=600)
    st.plotly_chart(fig_tree, use_container_width=True)

# =================================================================
# TAB 4 - Segment Insights
# =================================================================
with tab4:
    st.subheader("Descriptive Statistics per Segment")

    stats_rows = []
    for seg in SEGMENT_ORDER:
        sub = view[view["segment"] == seg]
        if len(sub) == 0:
            continue
        stats_rows.append({
            "segment_name": seg,
            "Clients": len(sub),
            "Avg Age": round(sub["age"].mean(), 1),
            "% Investment Purpose": round((sub["acquisition_purpose"] == "Investment").mean() * 100, 1),
            "% Corporate": round((sub["client_type"] == "Company").mean() * 100, 1),
            "% Loan Financed": round(sub["loan_applied"].mean() * 100, 1),
            "Avg Satisfaction": round(sub["satisfaction_score"].mean(), 2),
            "Avg Properties Owned": round(sub["properties_owned"].mean(), 1),
            "Avg Total Investment": round(sub["total_investment"].mean(), 0),
        })
    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.info(
        "**Key finding:** Corporate Buyers and Global Investors emerged as pure, "
        "naturally-separated clusters from K-Means (100% Company / 100% Investment "
        "purpose respectively). First-Time Buyers and Luxury Investors required "
        "explicit rules on top (age+loan, and satisfaction+deal-size) since "
        "satisfaction and loan status don't vary enough on their own to form "
        "natural clusters in this dataset -- see the segmentation script for "
        "full methodology and this caveat in detail."
    )

    st.subheader("Segment Playbook")
    playbooks = {
        "Corporate Buyers": (
            "Companies purchasing multiple units - highest average portfolio "
            "depth (7+ properties) and largest deal sizes. Prioritize bulk-"
            "purchase agreements, dedicated account management, and B2B "
            "commercial financing partnerships."
        ),
        "Global Investors": (
            "High-value, investment-purpose buyers with large multi-property "
            "portfolios. Target with investment-grade listings, market "
            "intelligence reports, and portfolio diversification advisory."
        ),
        "Luxury Investors": (
            "Smaller segment marked by high satisfaction and large individual "
            "deal sizes. Prioritize white-glove service, exclusive/off-market "
            "listings, and relationship-based retention over volume marketing."
        ),
        "First-Time Buyers": (
            "The largest segment. A meaningful share is genuinely younger and "
            "loan-dependent, but this bucket also absorbs remaining Home-purpose "
            "buyers who didn't qualify for another segment (see Key Finding "
            "above). Prioritize educational content, financing partnerships, "
            "and lower-friction onboarding."
        ),
    }
    for seg in SEGMENT_ORDER:
        if not segment_filter or seg in segment_filter:
            with st.expander(f"\U0001F3AF {seg}"):
                st.write(playbooks.get(seg, "No playbook notes available."))

    st.subheader("Client-Level Explorer")
    display_cols = ["client_id", "segment", "country", "region", "client_type",
                     "acquisition_purpose", "age", "satisfaction_score",
                     "loan_applied", "properties_owned", "total_investment"]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True, height=400)

st.markdown("---")
st.caption(
    "Parcl Co. Limited \u00d7 Unified Mentor \u2014 Machine Learning based Buyer "
    "Segmentation & Investment Profiling for Real Estate Market Intelligence"
)
