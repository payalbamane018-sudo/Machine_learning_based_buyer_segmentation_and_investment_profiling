import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path(__file__).parent / "FINAL_REAL_ESTATE_BUYER_SEGMENTATION.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.columns = [str(c).strip() for c in data.columns]

    # Remove accidental duplicate segment columns if both exist
    if "buyer_segment" in data.columns and "Buyer_Segment" in data.columns:
        data["Buyer_Segment"] = data["Buyer_Segment"].fillna(data["buyer_segment"])
        data = data.drop(columns=["buyer_segment"])

    # Normalize common binary fields for reliable filtering
    for col in ["loan_applied"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    if "age" not in data.columns and "date_of_birth" in data.columns:
        dob = pd.to_datetime(data["date_of_birth"], errors="coerce")
        today = pd.Timestamp.today()
        data["age"] = ((today - dob).dt.days / 365.25).round(1)

    return data

df = load_data()

st.title("🏠 Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Machine Learning Based Buyer Intelligence Dashboard | Parcl Real Estate Market Intelligence")

# ---------- Sidebar filters ----------
st.sidebar.header("🔎 Buyer Filters")

def options_for(col):
    if col not in df.columns:
        return ["All"]
    vals = df[col].dropna().astype(str).sort_values().unique().tolist()
    return ["All"] + vals

country = st.sidebar.selectbox("Country", options_for("country"))
region = st.sidebar.selectbox("Region", options_for("region"))
purpose = st.sidebar.selectbox("Acquisition Purpose", options_for("acquisition_purpose"))
client_type = st.sidebar.selectbox("Client Type", options_for("client_type"))

filtered = df.copy()

if country != "All" and "country" in filtered.columns:
    filtered = filtered[filtered["country"].astype(str) == country]
if region != "All" and "region" in filtered.columns:
    filtered = filtered[filtered["region"].astype(str) == region]
if purpose != "All" and "acquisition_purpose" in filtered.columns:
    filtered = filtered[filtered["acquisition_purpose"].astype(str) == purpose]
if client_type != "All" and "client_type" in filtered.columns:
    filtered = filtered[filtered["client_type"].astype(str) == client_type]

st.sidebar.markdown("---")
st.sidebar.info(f"Showing **{len(filtered):,}** buyers after applying filters.")

# ---------- Helper functions ----------
def pct(series, value):
    if len(series) == 0:
        return 0.0
    return float((series.astype(str).str.lower() == str(value).lower()).mean() * 100)

def find_cluster_col(data):
    for c in ["Cluster", "cluster", "cluster_label", "KMeans_Cluster"]:
        if c in data.columns:
            return c
    return None

def find_segment_col(data):
    for c in ["Buyer_Segment", "buyer_segment", "Segment", "segment"]:
        if c in data.columns:
            return c
    return None

cluster_col = find_cluster_col(df)
segment_col = find_segment_col(df)

# ---------- KPI row ----------
total_buyers = len(filtered)

if "satisfaction_score" in filtered.columns and total_buyers:
    avg_satisfaction = filtered["satisfaction_score"].mean()
else:
    avg_satisfaction = np.nan

investment_pct = (
    pct(filtered["acquisition_purpose"], "Investment")
    if "acquisition_purpose" in filtered.columns else 0
)

loan_pct = (
    pct(filtered["loan_applied"], "Yes")
    if "loan_applied" in filtered.columns else 0
)

segment_count = (
    filtered[segment_col].nunique()
    if segment_col and total_buyers else 0
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Total Buyers", f"{total_buyers:,}")
c2.metric("🎯 Buyer Segments", f"{segment_count:,}")
c3.metric("📈 Investment Buyers", f"{investment_pct:.1f}%")
c4.metric("💳 Loan Applied", f"{loan_pct:.1f}%")
c5.metric("⭐ Avg Satisfaction", f"{avg_satisfaction:.2f}" if not np.isnan(avg_satisfaction) else "N/A")

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segmentation Overview",
    "💰 Investor Behavior",
    "🌍 Geographic Analysis",
    "💡 Segment Insights"
])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("Buyer Segmentation Overview")

    if segment_col:
        seg_counts = (
            filtered[segment_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        seg_counts.columns = ["Buyer Segment", "Buyers"]
        seg_counts["Percentage"] = seg_counts["Buyers"] / max(len(filtered), 1) * 100

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                seg_counts,
                names="Buyer Segment",
                values="Buyers",
                hole=0.45,
                title="Buyer Segment Distribution"
            )
            fig.update_layout(legend_title_text="Segment")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                seg_counts,
                x="Buyer Segment",
                y="Buyers",
                text="Percentage",
                title="Buyers by Segment"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            seg_counts.style.format({"Percentage": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("A buyer segment column was not found in the CSV.")

    st.subheader("Cluster Distribution")

    if cluster_col:
        cluster_counts = (
            filtered[cluster_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        cluster_counts.columns = ["Cluster", "Buyers"]

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Buyers",
            text="Buyers",
            title="Machine Learning Cluster Distribution"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2 ----------
with tab2:
    st.subheader("Investor Behavior Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        if "acquisition_purpose" in filtered.columns:
            purpose_counts = filtered["acquisition_purpose"].fillna("Unknown").value_counts().reset_index()
            purpose_counts.columns = ["Purpose", "Buyers"]
            fig = px.bar(
                purpose_counts,
                x="Purpose",
                y="Buyers",
                color="Purpose",
                title="Acquisition Purpose"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "loan_applied" in filtered.columns:
            loan_counts = filtered["loan_applied"].fillna("Unknown").value_counts().reset_index()
            loan_counts.columns = ["Loan Applied", "Buyers"]
            fig = px.pie(
                loan_counts,
                names="Loan Applied",
                values="Buyers",
                hole=0.45,
                title="Financing / Loan Behavior"
            )
            st.plotly_chart(fig, use_container_width=True)

    if segment_col:
        st.subheader("Investment & Financing Behavior by Segment")

        group_cols = [segment_col]
        agg = filtered.groupby(group_cols, dropna=False).size().reset_index(name="Buyers")

        if "acquisition_purpose" in filtered.columns:
            inv = (
                filtered.assign(
                    _investment=filtered["acquisition_purpose"].astype(str).str.lower().eq("investment")
                )
                .groupby(segment_col)["_investment"]
                .mean()
                .mul(100)
                .reset_index(name="Investment %")
            )
            agg = agg.merge(inv, on=segment_col, how="left")

        if "loan_applied" in filtered.columns:
            loan = (
                filtered.assign(
                    _loan=filtered["loan_applied"].astype(str).str.lower().eq("yes")
                )
                .groupby(segment_col)["_loan"]
                .mean()
                .mul(100)
                .reset_index(name="Loan Applied %")
            )
            agg = agg.merge(loan, on=segment_col, how="left")

        if "satisfaction_score" in filtered.columns:
            sat = filtered.groupby(segment_col)["satisfaction_score"].mean().reset_index(name="Avg Satisfaction")
            agg = agg.merge(sat, on=segment_col, how="left")

        st.dataframe(
            agg.style.format({
                "Investment %": "{:.2f}%",
                "Loan Applied %": "{:.2f}%",
                "Avg Satisfaction": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    col3, col4 = st.columns(2)

    with col3:
        if "client_type" in filtered.columns:
            client_counts = filtered["client_type"].fillna("Unknown").value_counts().reset_index()
            client_counts.columns = ["Client Type", "Buyers"]
            fig = px.bar(
                client_counts,
                x="Client Type",
                y="Buyers",
                color="Client Type",
                title="Individual vs Corporate Buyers"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "satisfaction_score" in filtered.columns:
            fig = px.histogram(
                filtered,
                x="satisfaction_score",
                nbins=10,
                title="Customer Satisfaction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Geographic Buyer Analysis")

    if "country" in filtered.columns:
        country_counts = (
            filtered["country"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        country_counts.columns = ["Country", "Buyers"]

        fig = px.bar(
            country_counts.head(20),
            x="Country",
            y="Buyers",
            title="Top Countries by Buyer Count"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    if "region" in filtered.columns:
        region_counts = (
            filtered["region"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        region_counts.columns = ["Region", "Buyers"]

        fig = px.bar(
            region_counts,
            x="Region",
            y="Buyers",
            title="Buyer Distribution by Region"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    if "country" in filtered.columns and segment_col:
        geo = (
            filtered.groupby(["country", segment_col], dropna=False)
            .size()
            .reset_index(name="Buyers")
        )
        geo[segment_col] = geo[segment_col].fillna("Unknown").astype(str)

        st.subheader("Buyer Segments by Country")
        fig = px.bar(
            geo,
            x="country",
            y="Buyers",
            color=segment_col,
            barmode="stack",
            title="Segment Composition Across Countries"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Use a map when country names can be recognized by Plotly
        st.subheader("Geographic Buyer Map")
        map_df = filtered.copy()
        map_df["country"] = map_df["country"].astype(str).str.strip()
        map_counts = map_df.groupby("country").size().reset_index(name="Buyers")

        try:
            fig = px.choropleth(
                map_counts,
                locations="country",
                locationmode="country names",
                color="Buyers",
                hover_name="country",
                title="Buyer Distribution by Country"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("The map could not be rendered from the available country labels. The country charts above remain available.")

# ---------- Tab 4 ----------
with tab4:
    st.subheader("Segment Insights Panel")

    if segment_col and len(filtered):
        segments = filtered[segment_col].dropna().astype(str).unique().tolist()
        selected_segment = st.selectbox("Select Buyer Segment", sorted(segments))

        seg_df = filtered[filtered[segment_col].astype(str) == selected_segment]

        n = len(seg_df)
        share = n / max(len(filtered), 1) * 100
        avg_age = seg_df["age"].mean() if "age" in seg_df.columns else np.nan
        avg_sat = seg_df["satisfaction_score"].mean() if "satisfaction_score" in seg_df.columns else np.nan
        inv = pct(seg_df["acquisition_purpose"], "Investment") if "acquisition_purpose" in seg_df.columns else 0
        loan = pct(seg_df["loan_applied"], "Yes") if "loan_applied" in seg_df.columns else 0
        individual = pct(seg_df["client_type"], "Individual") if "client_type" in seg_df.columns else 0

        a, b, c, d, e = st.columns(5)
        a.metric("Segment Buyers", f"{n:,}")
        b.metric("Segment Share", f"{share:.1f}%")
        c.metric("Average Age", f"{avg_age:.1f}" if not np.isnan(avg_age) else "N/A")
        d.metric("Investment Buyers", f"{inv:.1f}%")
        e.metric("Loan Applied", f"{loan:.1f}%")

        st.markdown("### 📌 Segment Profile")

        profile = pd.DataFrame({
            "Metric": [
                "Buyer Count",
                "Average Age",
                "Average Satisfaction",
                "Investment Purpose",
                "Loan Applied = Yes",
                "Individual Buyers"
            ],
            "Value": [
                f"{n:,}",
                f"{avg_age:.2f}" if not np.isnan(avg_age) else "N/A",
                f"{avg_sat:.2f}" if not np.isnan(avg_sat) else "N/A",
                f"{inv:.2f}%",
                f"{loan:.2f}%",
                f"{individual:.2f}%"
            ]
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

        st.markdown("### 💼 Business Interpretation")

        # Keep interpretation data-driven rather than inventing unsupported segment labels.
        characteristics = []

        if inv >= 50:
            characteristics.append("strong investment-oriented behavior")
        elif inv >= 25:
            characteristics.append("a meaningful investment component")
        else:
            characteristics.append("primarily personal-use purchasing")

        if loan >= 50:
            characteristics.append("high financing dependence")
        elif loan >= 30:
            characteristics.append("moderate financing usage")
        else:
            characteristics.append("relatively low financing usage")

        if individual >= 80:
            characteristics.append("a predominantly individual-buyer profile")
        else:
            characteristics.append("a relatively stronger corporate-buyer presence")

        text = (
            f"**{selected_segment}** contains **{n:,} buyers ({share:.1f}% of the filtered population)**. "
            f"The segment shows {', '.join(characteristics)}. "
        )

        if avg_sat == avg_sat:
            if avg_sat >= 4:
                text += "Customer satisfaction is relatively high, supporting retention and referral-focused strategies."
            elif avg_sat >= 3:
                text += "Customer satisfaction is moderate, so targeted service improvements may help strengthen loyalty."
            else:
                text += "Customer satisfaction is relatively low, suggesting an opportunity for customer-experience improvements."

        st.info(text)

        st.markdown("### 🎯 Recommended Business Actions")
        st.markdown(
            "- Personalize property recommendations using acquisition purpose and buyer characteristics.\n"
            "- Use financing behavior to identify buyers who may benefit from suitable loan/property affordability options.\n"
            "- Use geographic concentration to focus local marketing campaigns.\n"
            "- Use satisfaction results to prioritize retention and service-improvement initiatives.\n"
            "- Compare segments continuously as new buyer data becomes available."
        )
    else:
        st.warning("No segment data is available for the selected filters.")

# ---------- Raw data ----------
with st.expander("📄 View Filtered Buyer Data"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built for the Machine Learning Based Buyer Segmentation and Investment Profiling project.")
