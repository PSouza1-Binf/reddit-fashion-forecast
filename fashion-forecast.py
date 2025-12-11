# fashion-forecast.py
# Clean, production-ready Streamlit app using fashion_model.pkl

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Reddit Fashion Forecast",
    layout="wide",
)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("fashion_model.pkl", "rb") as f:
        return pickle.load(f)

art = load_artifacts()

clf = art["clf"]
reg = art["reg"]
surge_threshold = art["surge_threshold"]
FEATURE_COLS = art["feature_cols"]
latest_features = art["latest_features"]        # DataFrame-like, already safe_X processed
latest_df = art["latest_df"].copy()             # microtopic-level latest week data

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

all_brands = sorted(latest_df["brand"].unique())
brand = st.sidebar.selectbox("Brand", ["(All Brands)"] + all_brands)

# item list depends on brand
if brand and brand != "(All Brands)":
    brand_items = sorted(latest_df[latest_df["brand"] == brand]["item"].unique())
else:
    brand_items = sorted(latest_df["item"].unique())

item = st.sidebar.selectbox("Item", ["(All Items)"] + brand_items)

# Apply filters
filtered = latest_df.copy()

if brand != "(All Brands)":
    filtered = filtered[filtered["brand"] == brand]

if item != "(All Items)":
    filtered = filtered[filtered["item"] == item]

# ---------------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------------
filtered = filtered.reset_index(drop=True)
filtered["surge_prob"] = clf.predict_proba(latest_features.iloc[filtered.index])[:, 1]
filtered["pred_next_engagement"] = reg.predict(latest_features.iloc[filtered.index])

filtered["weighted_surge"] = (
    filtered["surge_prob"] * np.log1p(filtered["engagement_sum"])
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Historical Trends", "Forecast", "Surge Analysis", "Explore Raw Data"]
)

# ---------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.title("Fashion Forecast Overview")

    st.write("""
    This tool forecasts microtopic-level trends in Reddit fashion communities,
    including next-week engagement predictions and surge probability alerts.
    """)

    colA, colB = st.columns(2)
    colA.metric("Microtopics (filtered)", f"{len(filtered):,}")
    colB.metric("Surge Threshold", f"{surge_threshold:.3f}")

    st.subheader("Top microtopics (filtered)")
    st.dataframe(
        filtered[[
            "microtopic", "brand", "item",
            "engagement_sum", "sentiment_mean",
            "surge_prob", "pred_next_engagement"
        ]].sort_values("pred_next_engagement", ascending=False).head(20)
    )

# ---------------------------------------------------------
# TAB 2 — HISTORICAL TRENDS (Optional Future Expansion)
# ---------------------------------------------------------
with tab2:
    st.title("Historical Trends")
    st.info("Historical charts can be added here once time-series data is packaged into the PKL.")

# ---------------------------------------------------------
# TAB 3 — FORECAST
# ---------------------------------------------------------
with tab3:
    st.title("Forecast")

    st.write("Predicted engagement for next week:")

    top_pred = filtered.sort_values("pred_next_engagement", ascending=False).head(15)

    fig = px.bar(
        top_pred,
        x="microtopic",
        y="pred_next_engagement",
        color="pred_next_engagement",
        title="Next Week Engagement Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(top_pred)

# ---------------------------------------------------------
# TAB 4 — SURGE ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.title("Surge Analysis")

    st.subheader("Surge Alerts (surge_prob ≥ 0.90)")
    alerts = filtered[filtered["surge_prob"] >= 0.90].sort_values("surge_prob", ascending=False)

    if alerts.empty:
        st.success("No major surge warnings this week.")
    else:
        st.error("Surge alerts detected!")
        st.dataframe(alerts)

    st.subheader("Top Surging Microtopics (Weighted)")
    top_surge = filtered.sort_values("weighted_surge", ascending=False).head(20)

    fig2 = px.bar(
        top_surge,
        x="microtopic",
        y="weighted_surge",
        color="surge_prob",
        title="Top Surging Microtopics"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(top_surge)

# ---------------------------------------------------------
# TAB 5 — RAW DATA
# ---------------------------------------------------------
with tab5:
    st.title("Explore Raw Latest-Week Microtopics")
    st.dataframe(filtered)

