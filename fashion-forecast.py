# fashion-forecast.py

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

latest_features = art["latest_features"]
latest_df = art["latest_df"].copy()

ts_agg = art.get("ts_agg")
ts_feat = art.get("ts_feat")

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")
st.sidebar.markdown(
    """
    <h1 style='text-align:center; font-size:26px; margin-bottom:0;'>
        👗 Reddit Fashion Forecast
    </h1>
    <p style='text-align:center; font-size:14px; color:gray; margin-top:0;'>
        Weekly Trend Insights & Engagement Predictions
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

all_brands = sorted(ts_agg["brand"].dropna().unique())
brand = st.sidebar.selectbox("Brand", ["(All Brands)"] + all_brands)

# Items depend on brand
if brand != "(All Brands)":
    items_list = sorted(ts_agg[ts_agg["brand"] == brand]["item"].dropna().unique())
else:
    items_list = sorted(ts_agg["item"].dropna().unique())

item = st.sidebar.selectbox("Item", ["(All Items)"] + items_list)

# ---------------------------------------------------------
# UNIVERSAL FILTER FUNCTION
# ---------------------------------------------------------
def apply_filters(df):
    out = df.copy()
    if brand != "(All Brands)":
        out = out[out["brand"] == brand]
    if item != "(All Items)":
        out = out[out["item"] == item]
    return out

filtered_latest = apply_filters(latest_df)
filtered_agg   = apply_filters(ts_agg)
filtered_ts    = apply_filters(ts_feat)

# ---------------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------------
if not filtered_latest.empty:

    features_lookup = latest_features.copy()
    features_lookup["microtopic"] = latest_df["microtopic"].values

    merged = filtered_latest.merge(
        features_lookup,
        on="microtopic",
        how="left",
        suffixes=("", "_feat")
    )

    X = merged[FEATURE_COLS].astype(float)

    filtered_latest["surge_prob"] = clf.predict_proba(X)[:, 1]
    filtered_latest["pred_next_engagement"] = reg.predict(X)

    filtered_latest["weighted_surge"] = (
        filtered_latest["surge_prob"] * np.log1p(filtered_latest["engagement_sum"])
    )

# Stop if empty
if filtered_latest.empty:
    st.warning("No microtopics match this brand + item selection.")
    st.stop()

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
    st.subheader("Top Microtopics (Visual Overview)")

    overview_chart_df = (
        filtered_latest.sort_values("pred_next_engagement", ascending=False)
        .head(15)
    )

    fig_overview = px.bar(
        overview_chart_df,
        x="microtopic",
        y="pred_next_engagement",
        color="surge_prob",
        labels={
            "microtopic": "Microtopic",
            "pred_next_engagement": "Predicted Engagement",
            "surge_prob": "Surge Probability"
    },
    title="Top Microtopics by Predicted Engagement",
    )

    st.plotly_chart(fig_overview, use_container_width=True)

    colA, colB = st.columns(2)
    colA.metric("Microtopics (filtered)", f"{len(filtered_latest):,}")
    colB.metric("Surge Threshold", f"{surge_threshold:.3f}")

    # Top 10 Items
    st.subheader("Top 10 Items for Next Week")

    top_items = (
        filtered_latest
        .dropna(subset=["item"])
        .groupby(["brand", "item"], as_index=False)
        .agg(
            total_pred_next_engagement=("pred_next_engagement", "sum"),
            avg_surge_prob=("surge_prob", "mean"),
            current_engagement=("engagement_sum", "sum"),
        )
        .sort_values("total_pred_next_engagement", ascending=False)
        .head(10)
    )

    top_items = top_items.rename(columns={
        "brand": "Brand",
        "item": "Item",
        "current_engagement": "Current Engagement (Sum)",
        "total_pred_next_engagement": "Next Week Engagement (Predicted Sum)",
        "avg_surge_prob": "Avg Surge Probability",
    })

    st.dataframe(top_items, use_container_width=True)

    overview_cols = [
        "microtopic", "brand", "item",
        "engagement_sum", "sentiment_mean",
        "surge_prob", "pred_next_engagement"
    ]

    st.dataframe(
        filtered_latest[overview_cols]
        .sort_values("pred_next_engagement", ascending=False)
        .head(20)
    )

# ---------------------------------------------------------
# TAB 2 — HISTORICAL TRENDS
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Historical Trends")

    if filtered_agg.empty:
        st.info("No time-series data available for this selection.")
    else:

        filtered_agg["week_start"] = pd.to_datetime(filtered_agg["week_start"])

        hist = filtered_agg.groupby("week_start", as_index=False).agg(
            engagement=("engagement_sum", "sum"),
            posts=("posts", "sum"),
            sentiment=("sentiment_mean", "mean"),
        )

        fig = px.line(
            hist,
            x="week_start",
            y="engagement",
            title="Engagement Over Time",
            labels={"week_start": "Week", "engagement": "Engagement"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(hist[["week_start", "engagement", "posts", "sentiment"]])

# ---------------------------------------------------------
# TAB 3 — FORECAST
# ---------------------------------------------------------
with tab3:
    st.title("Forecast")

    st.write("Predicted engagement for next week:")

    top_pred = (
        filtered_latest.sort_values("pred_next_engagement", ascending=False)
        .head(15)
    )

    # -------------------------------------------------
    # 6-WEEK FORECAST (REGRESSION BASED)
    # -------------------------------------------------
    if not filtered_agg.empty:

        brand_hist = (
            filtered_agg.groupby("week_start", as_index=False)
            .agg(engagement_sum=("engagement_sum", "sum"))
            .sort_values("week_start")
        )

        if not brand_hist.empty:

            brand_hist["week_start"] = pd.to_datetime(brand_hist["week_start"])

            st.subheader("📈 6-Week Engagement Forecast (Regression-Based)")

            brand_hist["t"] = range(len(brand_hist))
            slope, intercept = np.polyfit(
                brand_hist["t"], brand_hist["engagement_sum"], 1
            )

            next_week_pred = float(filtered_latest["pred_next_engagement"].sum())

            horizon = 6
            t_future = np.arange(len(brand_hist), len(brand_hist) + horizon)
            trend_future = intercept + slope * t_future

            trend_future += (next_week_pred - trend_future[0])

            future_dates = [
                brand_hist["week_start"].max() + pd.Timedelta(weeks=i)
                for i in range(1, horizon + 1)
            ]

            fc_df = pd.DataFrame({
                "week_start": list(brand_hist["week_start"]) + future_dates,
                "engagement": list(brand_hist["engagement_sum"]) + list(trend_future),
                "type": ["history"] * len(brand_hist) + ["forecast"] * horizon
            })

            brand_label = brand if brand != "(All Brands)" else "All Brands"
            item_label = item if item != "(All Items)" else "All Items"

            fig_fc = px.line(
                fc_df,
                x="week_start",
                y="engagement",
                color="type",
                markers=True,
                title=f"Brand-Level Forecast — {brand_label} ({item_label})",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

    # Microtopic bar chart
    fig = px.bar(
        top_pred,
        x="microtopic",
        y="pred_next_engagement",
        color="pred_next_engagement",
        title="Next Week Engagement Forecast (Microtopic-Level)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(
        top_pred[
            [
                "microtopic",
                "brand",
                "item",
                "engagement_sum",
                "sentiment_mean",
                "surge_prob",
                "pred_next_engagement",
            ]
        ]
    )

# ---------------------------------------------------------
# TAB 4 — SURGE ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.title("Surge Analysis")

    st.subheader("Surge Alerts (surge_prob ≥ 0.50)")

    alerts = filtered_latest[
        filtered_latest["surge_prob"] >= 0.50
    ].sort_values("surge_prob", ascending=False)

    if alerts.empty:
        st.success("No major surge warnings this week.")
    else:
        st.error("Surge alerts detected!")

        alert_cols = [
            "microtopic", "brand", "item",
            "surge_prob", "engagement_sum"
        ]
        st.dataframe(alerts[alert_cols])

    st.subheader("Top Surging Microtopics (Weighted)")

    top_surge = filtered_latest.sort_values("weighted_surge", ascending=False).head(20)

    fig2 = px.bar(
        top_surge,
        x="microtopic",
        y="weighted_surge",
        color="surge_prob",
        title="Top Surging Microtopics",
    )
    st.plotly_chart(fig2, use_container_width=True)

    surge_cols = [
        "microtopic", "brand", "item",
        "weighted_surge", "surge_prob"
    ]
    st.dataframe(top_surge[surge_cols])

# ---------------------------------------------------------
# TAB 5 — RAW DATA
# ---------------------------------------------------------
with tab5:
    st.title("Explore Raw Latest-Week Microtopics")
    st.dataframe(filtered_latest)
