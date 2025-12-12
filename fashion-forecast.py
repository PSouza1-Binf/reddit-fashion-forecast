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
latest_features = art["latest_features"]        # DataFrame-like, already safe_X processed
latest_df = art["latest_df"].copy()             # microtopic-level latest week data
ts_agg = art.get("ts_agg")
ts_feat = art.get("ts_feat")


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Always build filter options from ALL time-series
all_brands = sorted(ts_agg["brand"].dropna().unique())
brand = st.sidebar.selectbox("Brand", ["(All Brands)"] + all_brands)

# Items depend on brand selection
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

# Apply filters universally
filtered_latest = apply_filters(latest_df)
filtered_agg = apply_filters(ts_agg)
filtered_ts  = apply_filters(ts_feat)


# ---------------------------------------------------------
# MODEL PREDICTIONS (safe merge on microtopic)
# ---------------------------------------------------------
if not filtered_latest.empty:
    # Build a features lookup table by microtopic
    features_lookup = latest_features.copy()
    features_lookup["microtopic"] = latest_df["microtopic"].values

    # Merge filtered_latest with its matching features row
    merged = filtered_latest.merge(
        features_lookup,
        on="microtopic",
        how="left",
        suffixes=("", "_feat")
    )

    # Extract only the feature columns used in training
    X = merged[FEATURE_COLS].astype(float)

    # Predict
    filtered_latest["surge_prob"] = clf.predict_proba(X)[:, 1]
    filtered_latest["pred_next_engagement"] = reg.predict(X)

    # Weighted surge
    filtered_latest["weighted_surge"] = (
        filtered_latest["surge_prob"] * np.log1p(filtered_latest["engagement_sum"])
    )

# ---------------------------------------------------------
# SAFETY CHECK: No rows after filtering
# ---------------------------------------------------------
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

    colA, colB = st.columns(2)
    colA.metric("Microtopics (filtered)", f"{len(filtered_latest):,}")
    colB.metric("Surge Threshold", f"{surge_threshold:.3f}")

    overview_cols = [
    "microtopic", "brand", "item",
    "engagement_sum", "sentiment_mean",
    "surge_prob", "pred_next_engagement"
                    ]

    st.dataframe(
    filtered_latest[overview_cols].sort_values("pred_next_engagement", ascending=False).head(20)
                )
    # ---- NEW: Top 10 items for next week ----
    st.subheader("Top 10 items for next week")

# Aggregate predictions at the item level within the current filters
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

# Make the columns a bit friendlier for display
    top_items = top_items.rename(
        columns={
            "brand": "Brand",
            "item": "Item",
            "current_engagement": "Current Engagement (sum)",
            "total_pred_next_engagement": "Next Week Engagement (pred, sum)",
            "avg_surge_prob": "Avg Surge Probability",
        }
    )

    st.dataframe(top_items, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — HISTORICAL TRENDS
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Historical Trends")

    if filtered_agg.empty:
        st.info("No time-series data available for this selection.")
    else:
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

        # Optional column filtering:
        st.dataframe(hist[["week_start", "engagement", "posts", "sentiment"]])

# ---------------------------------------------------------
# TAB 3 — FORECAST
# ---------------------------------------------------------
with tab3:
    st.title("Forecast")

    st.write("Predicted engagement for next week:")

    # Top predicted microtopics
    top_pred = (
        filtered_latest.sort_values("pred_next_engagement", ascending=False)
        .head(15)
    )

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

    # -----------------------------------------------------------------
    # BRAND-LEVEL FORECAST LINE CHART (Regression-Based, No Prophet)
    # -----------------------------------------------------------------
    if brand != "(All Brands)" and not filtered_agg.empty:

        brand_hist = filtered_agg.copy().groupby(
            "week_start", as_index=False
        ).agg(engagement_sum=("engagement_sum", "sum"))

        if not brand_hist.empty:

            brand_hist = brand_hist.sort_values("week_start")

            # Compute linear trend from historical engagement
            brand_hist["t"] = range(len(brand_hist))
            X = brand_hist["t"].values.reshape(-1, 1)
            y = brand_hist["engagement_sum"].values
            slope, intercept = np.polyfit(brand_hist["t"], y, 1)

            # Predicted next-week engagement from microtopics
            next_week_pred = float(filtered_latest["pred_next_engagement"].sum())

            # Build forecast horizon (6 weeks)
            future_weeks = 6
            t_future = np.arange(len(brand_hist), len(brand_hist) + future_weeks)
            trend_future = intercept + slope * t_future

            # Offset future line so week +1 matches model prediction
            offset = next_week_pred - trend_future[0]
            trend_future += offset

            fc_df = pd.DataFrame({
                "week_start": list(brand_hist["week_start"]) +
                              [brand_hist["week_start"].max() + pd.Timedelta(weeks=i) for i in range(1, future_weeks+1)],
                "engagement": list(brand_hist["engagement_sum"]) +
                              list(trend_future),
                "type": ["history"] * len(brand_hist) + ["forecast"] * future_weeks
            })

            st.subheader("📈 6-Week Engagement Forecast (Regression-Based)")

            fig_fc = px.line(
                fc_df,
                x="week_start",
                y="engagement",
                color="type",
                markers=True,
                labels={"week_start": "Week", "engagement": "Engagement"},
                title=f"Brand-Level Forecast — {brand} ({item})",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

# ---------------------------------------------------------
# TAB 4 — SURGE ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.title("Surge Analysis")

    st.subheader("Surge Alerts (surge_prob ≥ 0.40)")
    alerts = filtered_latest[filtered_latest["surge_prob"] >= 0.40].sort_values("surge_prob", ascending=False)

    if alerts.empty:
        st.success("No major surge warnings this week.")
    else:
        st.error("Surge alerts detected!")
        surge_alert_cols = [
        "microtopic", "brand", "item",
        "surge_prob", "engagement_sum"
                            ]
    st.dataframe(alerts[surge_alert_cols])


    st.subheader("Top Surging Microtopics (Weighted)")
    top_surge = filtered_latest.sort_values("weighted_surge", ascending=False).head(20)

    fig2 = px.bar(
        top_surge,
        x="microtopic",
        y="weighted_surge",
        color="surge_prob",
        title="Top Surging Microtopics"
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

