# fashion-forecast.py
# Streamlit app that lazily calls the pipeline and uses its results

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Any, Dict

from new_brand_model import build_model_bundle  # stays lightweight at import

st.set_page_config(page_title="Brand Trend Forecaster", layout="wide")
st.title("🔎 Brand Trend Forecaster")
st.caption("Search brand/item and view historical + forecast engagement with surge alerts.")

# Sidebar: data paths & build control
with st.sidebar:
    st.header("Data & build controls")
    posts_path = st.text_input("Posts ZIP/CSV path", value="Data/merged_reddit_posts_final.zip")
    comments_path = st.text_input("Comments ZIP/CSV path", value="Data/all_comments_multi2.zip")
    allow_downloads = st.checkbox("Allow NLP asset downloads (only if missing)", value=False)
    st.markdown("---")
    st.write("Press **Build** to run the pipeline (cached). This may take several minutes on first run.")
    build_clicked = st.button("Build")

# cached loader - heavy work runs only here
@st.cache_resource(show_spinner=True)
def load_bundle_cached(posts: str, comments: str, allow_downloads: bool) -> Dict[str, Any]:
    return build_model_bundle(posts, comments, do_downloads=allow_downloads)

if not build_clicked and "bundle_loaded" not in st.session_state:
    st.info("Click **Build** in the sidebar to run the pipeline and populate the app.")
    st.stop()

try:
    with st.spinner("Building / loading model bundle (this may take several minutes)..."):
        bundle = load_bundle_cached(posts_path, comments_path, allow_downloads)
except Exception as e:
    st.error("Pipeline failed. See the error below; try running new_brand_model.py offline to debug.")
    st.exception(e)
    st.stop()

# Unpack bundle
agg = bundle["agg"]
watchlist = bundle["watchlist"]
pr_auc = bundle.get("pr_auc", None)
mae = bundle.get("mae", None)

# Sidebar filter controls (populated from agg)
with st.sidebar:
    st.markdown("---")
    st.header("Filters")
    all_brands = sorted(agg["brand"].dropna().unique().tolist())
    all_items = sorted(agg["item"].dropna().unique().tolist())

    SELECT_MANUAL_BRAND = "-- Type brand manually --"
    SELECT_MANUAL_ITEM = "-- Type item manually --"

    brand_choice = st.selectbox("Brand (searchable)", [SELECT_MANUAL_BRAND] + all_brands)
    manual_brand = st.text_input("Brand manually") if brand_choice == SELECT_MANUAL_BRAND else ""
    item_choice = st.selectbox("Item (searchable)", [SELECT_MANUAL_ITEM] + all_items)
    manual_item = st.text_input("Item manually") if item_choice == SELECT_MANUAL_ITEM else ""
    search_clicked = st.button("Search")

# manage session state for search
if "brand_q" not in st.session_state:
    st.session_state["brand_q"] = ""
if "item_q" not in st.session_state:
    st.session_state["item_q"] = ""

if search_clicked:
    st.session_state["brand_q"] = (manual_brand if brand_choice == SELECT_MANUAL_BRAND else brand_choice).strip()
    st.session_state["item_q"] = (manual_item if item_choice == SELECT_MANUAL_ITEM else item_choice).strip()

active_brand = st.session_state["brand_q"].strip()
active_item = st.session_state["item_q"].strip()

if not active_brand and not active_item:
    st.info("Choose a brand and/or item (sidebar) and click Search.")
    st.stop()

# Filtering logic
if active_brand and active_item:
    microtopic = f"{active_brand} | {active_item}"
    filtered = agg[agg["microtopic"] == microtopic]
elif active_brand:
    filtered = agg[agg["brand"] == active_brand]
else:
    filtered = agg[agg["item"] == active_item]

if filtered.empty:
    st.warning("No data found for this selection.")
    st.stop()

# Build watchlist subset
if active_brand and active_item:
    w = watchlist[(watchlist["brand"] == active_brand) & (watchlist["item"] == active_item)].copy()
elif active_brand:
    w = watchlist[watchlist["brand"] == active_brand].copy()
else:
    w = watchlist[watchlist["item"] == active_item].copy()

if w.empty:
    st.info("No forecast rows available for this selection in the latest week.")
    st.stop()

# weighted surge prob
w["weight"] = w["engagement_sum"] / (w["engagement_sum"].sum() if w["engagement_sum"].sum() > 0 else 1)
brand_surge_prob = float((w["surge_prob"] * w["weight"]).sum())

# Tabs
tab_overview, tab_hist, tab_forecast, tab_surge, tab_raw = st.tabs([
    "📌 Overview", "📆 Historical Trends", "🔮 Forecast", "🔥 Surge", "📄 Explore Raw Data"
])

# Overview
with tab_overview:
    st.subheader("Overview")
    latest_week = w["week_start"].iloc[0]
    latest_actual = float(filtered[filtered["week_start"] == latest_week]["engagement_sum"].sum())
    pred_next = float(w["pred_next_engagement"].sum())
    change_pct = (pred_next - latest_actual) / latest_actual * 100.0 if latest_actual > 0 else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest engagement", f"{latest_actual:.1f}")
    col2.metric("Predicted next-week engagement", f"{pred_next:.1f}")
    col3.metric("Expected change", f"{change_pct:+.1f}%" if np.isfinite(change_pct) else "N/A")

    st.metric("Weighted surge probability", f"{brand_surge_prob:.2%}")
    if pr_auc is not None and mae is not None:
        st.caption(f"Model eval — PR-AUC: {pr_auc:.3f}, MAE: {mae:.2f}")

# Historical
with tab_hist:
    st.subheader("Historical Trends")
    hist = (filtered.groupby("week_start", as_index=False)
            .agg(posts=("posts", "sum"), engagement=("engagement_sum", "sum"), sentiment=("sentiment_mean", "mean")))
    fig = px.line(hist, x="week_start", y="engagement", labels={"week_start": "Week", "engagement": "Engagement"})
    st.plotly_chart(fig, use_container_width=True)

# Forecast
with tab_forecast:
    st.subheader("Forecast")
    w_sorted = w.sort_values("pred_next_engagement", ascending=False)
    st.write("Forecasted microtopics contributing to next week's engagement:")
    st.dataframe(w_sorted.head(20))

    # Next-week chart
    next_week_df = pd.DataFrame({
        "week": ["Latest Week", "Next Week"],
        "engagement": [latest_actual, pred_next]
    })
    fig_next = px.line(next_week_df, x="week", y="engagement", markers=True, title="Next-Week Engagement Forecast")
    st.plotly_chart(fig_next, use_container_width=True)

    # Two-week heuristic projection
    st.subheader("Two-week projection (simple heuristic)")
    pct_change = (pred_next - latest_actual) / latest_actual if latest_actual > 0 else 0.0
    week2 = pred_next * (1 + pct_change)
    two_df = pd.DataFrame({
        "week": ["Latest Week", "Next Week", "Week 2"],
        "engagement": [latest_actual, pred_next, week2]
    })
    fig_two = px.line(two_df, x="week", y="engagement", markers=True, title="Two-Week Engagement Forecast (heuristic)")
    st.plotly_chart(fig_two, use_container_width=True)

# Surge
with tab_surge:
    st.subheader("Surge Alerts & Top Microtopics")
    if brand_surge_prob > 0.90:
        st.error(f"🚨 Surge Alert! Weighted surge probability = {brand_surge_prob:.2%}")
    else:
        st.success(f"Surge probability = {brand_surge_prob:.2%}")

    micro_surge = (w.assign(weighted_surge=w["surge_prob"] * w["weight"])
                   .sort_values("weighted_surge", ascending=False)
                   .head(10))
    st.write("Top microtopics contributing most to expected surge:")
    st.dataframe(micro_surge[["microtopic", "surge_prob", "weight", "weighted_surge", "pred_next_engagement"]])

# Raw
with tab_raw:
    st.subheader("Explore Raw Data")
    st.write("Filtered aggregated data (agg):")
    st.dataframe(filtered)
    st.write("Forecast rows (watchlist subset):")
    st.dataframe(w)

# Footer
if pr_auc is not None and mae is not None:
    st.caption(f"Model eval — PR-AUC: {pr_auc:.3f}, MAE: {mae:.2f}")
