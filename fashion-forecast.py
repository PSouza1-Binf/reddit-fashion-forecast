import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from brand_model import build_everything_from_posts

st.set_page_config(page_title="Brand & Item Trend Forecaster", layout="wide")

# =====================================================================
# CACHING HELPERS
# =====================================================================
@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)

@st.cache_resource(show_spinner=True)
def build_models_from_df(df):
    return build_everything_from_posts(df)


# =====================================================================
# TITLE
# =====================================================================
st.title("🔎 Brand & Item Trend Forecaster")
st.caption(
    "Search a brand and/or item, then explore historical engagement and model-predicted engagement for the next week."
)

# =====================================================================
# LOAD RAW DATA + RUN MODEL PIPELINE
# =====================================================================
raw_df = pd.read_csv("merged_reddit_posts_final.zip", compression="zip")

with st.spinner("Building models and aggregations (cached)..."):
    bundle = build_models_from_df(raw_df)

agg = bundle["agg"]
watchlist = bundle["watchlist"]
pr_auc = bundle["pr_auc"]
mae = bundle["mae"]

if agg.empty or watchlist.empty:
    st.error(
        "Not enough usable data after preprocessing to train the model.\n\n"
        "Make sure your CSV contains: `created_utc`, `title`, `selftext`, `engagement`, `num_comments`."
    )
    st.stop()


# =====================================================================
# SIDEBAR — BRAND + ITEM SEARCH
# =====================================================================
all_brands = sorted(agg["brand"].dropna().unique().tolist())
all_items = sorted(agg["item"].dropna().unique().tolist())

with st.sidebar:
    st.header("⚙️ Search Controls")

    # ---------------- BRAND SELECTBOX ----------------
    SELECT_MANUAL_BRAND = "-- Type brand manually --"
    brand_options = [SELECT_MANUAL_BRAND] + all_brands

    selected_brand_from_list = st.selectbox(
        "Choose brand (searchable):",
        brand_options,
        index=0
    )

    manual_brand = ""
    if selected_brand_from_list == SELECT_MANUAL_BRAND:
        manual_brand = st.text_input(
            "Or type brand manually:",
            placeholder="e.g. nike, adidas, shein..."
        )

    # ---------------- ITEM SELECTBOX ----------------
    SELECT_MANUAL_ITEM = "-- Type item manually --"
    item_options = [SELECT_MANUAL_ITEM] + all_items

    selected_item_from_list = st.selectbox(
        "Choose item (searchable):",
        item_options,
        index=0
    )

    manual_item = ""
    if selected_item_from_list == SELECT_MANUAL_ITEM:
        manual_item = st.text_input(
            "Or type item manually:",
            placeholder="e.g. dunk, hoodie, tote bag..."
        )

    # Search button controls session state
    search_clicked = st.button("Search")


# =====================================================================
# SESSION STATE — REMEMBER SEARCH SELECTION BETWEEN RERUNS
# =====================================================================
if "last_brand_query" not in st.session_state:
    st.session_state["last_brand_query"] = ""

if "last_item_query" not in st.session_state:
    st.session_state["last_item_query"] = ""

if search_clicked:
    # BRAND
    if selected_brand_from_list != SELECT_MANUAL_BRAND:
        st.session_state["last_brand_query"] = selected_brand_from_list.strip()
    elif manual_brand.strip():
        st.session_state["last_brand_query"] = manual_brand.strip()

    # ITEM
    if selected_item_from_list != SELECT_MANUAL_ITEM:
        st.session_state["last_item_query"] = selected_item_from_list.strip()
    elif manual_item.strip():
        st.session_state["last_item_query"] = manual_item.strip()


active_brand = st.session_state["last_brand_query"].strip()
active_item = st.session_state["last_item_query"].strip()


# =====================================================================
# VALIDATION
# =====================================================================
if not active_brand and not active_item:
    st.info("Choose a brand and/or item in the sidebar, then click **Search**.")
    st.stop()


# =====================================================================
# FILTER LOGIC (BRAND • ITEM • MICROTOPIC)
# =====================================================================
filtered = None
microtopic = None

# CASE 1: brand + item → specific microtopic
if active_brand and active_item:
    microtopic = f"{active_brand} | {active_item}"
    filtered = agg[agg["microtopic"] == microtopic]

# CASE 2: brand only
elif active_brand:
    filtered = agg[agg["brand"] == active_brand]

# CASE 3: item only
elif active_item:
    filtered = agg[agg["item"] == active_item]

if filtered is None or filtered.empty:
    st.warning("No data found for this brand/item combination.")
    st.stop()


# =====================================================================
# HISTORICAL WEEKLY VIEW
# =====================================================================
title_label = microtopic if microtopic else (active_brand or active_item)

st.subheader(f"📆 Historical Weekly Engagement — **{title_label}**")

weekly = (
    filtered.groupby("week_start", as_index=False)
    .agg(
        posts=("posts", "sum"),
        engagement=("engagement_sum", "sum"),
        sentiment=("sentiment_mean", "mean"),
    )
)

fig_hist = px.line(
    weekly,
    x="week_start",
    y="engagement",
    labels={"week_start": "Week", "engagement": "Engagement"},
)
st.plotly_chart(fig_hist, use_container_width=True)


# =====================================================================
# FORECAST VIEW
# =====================================================================
st.subheader(f"🔮 Forecast — Next Week Engagement for **{title_label}**")

# SELECT CORRECT WATCHLIST SLICE
if microtopic:
    watch = watchlist[watchlist["microtopic"] == microtopic]
elif active_brand:
    watch = watchlist[watchlist["brand"] == active_brand]
elif active_item:
    watch = watchlist[watchlist["item"] == active_item]

if watch.empty:
    st.info("No forecast rows available for this selection in the latest week.")
    st.stop()

latest_week_start = watch["week_start"].iloc[0]

latest_actual = (
    filtered[filtered["week_start"] == latest_week_start]["engagement_sum"].sum()
)

predicted_next = watch["pred_next_engagement"].sum()

change_pct = (
    (predicted_next - latest_actual) / latest_actual * 100.0
    if latest_actual > 0 else np.nan
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest week engagement", f"{latest_actual:.1f}")
with col2:
    st.metric("Predicted next-week engagement", f"{predicted_next:.1f}")
with col3:
    if np.isfinite(change_pct):
        st.metric("Expected change", f"{change_pct:+.1f}%")
    else:
        st.metric("Expected change", "N/A")

st.caption(
    f"Model evaluation (time-based split): PR-AUC = {pr_auc:.3f}, MAE = {mae:.2f}"
)

st.write("Top contributing microtopics for the selected brand/item:")
st.dataframe(watch.head(20))

