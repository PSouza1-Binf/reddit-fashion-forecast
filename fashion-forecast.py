import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from brand_model import build_everything_from_posts

st.set_page_config(page_title="Brand Trend Forecaster", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(p): return pd.read_csv(p)

@st.cache_resource(show_spinner=True)
def build_models_from_df(df): return build_everything_from_posts(df)

st.title("🔎 Brand Trend Forecaster")
st.caption("Search brand/item and view historical + forecast engagement with surge alerts.")

raw_df = pd.read_csv("merged_reddit_posts_final.zip", compression="zip")
bundle = build_models_from_df(raw_df)
agg, watchlist, pr_auc, mae = bundle["agg"], bundle["watchlist"], bundle["pr_auc"], bundle["mae"]

all_brands = sorted(agg["brand"].dropna().unique().tolist())
all_items = sorted(agg["item"].dropna().unique().tolist())

with st.sidebar:
    st.header("⚙️ Brand & Item Search")

    SELECT_MANUAL_BRAND = "-- Type brand manually --"
    SELECT_MANUAL_ITEM = "-- Type item manually --"

    brand_choice = st.selectbox("Brand (searchable)", [SELECT_MANUAL_BRAND] + all_brands)
    manual_brand = st.text_input("Brand manually") if brand_choice == SELECT_MANUAL_BRAND else ""

    item_choice = st.selectbox("Item (searchable)", [SELECT_MANUAL_ITEM] + all_items)
    manual_item = st.text_input("Item manually") if item_choice == SELECT_MANUAL_ITEM else ""

    search_clicked = st.button("Search")

if "brand_q" not in st.session_state:
    st.session_state["brand_q"] = ""
if "item_q" not in st.session_state:
    st.session_state["item_q"] = ""

if search_clicked:
    st.session_state["brand_q"] = (manual_brand if brand_choice == SELECT_MANUAL_BRAND else brand_choice).strip()
    st.session_state["item_q"] = (manual_item if item_choice == SELECT_MANUAL_ITEM else item_choice).strip()

active_brand = st.session_state["brand_q"]
active_item = st.session_state["item_q"]

# FILTERING
if active_brand and active_item:
    filtered = agg[(agg.brand == active_brand) & (agg.item == active_item)]
elif active_brand:
    filtered = agg[(agg.brand == active_brand)]
elif active_item:
    filtered = agg[(agg.item == active_item)]
else:
    st.info("Select a brand or item and press Search.")
    st.stop()

if filtered.empty:
    st.warning("No data found.")
    st.stop()

# HISTORICAL
hist = filtered.groupby("week_start", as_index=False).agg(
    posts=("posts", "sum"),
    engagement=("engagement_sum", "sum")
)

st.subheader("📆 Historical Engagement")
st.plotly_chart(px.line(hist, x="week_start", y="engagement"), use_container_width=True)

# FORECAST ROWS
if active_brand and active_item:
    w = watchlist[(watchlist.brand == active_brand) & (watchlist.item == active_item)].copy()
elif active_brand:
    w = watchlist[watchlist.brand == active_brand].copy()
else:
    w = watchlist[watchlist.item == active_item].copy()

if w.empty:
    st.info("No forecasts available.")
    st.stop()

latest_week = w.week_start.iloc[0]
latest_actual = filtered[filtered.week_start == latest_week].engagement_sum.sum()
pred_next = w.pred_next_engagement.sum()

# SURGE PROBABILITY — weighted (option B)
w["weight"] = w.engagement_sum / (w.engagement_sum.sum() or 1)
brand_surge_prob = float((w.surge_prob * w.weight).sum())

# ALERT
if brand_surge_prob > 0.90:
    st.error(f"🚨 Surge Alert! Weighted surge probability = {brand_surge_prob:.2%}")
else:
    st.success(f"Surge probability = {brand_surge_prob:.2%}")

change_pct = (pred_next - latest_actual) / latest_actual * 100 if latest_actual > 0 else np.nan

col1, col2, col3 = st.columns(3)
col1.metric("Latest engagement", f"{latest_actual:.1f}")
col2.metric("Predicted next week", f"{pred_next:.1f}")
col3.metric("Expected change", f"{change_pct:+.1f}%" if np.isfinite(change_pct) else "N/A")

st.caption(f"Model eval — PR-AUC: {pr_auc:.3f}, MAE: {mae:.2f}")
st.dataframe(w.head(20))

