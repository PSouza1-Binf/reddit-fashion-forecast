import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from brand_model import build_everything_from_posts

st.set_page_config(page_title="Brand Trend Forecaster", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


@st.cache_resource(show_spinner=True)
def build_models_from_df(df):
    return build_everything_from_posts(df)


st.title("🔎 Brand Trend Forecaster")
st.caption(
    "Search a brand, then see previous weekly engagement and model-predicted engagement for the next week."
)

# ---------------- Sidebar: data + search UI ----------------
with st.sidebar:
    st.header("⚙️ Data & brand search")

    # Search box
    brand_query = st.text_input("Brand search", placeholder="e.g. nike, adidas, shein...")

    # Explicit search button
    search_clicked = st.button("Search")


# ---------------- Load data ----------------
raw_df = pd.read_csv("merged_reddit_posts.zip", compression="zip")


# Run the ML pipeline from brand_model.py
bundle = build_models_from_df(raw_df)
agg = bundle["agg"]
watchlist = bundle["watchlist"]
pr_auc = bundle["pr_auc"]
mae = bundle["mae"]

if agg.empty or watchlist.empty:
    st.error(
        "Not enough usable data after preprocessing to train the model.\n\n"
        "Make sure your CSV has at least these columns: "
        "`created_utc`, `title`, `selftext`, `engagement`, `num_comments`."
    )
    st.stop()

# ---------------- Handle search state with button ----------------
if "last_brand_query" not in st.session_state:
    st.session_state["last_brand_query"] = ""

# When user clicks Search, store their query
if search_clicked and brand_query.strip():
    st.session_state["last_brand_query"] = brand_query.strip()

active_query = st.session_state["last_brand_query"].strip()

if not active_query:
    st.info("Type a brand in the sidebar and click **Search** to see its trends and forecast.")
    st.stop()

# Case-insensitive brand lookup
all_brands = sorted(agg["brand"].dropna().unique().tolist())
brand_map = {b.lower(): b for b in all_brands}
q = active_query.lower()

if q not in brand_map:
    examples = ", ".join(all_brands[:10])
    st.error(
        f"Brand '{active_query}' not found in the data. "
        f"Try one of: {examples} ..."
    )
    st.stop()

brand = brand_map[q]

# ---------------- Previous weeks' engagement for brand ----------------
brand_weekly = (
    agg[agg["brand"] == brand]
    .groupby(["week_start"], as_index=False)
    .agg(
        posts=("posts", "sum"),
        engagement=("engagement_sum", "sum"),
        sentiment=("sentiment_mean", "mean"),
    )
)

if brand_weekly.empty:
    st.warning("No weekly aggregates found for this brand.")
    st.stop()

st.subheader(f"📆 Previous weekly engagement for '{brand}'")
fig_hist = px.line(
    brand_weekly,
    x="week_start",
    y="engagement",
    labels={"week_start": "Week", "engagement": "Engagement"},
)
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------- Future week engagement for this brand ----------------
st.subheader(f"🔮 Future week engagement forecast for '{brand}'")

brand_watch = watchlist[watchlist["brand"] == brand].copy()

if brand_watch.empty:
    st.info("No forecast rows for this brand in the latest week.")
else:
    # Brand-level latest actual engagement
    latest_week_start = brand_watch["week_start"].iloc[0]
    latest_actual = (
        agg[
            (agg["brand"] == brand)
            & (agg["week_start"] == latest_week_start)
        ]["engagement_sum"]
        .sum()
    )

    # Brand-level predicted next-week engagement: sum predictions over microtopics
    predicted_next = brand_watch["pred_next_engagement"].sum()

    if latest_actual > 0:
        change_pct = (predicted_next - latest_actual) / latest_actual * 100.0
    else:
        change_pct = np.nan

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
        f"Model evaluation (time-based split): "
        f"PR-AUC = {pr_auc:.3f}, MAE (next engagement) = {mae:.2f}"
    )

    st.write("Top microtopics for this brand (latest week → next-week forecast):")
    st.dataframe(brand_watch.head(20))
