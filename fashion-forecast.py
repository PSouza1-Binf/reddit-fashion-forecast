
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Brand Trend Forecaster", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)
    return df

def prepare_df(df):
    # Basic expected columns
    expected = {"title", "selftext", "score", "num_comments", "created_utc"}
    missing = expected - set(df.columns)
    if missing:
        st.warning(f"Your CSV is missing columns: {sorted(list(missing))}. The app will try to proceed with what's available.")
    # Ensure fields exist
    if "title" not in df.columns: df["title"] = ""
    if "selftext" not in df.columns: df["selftext"] = ""
    if "score" not in df.columns: df["score"] = 0
    if "num_comments" not in df.columns: df["num_comments"] = 0
    if "created_utc" in df.columns:
        # Handle seconds since epoch (int) to datetime
        try:
            df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        except Exception:
            # Try generic parse
            df["created_dt"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    else:
        # fallback: try a 'created' column
        if "created" in df.columns:
            df["created_dt"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
        else:
            df["created_dt"] = pd.NaT

    # Engagement metric
    if "score" in df.columns and "num_comments" in df.columns:
        df["engagement"] = df["score"].fillna(0) + df["num_comments"].fillna(0)
    else:
        # If missing, count each post as 1
        df["engagement"] = 1

    # Clean text for search
    for col in ["title", "selftext"]:
        df[col] = df[col].astype(str).fillna("")
    df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.lower()

    df["date"] = df["created_dt"].dt.tz_convert("UTC").dt.date if df["created_dt"].notna().any() else pd.NaT
    return df

def filter_by_brand(df, brand, exact_word=False):
    if not brand:
        return df.iloc[0:0].copy()
    query = brand.lower().strip()
    if exact_word:
        # word boundary match
        mask = df["text"].str.contains(rf"\b{query}\b", regex=True, na=False)
    else:
        mask = df["text"].str.contains(query, regex=False, na=False)
    return df[mask].copy()

def daily_agg(df, start_date=None, end_date=None, engagement_col="engagement"):
    if "date" not in df.columns or df["date"].isna().all():
        return pd.DataFrame(columns=["date", "engagement"])
    d = df.copy()
    if start_date:
        d = d[d["date"] >= start_date]
    if end_date:
        d = d[d["date"] <= end_date]
    out = d.groupby("date", as_index=False)[engagement_col].sum().sort_values("date")
    return out

def forecast_series(daily_df, periods=14):
    # Require at least 10 points
    if daily_df.shape[0] < 10:
        return None, None
    series = daily_df.set_index("date")["engagement"].asfreq("D").fillna(0)
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
        fit = model.fit(optimized=True, use_brute=True)
        fc = fit.forecast(periods)
        # simple CI via residual std
        resid = fit.resid
        std = resid.std(ddof=1) if len(resid) > 1 else 0
        ci_low = fc - 1.96 * std
        ci_high = fc + 1.96 * std
        fc_df = pd.DataFrame({
            "date": fc.index,
            "forecast": fc.values,
            "ci_low": ci_low.values,
            "ci_high": ci_high.values
        })
        return fc_df, series.reset_index().rename(columns={"index":"date"})
    except Exception as e:
        st.warning(f"Forecast failed: {e}")
        return None, series.reset_index().rename(columns={"index":"date"})

# -------------------------
# UI
# -------------------------
st.title("🔎 Brand Trend Forecaster")
st.caption("Search a brand name to see past engagement and a short-term popularity forecast.")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    default_path = st.text_input("...or use a local path (optional)", value="merged_reddit_posts.csv")
    brand = st.text_input("Brand to search", placeholder="e.g., Nike, Shein, H&M")
    exact = st.checkbox("Exact word match", value=False)
    lookback_days = st.slider("Lookback window (days)", min_value=7, max_value=90, value=30, step=1)
    engagement_choice = st.selectbox("Engagement metric", ["score + num_comments", "score only", "num_comments only"])
    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)

# Load data
df = None
if uploaded is not None:
    df = load_csv(uploaded)
elif default_path:
    try:
        df = load_csv(default_path)
    except Exception as e:
        st.info("No CSV loaded yet. Upload a file in the sidebar.")

if df is None:
    st.stop()

# Prepare
df = prepare_df(df)
if engagement_choice == "score only" and "score" in df.columns:
    df["engagement"] = df["score"].fillna(0)
elif engagement_choice == "num_comments only" and "num_comments" in df.columns:
    df["engagement"] = df["num_comments"].fillna(0)
else:
    df["engagement"] = df["score"].fillna(0) + df["num_comments"].fillna(0)

# Date filter: last N days
if df["date"].isna().all():
    st.error("Could not parse dates from your CSV. Ensure 'created_utc' (seconds since epoch) or a valid 'created' datetime column is present.")
    st.stop()
end_date = df["date"].max()
start_date = end_date - pd.Timedelta(days=lookback_days - 1)

# Filter by brand
sub = filter_by_brand(df, brand, exact_word=exact)
sub_range = sub[(sub["date"] >= start_date) & (sub["date"] <= end_date)] if not sub.empty else sub

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Posts (last 7 days)", int(sub[sub["date"] >= (end_date - pd.Timedelta(days=6))].shape[0]) if not sub.empty else 0)
with col2:
    last7 = sub[sub["date"] >= (end_date - pd.Timedelta(days=6))]["engagement"].sum() if not sub.empty else 0
    prev7 = sub[(sub["date"] >= (end_date - pd.Timedelta(days=13))) & (sub["date"] <= (end_date - pd.Timedelta(days=7))) ]["engagement"].sum() if not sub.empty else 0
    delta = (last7 - prev7) if prev7 == 0 else ( (last7 - prev7) / max(prev7, 1) * 100 )
    st.metric("Engagement (last 7 days)", int(last7), f"{delta:+.1f}% vs prior week" if prev7>0 else f"{delta:+.0f}")
with col3:
    st.metric("Unique days in window", int(sub_range["date"].nunique()) if not sub_range.empty else 0)

if not brand:
    st.info("Enter a brand in the sidebar to see trends.")
    st.stop()

if sub.empty:
    st.warning("No posts matched your brand query in the dataset.")
    st.stop()

# Daily aggregation & charts
daily = daily_agg(sub, start_date, end_date)
if daily.empty:
    st.warning("No data in the selected window.")
    st.stop()

daily["rolling_7d"] = daily["engagement"].rolling(7).mean()

with st.container():
    st.subheader(f"📆 Daily engagement for '{brand}' (last {lookback_days} days)")
    fig = px.line(daily, x="date", y=["engagement", "rolling_7d"], labels={"value":"Engagement", "date":"Date", "variable":""})
    st.plotly_chart(fig, use_container_width=True)

# Forecast
st.subheader("🔮 14-day popularity forecast" if forecast_days==14 else f"🔮 {forecast_days}-day popularity forecast")
fc_df, series = forecast_series(daily[["date", "engagement"]], periods=forecast_days)
if fc_df is None:
    st.info("Not enough data to forecast reliably. Need at least ~10 days of history.")
else:
    hist_plot = daily[["date","engagement"]].copy()
    hist_plot["type"] = "history"
    fc_plot = fc_df.rename(columns={"forecast":"engagement"})[["date","engagement"]].copy()
    fc_plot["type"] = "forecast"
    combo = pd.concat([hist_plot, fc_plot], ignore_index=True)

    fig2 = px.line(combo, x="date", y="engagement", color="type", labels={"engagement":"Engagement", "date":"Date", "type":""})
    # Add CI as filled area
    fig2.add_traces(px.scatter(fc_df, x="date", y="ci_low").update_traces(visible=False).data)
    fig2.add_traces(px.scatter(fc_df, x="date", y="ci_high").update_traces(visible=False).data)
    # Streamlit ignores traces visibility; not adding filled area to keep simple

    st.plotly_chart(fig2, use_container_width=True)

    last7_mean = hist_plot.set_index("date")["engagement"].tail(7).mean()
    next7_mean = fc_df.set_index("date")["forecast"].head(7).mean()
    change = (next7_mean - last7_mean) / max(last7_mean, 1) * 100 if last7_mean else np.nan
    trend = "rising 📈" if change > 5 else ("falling 📉" if change < -5 else "flat ➖")
    st.markdown(f"**Expected trend:** {trend} (next 7d vs last 7d: {change:+.1f}%).")

# Table of recent matching posts
st.subheader("🧾 Recent posts matching your query")
recent = sub.sort_values("created_dt", ascending=False).loc[:, ["created_dt", "title", "score", "num_comments", "subreddit"] if "subreddit" in sub.columns else ["created_dt","title","score","num_comments"]].head(50)
st.dataframe(recent)
