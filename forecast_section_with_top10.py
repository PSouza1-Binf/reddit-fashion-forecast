# forecast_section_with_top10.py
# This file contains the Forecast Tab logic including:
# 1. Top 10 microtopic predictions
# 2. 4-week independent forecast line chart

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

def render_forecast_tab(filtered_latest, filtered_agg, brand, item):
    st.title("Forecast")

    # --- Section 1: Top 10 Microtopic Predictions (Bar Chart) ---
    st.subheader("Top 10 Microtopic Predictions for Next Week")

    top10 = filtered_latest.sort_values(
        "pred_next_engagement", ascending=False
    ).head(10)

    fig_top10 = px.bar(
        top10,
        x="microtopic",
        y="pred_next_engagement",
        color="pred_next_engagement",
        title="Top 10 Predicted Microtopics for Next Week",
    )
    st.plotly_chart(fig_top10, use_container_width=True)

    st.dataframe(top10)

    st.markdown("---")

    # --- Section 2: 4-Week Brand/Item Trend-Based Forecast Line Chart ---
    st.subheader("Brand / Item Engagement — 4 Week Forecast")

    if filtered_agg.empty:
        st.info("No time-series data available for this brand / item selection.")
        return

    # Historical weekly engagement data
    hist_bi = filtered_agg.groupby("week_start", as_index=False).agg(
        engagement=("engagement_sum", "sum"),
    )
    hist_bi = hist_bi.sort_values("week_start").reset_index(drop=True)
    hist_bi["type"] = "History"

    # Model prediction (week +1 baseline)
    if "pred_next_engagement" in filtered_latest.columns and not filtered_latest.empty:
        predicted_next = filtered_latest["pred_next_engagement"].sum()
    else:
        predicted_next = None

    if predicted_next is None or hist_bi.empty:
        st.info("Forecast unavailable for this brand / item selection.")
        return

    # Fit a simple trend if there are at least 2 historical points
    if len(hist_bi) >= 2:
        t = np.arange(len(hist_bi))
        y = hist_bi["engagement"].values

        slope, intercept = np.polyfit(t, y, 1)

        n_hist = len(hist_bi)
        t_future = np.arange(n_hist, n_hist + 4)
        y_future = intercept + slope * t_future

        # Calibrate so Week +1 matches ML model prediction
        offset = predicted_next - y_future[0]
        y_future = y_future + offset
    else:
        y_future = np.array([predicted_next] * 4)

    # Future week timestamps
    last_week = hist_bi["week_start"].max()
    future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, 5)]

    forecast_df = pd.DataFrame(
        {
            "week_start": future_weeks,
            "engagement": y_future,
            "type": ["Forecast"] * 4,
        }
    )

    # Combine history + forecast
    chart_df = pd.concat([hist_bi, forecast_df], ignore_index=True)

    title_suffix = ""
    if brand != "(All Brands)":
        title_suffix += f" — Brand: {brand}"
    if item != "(All Items)":
        title_suffix += f" — Item: {item}"

    fig_line = px.line(
        chart_df,
        x="week_start",
        y="engagement",
        color="type",
        markers=True,
        title=f"4-Week Trend Forecast{title_suffix}",
        labels={"week_start": "Week", "engagement": "Engagement"},
    )

    st.plotly_chart(fig_line, use_container_width=True)

    st.caption(
        "Forecast combines historical engagement (History) with a calibrated 4-week trend-based forecast "
        "rooted in the ML prediction for Week +1."
    )
