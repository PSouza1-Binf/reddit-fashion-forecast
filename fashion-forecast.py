import streamlit as st
import pandas as pd
import numpy as np

st.title("🧪 Tabs Example Demo")

tab1, tab2, tab3 = st.tabs(["📈 Chart", "📄 Table", "🔍 Details"])

with tab1:
    st.subheader("Chart View")
    st.line_chart(np.random.randn(20, 3))

with tab2:
    st.subheader("Table View")
    df = pd.DataFrame({
        "A": np.random.randint(0, 100, 10),
        "B": np.random.randn(10),
        "C": list("ABCDEFGHIJ")
    })
    st.dataframe(df)

with tab3:
    st.subheader("Details / Commentary")
    st.write("""
        Tabs let you organize content clearly.
        Useful for:
        - separating brand vs item vs microtopic
        - showing historical vs forecast vs alerts
        - lots of charts without scrolling
    """)

st.write("These microtopics contribute most to the brand's predicted surge:")
st.dataframe(micro_surge[["microtopic", "surge_prob", "weight", "weighted_surge", "pred_next_engagement"]])
st.caption(f"Model eval — PR-AUC: {pr_auc:.3f}, MAE: {mae:.2f}")
st.dataframe(w.head(20))

