import streamlit as st

st.set_page_config(
    page_title="Jet Engine Hospital",
    page_icon="✈️",
    layout="wide",
)

st.title("Jet Engine Hospital")
st.subheader("NASA C-MAPSS Predictive Maintenance System")

st.info(
    "This dashboard will display RUL predictions, failure risks, "
    "anomaly scores, uncertainty, and maintenance recommendations."
)
