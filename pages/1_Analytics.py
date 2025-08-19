import streamlit as st
import pandas as pd

st.header("Analytics")
st.write("This is the Analytics page.")

try:
    df = pd.read_csv("data/sales_data.csv")
    st.write("Sales Data:")
    st.dataframe(df)
except FileNotFoundError:
    st.error("Sales data file not found. Please make sure 'sales_data.csv' is in the 'data' directory.")
