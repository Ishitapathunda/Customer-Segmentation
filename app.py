import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🛍️ Customer Segmentation Dashboard")

st.write("Upload the Online Retail dataset (CSV format) to view segmentation insights.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    st.write("### Sample Data")
    st.dataframe(df.head())

    # Show distribution of monetary values
    st.write("### Distribution of Purchase Amounts")
    fig, ax = plt.subplots()
    sns.histplot(df["UnitPrice"]*df["Quantity"], bins=50, ax=ax)
    st.pyplot(fig)
