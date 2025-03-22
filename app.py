import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Uber Eats Delivery Analysis")

df = pd.read_csv("uber-eats-deliveries.csv")

if st.checkbox("Show raw data"):
    st.write(df.head())

st.subheader("Delivery Time Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Time_taken(min)"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Traffic vs. Delivery Time")
fig, ax = plt.subplots()
sns.boxplot(x="Road_traffic_density", y="Time_taken(min)", data=df, ax=ax)
st.pyplot(fig)
