import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Title of the App
st.title("🚀 Uber Eats Delivery Analysis App")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload Uber Eats Dataset", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Convert numeric columns
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")

    # Remove "(min)" from Time_taken(min) and convert to integer
    df["Time_taken(min)"] = df["Time_taken(min)"].str.extract("(\\d+)").astype(int)

    # Fix Weatherconditions (remove "conditions ")
    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=True)

    # Convert time columns to datetime format
    df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], errors="coerce")
    df["Time_Order_picked"] = pd.to_datetime(df["Time_Order_picked"], errors="coerce")

    st.success("✅ Data Loaded Successfully!")

    # Show raw data
    if st.checkbox("🔍 Show Raw Data"):
        st.write(df.head())

    # Visualization 1: Delivery Time Distribution
    st.subheader("📊 Delivery Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Time_taken(min)"], bins=30, kde=True, color="blue", ax=ax)
    st.pyplot(fig)

    # Visualization 2: Impact of Traffic on Delivery Time
    st.subheader("🚦 Impact of Traffic on Delivery Time")
    fig, ax = plt.subplots()
    sns.boxplot(x="Road_traffic_density", y="Time_taken(min)", data=df, ax=ax)
    st.pyplot(fig)

    # Clustering Analysis
    st.subheader("🌀 Clustering Analysis of Delivery Times")
    features = ["Time_taken(min)", "Delivery_person_Age", "Delivery_person_Ratings"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    
    fig, ax = plt.subplots()
    sns.scatterplot(x="Delivery_person_Age", y="Time_taken(min)", hue=df["Cluster"], palette="viridis", ax=ax)
    st.pyplot(fig)

    st.success("✅ Clustering Analysis Completed!")

    # Summary Metrics
    st.subheader("📌 Summary Statistics")
    st.write(df.describe())

    # Filter data by city
    st.subheader("🏙️ Filter Data by City")
    selected_city = st.selectbox("Select a city:", df["City"].unique())
    filtered_df = df[df["City"] == selected_city]
    st.write(filtered_df.head())
