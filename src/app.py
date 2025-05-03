# src/app.py

import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from preprocessing import load_and_clean_data, aggregate_by_location

st.set_page_config(layout="wide")
st.title("🗺️ Spending Map AI")

# Load data
df = load_and_clean_data("data/transactions.csv")
agg = aggregate_by_location(df)

# Base map
center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for _, row in agg.iterrows():
    popup = f"""
    <b>Category:</b> {row['merchant_category']}<br>
    <b>Total spent:</b> ${row['total_spent']:.2f}<br>
    <b>Transactions:</b> {row['transaction_count']}
    """
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(marker_cluster)

folium_static(m)

