import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic

def load_and_clean_data(file_path="data/transactions.csv"):
    df = pd.read_csv(file_path)

    # Rename columns to match the app's standard
    rename_map = {
        "amt": "amount",
        "trans_date_trans_time": "timestamp",
        "lat": "latitude",
        "long": "longitude",
        "merch_lat": "merchant_latitude", 
        "merch_long": "merchant_longitude",
        "category": "merchant_category",
        "merchant": "merchant_name",
        "cc_num": "user_id",  # We assume cc_num can serve as user identifier
        "first": "first",
        "last": "last"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Remove unnecessary column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convert date column
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    
    # Filter records with invalid values
    df = df.dropna(subset=["latitude", "longitude", "amount"])
    
    # Add column for time of day (morning, afternoon, night)
    df["time_of_day"] = pd.cut(
        df["hour"], 
        bins=[0, 6, 12, 18, 24], 
        labels=["night", "morning", "afternoon", "night"],
        ordered=False
    )
    
    # Calculate distance between user and merchant
    df["distance_to_merchant"] = df.apply(
        lambda row: calculate_distance(
            (row["latitude"], row["longitude"]),
            (row["merchant_latitude"], row["merchant_longitude"])
        ) if pd.notna(row["merchant_latitude"]) and pd.notna(row["merchant_longitude"]) else None,
        axis=1
    )
    
    return df

def calculate_distance(user_coords, merchant_coords):
    """
    Calculates the distance in kilometers between user and merchant coordinates
    using the Haversine formula (geodesic)
    """
    try:
        distance = geodesic(user_coords, merchant_coords).kilometers
        return distance
    except:
        return None

def aggregate_by_location(df, lat_col='latitude', lon_col='longitude', include_merchant=False):
    """
    Aggregates transactions by location and category.
    If include_merchant is True, aggregates by merchant location instead of user.
    """
    if include_merchant:
        lat_col = 'merchant_latitude'
        lon_col = 'merchant_longitude'
        
    grouped = df.groupby([lat_col, lon_col, 'merchant_category']).agg(
        total_spent=('amount', 'sum'),
        avg_spent=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        avg_distance=('distance_to_merchant', 'mean')
    ).reset_index()
    return grouped

def find_nearby_merchants(df, user_lat, user_lon, category, max_distance=5.0):
    """
    Finds merchants near a given location that belong to a specific category.
    
    Args:
        df: DataFrame with transaction data
        user_lat: User's latitude
        user_lon: User's longitude
        category: Merchant category to search for
        max_distance: Maximum distance in kilometers
        
    Returns:
        DataFrame with nearby merchants ordered by price (cheapest first)
    """
    # Filter by category
    category_df = df[df['merchant_category'] == category]
    
    # Group by merchant to get average price
    merchants = category_df.groupby(['merchant_name', 'merchant_latitude', 'merchant_longitude']).agg(
        avg_price=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    
    # Calculate distance from the user's location to each merchant
    merchants['distance_from_user'] = merchants.apply(
        lambda row: calculate_distance(
            (user_lat, user_lon),
            (row['merchant_latitude'], row['merchant_longitude'])
        ),
        axis=1
    )
    
    # Filter by maximum distance and sort by price
    nearby = merchants[merchants['distance_from_user'] <= max_distance].sort_values('avg_price')
    
    return nearby

def aggregate_by_time_location(df, time_period='month'):
    """Aggregates data by location and time period"""
    if time_period == 'month':
        time_col = 'month'
    elif time_period == 'day_of_week':
        time_col = 'day_of_week'
    elif time_period == 'time_of_day':
        time_col = 'time_of_day'
    else:
        time_col = 'date'
        
    grouped = df.groupby(['latitude', 'longitude', 'merchant_category', time_col]).agg(
        total_spent=('amount', 'sum'),
        avg_spent=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        avg_distance=('distance_to_merchant', 'mean')
    ).reset_index()
    return grouped

def compare_locations(df, radius=0.01):
    """
    Compares spending between different geographic areas
    radius: radius in degrees to consider nearby locations
    """
    # Create approximate clusters based on geographic proximity
    df['location_cluster'] = (
        (df['latitude'] * 100).astype(int) / 100 * 1000 + 
        (df['longitude'] * 100).astype(int) / 100
    )
    
    # Analysis by cluster
    cluster_analysis = df.groupby('location_cluster').agg(
        avg_spent=('amount', 'mean'),
        total_spent=('amount', 'sum'),
        transaction_count=('amount', 'count'),
        avg_lat=('latitude', 'mean'),
        avg_lon=('longitude', 'mean'),
        main_category=('merchant_category', lambda x: x.value_counts().index[0])
    ).reset_index()
    
    return cluster_analysis