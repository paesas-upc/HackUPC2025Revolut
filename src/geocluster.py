# src/geocluster.py

import hdbscan
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle

def cluster_locations(df, min_cluster_size=5, method='hdbscan'):
    """
    Groups locations into geographic clusters.
    
    Args:
        df: DataFrame with latitude and longitude columns
        min_cluster_size: Minimum cluster size for HDBSCAN
        method: 'hdbscan' or 'dbscan'
    
    Returns:
        DataFrame with cluster column added
    """
    # Get unique coordinates
    coords = df[['latitude', 'longitude']].drop_duplicates().values
    
    # Check if there are enough points to perform clustering
    if len(coords) < min_cluster_size:
        # If there aren't enough points, assign all to the same cluster (0)
        df['cluster'] = 0
        return df
    
    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            metric='haversine',
            gen_min_span_tree=True
        )
        cluster_labels = clusterer.fit_predict(coords)
        
        # Create a mapping of (lat, lon) to cluster label
        location_to_cluster = {
            (coords[i][0], coords[i][1]): label 
            for i, label in enumerate(cluster_labels)
        }
        
        # Apply the mapping to the original DataFrame
        df['cluster'] = df.apply(
            lambda row: location_to_cluster.get((row['latitude'], row['longitude']), -1), 
            axis=1
        )
    
    elif method == 'dbscan':
        # Convert lat/lon to radians for DBSCAN with haversine metric
        coords_rad = np.radians(coords)
        
        # Epsilon in radians (~500 meters)
        epsilon = 500 / 6371000
        
        # Execute DBSCAN
        db = DBSCAN(
            eps=epsilon, 
            min_samples=min_cluster_size, 
            algorithm='ball_tree', 
            metric='haversine'
        ).fit(coords_rad)
        
        # Create mapping
        location_to_cluster = {
            (coords[i][0], coords[i][1]): label 
            for i, label in enumerate(db.labels_)
        }
        
        # Apply the mapping to the original DataFrame
        df['cluster'] = df.apply(
            lambda row: location_to_cluster.get((row['latitude'], row['longitude']), -1), 
            axis=1
        )
    
    return df

def analyze_clusters(df):
    """
    Analyzes the clusters to obtain relevant statistics
    """
    if 'cluster' not in df.columns:
        raise ValueError("The DataFrame must have a 'cluster' column. Run cluster_locations first.")
    
    # If all points have the same cluster or there isn't enough data
    if df['cluster'].nunique() <= 1 or len(df) < 3:
        # Verify there's data before calculating statistics
        if len(df) > 0:
            try:
                dominant_category = df['merchant_category'].value_counts().index[0]
            except (IndexError, KeyError):
                dominant_category = 'Unknown'
                
            stats = pd.DataFrame({
                'cluster': [0],
                'avg_spent': [df['amount'].mean()],
                'total_spent': [df['amount'].sum()],
                'transaction_count': [len(df)],
                'avg_lat': [df['latitude'].mean()],
                'avg_lon': [df['longitude'].mean()],
                'dominant_category': [dominant_category],
                'category_diversity': [len(df['merchant_category'].unique())],
                'avg_hour': [df['hour'].mean() if 'hour' in df.columns else 12]
            })
        else:
            # If there's no data, create a DataFrame with default values
            stats = pd.DataFrame({
                'cluster': [0],
                'avg_spent': [0],
                'total_spent': [0],
                'transaction_count': [0],
                'avg_lat': [0],
                'avg_lon': [0],
                'dominant_category': ['Unknown'],
                'category_diversity': [0],
                'avg_hour': [12]
            })
        
        stats['center_coords'] = list(zip(stats['avg_lat'], stats['avg_lon']))
        return stats
    
    # For cases with multiple clusters, proceed with normal analysis
    # Filter points without a cluster (-1)
    df_clustered = df[df['cluster'] != -1]
    
    # If after filtering there's no data left, create an empty dataframe with the correct structure
    if df_clustered.empty:
        stats = pd.DataFrame({
                'cluster': [0],
                'avg_spent': [0],
                'total_spent': [0],
                'transaction_count': [0],
                'avg_lat': [0],
                'avg_lon': [0],
                'dominant_category': ['Unknown'],
                'category_diversity': [0],
                'avg_hour': [12]
            })
        stats['center_coords'] = list(zip(stats['avg_lat'], stats['avg_lon']))
        return stats
    
    # Analysis by cluster
    try:
        cluster_stats = df_clustered.groupby('cluster').agg(
            avg_spent=('amount', 'mean'),
            total_spent=('amount', 'sum'),
            transaction_count=('amount', 'count'),
            avg_lat=('latitude', 'mean'),
            avg_lon=('longitude', 'mean'),
            dominant_category=('merchant_category', lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'),
            category_diversity=('merchant_category', lambda x: len(x.unique())),
            avg_hour=('hour', 'mean' if 'hour' in df.columns else lambda x: 12)
        ).reset_index()
        
        # Calculate geographic center of each cluster
        cluster_stats['center_coords'] = list(zip(cluster_stats['avg_lat'], cluster_stats['avg_lon']))
        
        return cluster_stats
    
    except Exception as e:
        # In case of error, return a basic DataFrame
        print(f"Error in analyze_clusters: {str(e)}")
        stats = pd.DataFrame({
            'cluster': [0],
            'avg_spent': [df['amount'].mean() if 'amount' in df.columns else 0],
            'total_spent': [df['amount'].sum() if 'amount' in df.columns else 0],
            'transaction_count': [len(df)],
            'avg_lat': [df['latitude'].mean() if 'latitude' in df.columns else 0],
            'avg_lon': [df['longitude'].mean() if 'longitude' in df.columns else 0],
            'dominant_category': ['Unknown'],
            'category_diversity': [0],
            'avg_hour': [12]
        })
        stats['center_coords'] = list(zip(stats['avg_lat'], stats['avg_lon']))
        return stats