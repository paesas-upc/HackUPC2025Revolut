# src/geocluster.py

import hdbscan
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle

def cluster_locations(df, min_cluster_size=5, method='hdbscan'):
    """
    Agrupa ubicaciones en clusters geográficos.
    
    Args:
        df: DataFrame con columnas latitude y longitude
        min_cluster_size: Tamaño mínimo de cluster para HDBSCAN
        method: 'hdbscan' o 'dbscan'
    
    Returns:
        DataFrame con columna cluster añadida
    """
    coords = df[['latitude', 'longitude']].drop_duplicates().values
    
    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            metric='haversine',
            gen_min_span_tree=True
        )
        cluster_labels = clusterer.fit_predict(coords)
        
        # Crear un mapeo de (lat, lon) a etiqueta de cluster
        location_to_cluster = {
            (coords[i][0], coords[i][1]): label 
            for i, label in enumerate(cluster_labels)
        }
        
        # Aplicar el mapeo al DataFrame original
        df['cluster'] = df.apply(
            lambda row: location_to_cluster.get((row['latitude'], row['longitude']), -1), 
            axis=1
        )
    
    elif method == 'dbscan':
        # Convertir lat/lon a radianes para DBSCAN con métrica haversine
        coords_rad = np.radians(coords)
        
        # Epsilon en radianes (~500 metros)
        epsilon = 500 / 6371000
        
        # Ejecutar DBSCAN
        db = DBSCAN(
            eps=epsilon, 
            min_samples=min_cluster_size, 
            algorithm='ball_tree', 
            metric='haversine'
        ).fit(coords_rad)
        
        # Crear mapeo
        location_to_cluster = {
            (coords[i][0], coords[i][1]): label 
            for i, label in enumerate(db.labels_)
        }
        
        # Aplicar el mapeo al DataFrame original
        df['cluster'] = df.apply(
            lambda row: location_to_cluster.get((row['latitude'], row['longitude']), -1), 
            axis=1
        )
    
    return df

def analyze_clusters(df):
    """
    Analiza los clusters para obtener estadísticas relevantes
    """
    if 'cluster' not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'cluster'. Ejecuta cluster_locations primero.")
    
    # Filtrar puntos sin cluster (-1)
    df_clustered = df[df['cluster'] != -1]
    
    # Análisis por cluster
    cluster_stats = df_clustered.groupby('cluster').agg(
        avg_spent=('amount', 'mean'),
        total_spent=('amount', 'sum'),
        transaction_count=('amount', 'count'),
        avg_lat=('latitude', 'mean'),
        avg_lon=('longitude', 'mean'),
        dominant_category=('merchant_category', lambda x: x.value_counts().index[0]),
        category_diversity=('merchant_category', lambda x: len(x.unique())),
        avg_hour=('hour', 'mean')
    ).reset_index()
    
    # Calcular centro geográfico de cada cluster
    cluster_stats['center_coords'] = list(zip(cluster_stats['avg_lat'], cluster_stats['avg_lon']))
    
    return cluster_stats