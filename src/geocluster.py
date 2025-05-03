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
    # Obtener coordenadas únicas
    coords = df[['latitude', 'longitude']].drop_duplicates().values
    
    # Verificar si hay suficientes puntos para hacer clustering
    if len(coords) < min_cluster_size:
        # Si no hay suficientes puntos, asignar todos al mismo cluster (0)
        df['cluster'] = 0
        return df
    
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
    
    # Si todos los puntos tienen el mismo cluster o no hay suficientes datos
    if df['cluster'].nunique() <= 1 or len(df) < 3:
        # Verificar que hay datos antes de calcular estadísticas
        if len(df) > 0:
            try:
                dominant_category = df['merchant_category'].value_counts().index[0]
            except (IndexError, KeyError):
                dominant_category = 'Desconocido'
                
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
            # Si no hay datos, crear un DataFrame con valores por defecto
            stats = pd.DataFrame({
                'cluster': [0],
                'avg_spent': [0],
                'total_spent': [0],
                'transaction_count': [0],
                'avg_lat': [0],
                'avg_lon': [0],
                'dominant_category': ['Desconocido'],
                'category_diversity': [0],
                'avg_hour': [12]
            })
        
        stats['center_coords'] = list(zip(stats['avg_lat'], stats['avg_lon']))
        return stats
    
    # Para casos con múltiples clusters, proceder con el análisis normal
    # Filtrar puntos sin cluster (-1)
    df_clustered = df[df['cluster'] != -1]
    
    # Si después de filtrar no quedan datos, crear un dataframe vacío con la estructura correcta
    if df_clustered.empty:
        stats = pd.DataFrame({
                'cluster': [0],
                'avg_spent': [0],
                'total_spent': [0],
                'transaction_count': [0],
                'avg_lat': [0],
                'avg_lon': [0],
                'dominant_category': ['Desconocido'],
                'category_diversity': [0],
                'avg_hour': [12]
            })
        stats['center_coords'] = list(zip(stats['avg_lat'], stats['avg_lon']))
        return stats
    
    # Análisis por cluster
    try:
        cluster_stats = df_clustered.groupby('cluster').agg(
            avg_spent=('amount', 'mean'),
            total_spent=('amount', 'sum'),
            transaction_count=('amount', 'count'),
            avg_lat=('latitude', 'mean'),
            avg_lon=('longitude', 'mean'),
            dominant_category=('merchant_category', lambda x: x.value_counts().index[0] if len(x) > 0 else 'Desconocido'),
            category_diversity=('merchant_category', lambda x: len(x.unique())),
            avg_hour=('hour', 'mean' if 'hour' in df.columns else lambda x: 12)
        ).reset_index()
        
        # Calcular centro geográfico de cada cluster
        cluster_stats['center_coords'] = list(zip(cluster_stats['avg_lat'], cluster_stats['avg_lon']))
        
        return cluster_stats
    
    except Exception as e:
        # En caso de error, devolver un DataFrame básico
        print(f"Error en analyze_clusters: {str(e)}")
        stats = pd.DataFrame({
            'cluster': [0],
            'avg_spent': [df['amount'].mean() if 'amount' in df.columns else 0],
            'total_spent': [df['amount'].sum() if 'amount' in df.columns else 0],
            'transaction_count': [len(df)],
            'avg_lat': [df['latitude'].mean() if 'latitude' in df.columns else 0],
            'avg_lon': [df['longitude'].mean() if 'longitude' in df.columns else 0],
            'dominant_category': ['Desconocido'],
            'category_diversity': [0],
            'avg_hour': [12]
        })
        stats['center_coords'] = list(zip(stats['avg_lat'], stats['avg_lon']))
        return stats