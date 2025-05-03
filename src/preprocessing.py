import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic

def load_and_clean_data(file_path="data/transactions.csv"):
    df = pd.read_csv(file_path)

    # Renombrar columnas para que coincidan con el estándar de la app
    rename_map = {
        "amt": "amount",
        "trans_date_trans_time": "timestamp",
        "lat": "latitude",
        "long": "longitude",
        "merch_lat": "merchant_latitude", 
        "merch_long": "merchant_longitude",
        "category": "merchant_category",
        "merchant": "merchant_name",
        "cc_num": "user_id",  # Asumimos que cc_num puede servir como identificador de usuario
        "first": "first",
        "last": "last"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Eliminar columna innecesaria si existe
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convertir columna de fecha
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Aseguramos que amount sea numérico
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    
    # Filtrar registros con valores no válidos
    df = df.dropna(subset=["latitude", "longitude", "amount"])
    
    # Añadir columna para tiempo (mañana, tarde, noche)
    df["time_of_day"] = pd.cut(
        df["hour"], 
        bins=[0, 6, 12, 18, 24], 
        labels=["noche", "mañana", "tarde", "noche"],
        ordered=False
    )
    
    # Calcular distancia entre usuario y comercio
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
    Calcula la distancia en kilómetros entre coordenadas de usuario y comerciante
    usando la fórmula de Haversine (geodesic)
    """
    try:
        distance = geodesic(user_coords, merchant_coords).kilometers
        return distance
    except:
        return None

def aggregate_by_location(df, lat_col='latitude', lon_col='longitude', include_merchant=False):
    """
    Agrega transacciones por ubicación y categoría.
    Si include_merchant es True, agrega por ubicación del comerciante en lugar de usuario.
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
    Encuentra comercios cercanos a una ubicación dada que pertenecen a una categoría específica.
    
    Args:
        df: DataFrame con datos de transacciones
        user_lat: Latitud del usuario
        user_lon: Longitud del usuario
        category: Categoría de comercio a buscar
        max_distance: Distancia máxima en kilómetros
        
    Returns:
        DataFrame con comercios cercanos ordenados por precio (más barato primero)
    """
    # Filtrar por categoría
    category_df = df[df['merchant_category'] == category]
    
    # Agrupar por comerciante para obtener precio promedio
    merchants = category_df.groupby(['merchant_name', 'merchant_latitude', 'merchant_longitude']).agg(
        avg_price=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    
    # Calcular distancia desde la ubicación del usuario a cada comerciante
    merchants['distance_from_user'] = merchants.apply(
        lambda row: calculate_distance(
            (user_lat, user_lon),
            (row['merchant_latitude'], row['merchant_longitude'])
        ),
        axis=1
    )
    
    # Filtrar por distancia máxima y ordenar por precio
    nearby = merchants[merchants['distance_from_user'] <= max_distance].sort_values('avg_price')
    
    return nearby

def aggregate_by_time_location(df, time_period='month'):
    """Agrega datos por ubicación y período de tiempo"""
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
    Compara gastos entre diferentes zonas geográficas
    radius: radio en grados para considerar ubicaciones cercanas
    """
    # Creamos clusters aproximados basados en cercanía geográfica
    df['location_cluster'] = (
        (df['latitude'] * 100).astype(int) / 100 * 1000 + 
        (df['longitude'] * 100).astype(int) / 100
    )
    
    # Análisis por cluster
    cluster_analysis = df.groupby('location_cluster').agg(
        avg_spent=('amount', 'mean'),
        total_spent=('amount', 'sum'),
        transaction_count=('amount', 'count'),
        avg_lat=('latitude', 'mean'),
        avg_lon=('longitude', 'mean'),
        main_category=('merchant_category', lambda x: x.value_counts().index[0])
    ).reset_index()
    
    return cluster_analysis