import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_clean_data(file_path="data/transactions.csv"):
    df = pd.read_csv(file_path)

    # Renombrar columnas para que coincidan con el estándar de la app
    rename_map = {
        "amt": "amount",
        "trans_date_trans_time": "timestamp",
        "lat": "latitude",
        "long": "longitude",
        "category": "merchant_category",
        "merchant": "merchant_name",
        "cc_num": "user_id"  # Asumimos que cc_num puede servir como identificador de usuario
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
    
    # Corregir el error: Añadir columna para tiempo (mañana, tarde, noche)
    # El problema es que "noche" aparece dos veces en los labels, lo que no está permitido cuando ordered=True
    # Solución: usar ordered=False o usar labels únicos
    
    # Opción 1: Usar ordered=False
    df["time_of_day"] = pd.cut(
        df["hour"], 
        bins=[0, 6, 12, 18, 24], 
        labels=["noche", "mañana", "tarde", "noche"],
        ordered=False  # Establecer ordered=False para permitir etiquetas duplicadas
    )
    
    # Opción 2 (alternativa): Usar etiquetas únicas
    # hour_mapping = {
    #     0: "noche_madrugada", 1: "noche_madrugada", 2: "noche_madrugada", 
    #     3: "noche_madrugada", 4: "noche_madrugada", 5: "noche_madrugada",
    #     6: "mañana", 7: "mañana", 8: "mañana", 9: "mañana", 10: "mañana", 11: "mañana",
    #     12: "tarde", 13: "tarde", 14: "tarde", 15: "tarde", 16: "tarde", 17: "tarde",
    #     18: "noche", 19: "noche", 20: "noche", 21: "noche", 22: "noche", 23: "noche"
    # }
    # df["time_of_day"] = df["hour"].map(hour_mapping)
    
    return df

def aggregate_by_location(df, lat_col='latitude', lon_col='longitude'):
    grouped = df.groupby([lat_col, lon_col, 'merchant_category']).agg(
        total_spent=('amount', 'sum'),
        avg_spent=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    return grouped

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
        transaction_count=('amount', 'count')
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