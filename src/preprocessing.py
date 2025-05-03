# src/preprocessing.py

import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    return df

def aggregate_by_location(df, lat_col='latitude', lon_col='longitude'):
    grouped = df.groupby([lat_col, lon_col, 'merchant_category']).agg(
        total_spent=('amount', 'sum'),
        avg_spent=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    return grouped

