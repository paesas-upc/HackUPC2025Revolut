import pandas as pd

def load_and_clean_data(file_path="data/transactions.csv"):
    df = pd.read_csv(file_path)

    # Renombrar columnas para que coincidan con el estándar de la app
    rename_map = {
        "amt": "amount",
        "trans_date_trans_time": "timestamp",
        "lat": "latitude",
        "long": "longitude",
        "category": "merchant_category"
    }
    df = df.rename(columns=rename_map)

    # Eliminar columna innecesaria si existe
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convertir columna de fecha
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    return df

def aggregate_by_location(df, lat_col='latitude', lon_col='longitude'):
    grouped = df.groupby([lat_col, lon_col, 'merchant_category']).agg(
        total_spent=('amount', 'sum'),
        avg_spent=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    return grouped
