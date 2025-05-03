# src/geocluster.py

import hdbscan
import pandas as pd

def cluster_locations(df, min_cluster_size=5):
    coords = df[['latitude', 'longitude']].drop_duplicates().values
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine')
    df['cluster'] = clusterer.fit_predict(pd.DataFrame(coords))
    return df

