# src/insights.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def identify_spending_patterns(df):
    """
    Identifica patrones de gasto por zona geográfica y categoría
    """
    # Patrones por categoría y hora del día
    time_patterns = df.groupby(['merchant_category', 'time_of_day']).agg(
        avg_transaction=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    
    # Patrones por categoría y día de la semana
    week_patterns = df.groupby(['merchant_category', 'day_of_week']).agg(
        avg_transaction=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()
    
    # Patrones por zona (si hay columna de cluster)
    if 'cluster' in df.columns:
        zone_patterns = df.groupby(['cluster', 'merchant_category']).agg(
            avg_transaction=('amount', 'mean'),
            transaction_count=('amount', 'count')
        ).reset_index()
    else:
        zone_patterns = pd.DataFrame()
    
    return {
        'time_patterns': time_patterns,
        'week_patterns': week_patterns,
        'zone_patterns': zone_patterns
    }

def generate_user_insights(df):
    """
    Genera insights personalizados sobre los hábitos de gasto del usuario
    """
    insights = []
    
    # Insight sobre categoría de mayor gasto
    top_category = df.groupby('merchant_category')['amount'].sum().idxmax()
    top_category_amount = df.groupby('merchant_category')['amount'].sum().max()
    total_amount = df['amount'].sum()
    percentage = (top_category_amount / total_amount) * 100
    
    insights.append(f"Tu categoría principal de gasto es {top_category}, representando el {percentage:.1f}% de tus gastos.")
    
    # Insight sobre momento de mayor gasto
    if 'time_of_day' in df.columns:
        expensive_time = df.groupby('time_of_day')['amount'].mean().idxmax()
        expensive_time_avg = df.groupby('time_of_day')['amount'].mean().max()
        overall_avg = df['amount'].mean()
        
        percentage_diff = ((expensive_time_avg - overall_avg) / overall_avg) * 100
        
        insights.append(f"Gastas en promedio un {percentage_diff:.1f}% más durante la {expensive_time}.")
    
    # Insight sobre zonas si hay clusters
    if 'cluster' in df.columns:
        valid_clusters = df[df['cluster'] != -1]
        if not valid_clusters.empty:
            expensive_zone = valid_clusters.groupby('cluster')['amount'].mean().idxmax()
            cheapest_zone = valid_clusters.groupby('cluster')['amount'].mean().idxmin()
            
            if expensive_zone != cheapest_zone:
                exp_zone_avg = valid_clusters[valid_clusters['cluster'] == expensive_zone]['amount'].mean()
                cheap_zone_avg = valid_clusters[valid_clusters['cluster'] == cheapest_zone]['amount'].mean()
                
                diff_percentage = ((exp_zone_avg - cheap_zone_avg) / cheap_zone_avg) * 100
                
                # Obtener categoría principal por zona
                exp_zone_cat = valid_clusters[valid_clusters['cluster'] == expensive_zone]['merchant_category'].value_counts().idxmax()
                cheap_zone_cat = valid_clusters[valid_clusters['cluster'] == cheapest_zone]['merchant_category'].value_counts().idxmax()
                
                insights.append(f"Gastas un {diff_percentage:.1f}% más en la zona {expensive_zone} (principalmente en {exp_zone_cat}) que en la zona {cheapest_zone} (donde sueles gastar en {cheap_zone_cat}).")
    
    return insights

def find_alternative_merchants(df):
    """
    Encuentra alternativas más baratas para comercios frecuentes
    """
    recommendations = []
    
    # Agrupamos por merchant_name para encontrar comercios frecuentes
    if 'merchant_name' in df.columns:
        merchants = df.groupby('merchant_name').agg(
            avg_amount=('amount', 'mean'),
            frequency=('amount', 'count'),
            category=('merchant_category', 'first'),
            avg_lat=('latitude', 'mean'),
            avg_lon=('longitude', 'mean')
        ).reset_index()
        
        # Filtramos solo comercios frecuentes (más de 3 transacciones)
        frequent_merchants = merchants[merchants['frequency'] > 3]
        
        for _, merchant in frequent_merchants.iterrows():
            # Buscamos alternativas en la misma categoría
            same_category = merchants[
                (merchants['category'] == merchant['category']) &
                (merchants['merchant_name'] != merchant['merchant_name']) &
                (merchants['avg_amount'] < merchant['avg_amount'])
            ]
            
            if not same_category.empty:
                # Calculamos distancias
                same_category['distance'] = same_category.apply(
                    lambda row: np.sqrt(
                        (row['avg_lat'] - merchant['avg_lat'])**2 + 
                        (row['avg_lon'] - merchant['avg_lon'])**2
                    ) * 111,  # Convertir a km aproximadamente
                    axis=1
                )
                
                # Filtrar por cercanía (menos de 2 km)
                nearby = same_category[same_category['distance'] < 2]
                
                if not nearby.empty:
                    # Ordenar por precio
                    alternatives = nearby.sort_values('avg_amount')
                    
                    best_alt = alternatives.iloc[0]
                    savings = merchant['avg_amount'] - best_alt['avg_amount']
                    savings_pct = (savings / merchant['avg_amount']) * 100
                    
                    recommendations.append({
                        'original_merchant': merchant['merchant_name'],
                        'alternative': best_alt['merchant_name'],
                        'category': merchant['category'],
                        'savings_percent': savings_pct,
                        'distance_km': best_alt['distance'],
                        'avg_saving': savings
                    })
    
    return recommendations