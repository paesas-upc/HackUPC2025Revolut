import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import numpy as np

def calculate_distance(coords1, coords2):
    """
    Calcula la distancia en kilómetros entre dos pares de coordenadas
    usando la fórmula de Haversine (geodesic)
    """
    try:
        distance = geodesic(coords1, coords2).kilometers
        return distance
    except:
        return None

def identify_spending_patterns(df):
    # Patrones por categoría y hora del día
    time_patterns = df.groupby(['merchant_category', 'time_of_day']).agg(
        avg_transaction=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        avg_distance=('distance_to_merchant', 'mean')
    ).reset_index()
    
    # Patrones por categoría y día de la semana
    week_patterns = df.groupby(['merchant_category', 'day_of_week']).agg(
        avg_transaction=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        avg_distance=('distance_to_merchant', 'mean')
    ).reset_index()
    
    # Identificar categorías con mayor variación de precios
    price_variation = df.groupby(['merchant_category', 'merchant_name']).agg(
        avg_price=('amount', 'mean')
    ).reset_index()

    category_price_variation = price_variation.groupby('merchant_category').agg(
        price_variance=('avg_price', 'var'),
        min_price=('avg_price', 'min'),
        max_price=('avg_price', 'max'),
        price_range=('avg_price', lambda x: x.max() - x.min())
    ).reset_index()
    
    return {
        'time_patterns': time_patterns,
        'week_patterns': week_patterns,
        'price_variation': category_price_variation
    }

def generate_user_insights(df):
    """
    Genera insights personalizados para el usuario basados en sus datos
    """
    insights = []
    
    # Verificar que hay suficientes datos
    if len(df) < 5:
        insights.append("No tenemos suficientes datos para generar insights personalizados. ¡Continúa utilizando la app para obtener recomendaciones más precisas!")
        return insights
    
    # Insight sobre distancias
    if 'distance_to_merchant' in df.columns:
        avg_distance = df['distance_to_merchant'].mean()
        if not pd.isna(avg_distance):
            insights.append(f"En promedio, te desplazas {avg_distance:.1f} km para realizar tus compras.")
            
            # Categorías con mayor distancia
            cat_dist = df.groupby('merchant_category')['distance_to_merchant'].mean().sort_values(ascending=False)
            if not cat_dist.empty:
                furthest_cat = cat_dist.index[0]
                furthest_dist = cat_dist.iloc[0]
                if not pd.isna(furthest_dist) and furthest_dist > 0:
                    insights.append(f"Te desplazas más lejos para '{furthest_cat}', con un promedio de {furthest_dist:.1f} km.")
    
    # Insight sobre categorías más frecuentes
    if len(df) >= 10:
        top_categories = df['merchant_category'].value_counts().head(3)
        if not top_categories.empty:
            top_cat = top_categories.index[0]
            top_count = top_categories.iloc[0]
            insights.append(f"Tu categoría de gasto más frecuente es '{top_cat}' con {top_count} transacciones.")
    
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
                
                insights.append(f"Gastas en promedio un {diff_percentage:.1f}% más en la zona {expensive_zone} comparado con la zona {cheapest_zone}.")
    
    # Insight sobre días más caros
    if 'day_of_week' in df.columns:
        day_spending = df.groupby('day_of_week')['amount'].mean()
        expensive_day = day_spending.idxmax()
        cheapest_day = day_spending.idxmin()
        
        days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        
        diff_day_percentage = ((day_spending.max() - day_spending.min()) / day_spending.min()) * 100
        
        if diff_day_percentage > 20:  # Solo si la diferencia es significativa
            insights.append(f"Tus gastos son en promedio un {diff_day_percentage:.1f}% más altos el {days[expensive_day]} comparado con el {days[cheapest_day]}.")
    
    # Insight sobre momento del día
    if 'time_of_day' in df.columns:
        time_spending = df.groupby('time_of_day')['amount'].mean()
        expensive_time = time_spending.idxmax()
        
        insights.append(f"Tiendes a gastar más durante la {expensive_time}.")
    
    return insights

def find_alternative_merchants(df):
    """
    Encuentra comercios alternativos que podrían ofrecer mejores precios
    """
    if len(df) < 10:
        return []
    
    # Asegurarse de que tenemos las columnas necesarias
    required_cols = ['merchant_name', 'merchant_category', 'amount', 
                     'merchant_latitude', 'merchant_longitude',
                     'latitude', 'longitude']
    
    if not all(col in df.columns for col in required_cols):
        return []
    
    # Encontrar comercios frecuentes (al menos 2 visitas)
    merchant_counts = df['merchant_name'].value_counts()
    frequent_merchants = merchant_counts[merchant_counts >= 2].index.tolist()
    
    # Si no hay suficientes comercios frecuentes, no podemos hacer recomendaciones
    if len(frequent_merchants) < 2:
        return []
    
    # Agrupar datos por comerciante para obtener precio promedio y ubicación
    merchants = df.groupby(['merchant_name', 'merchant_category']).agg(
        avg_amount=('amount', 'mean'),
        visit_count=('amount', 'count'),
        avg_lat=('merchant_latitude', 'mean'),
        avg_lon=('merchant_longitude', 'mean')
    ).reset_index()
    
    # Filtrar para obtener solo comercios frecuentes
    frequent_merchants_data = merchants[merchants['merchant_name'].isin(frequent_merchants)]
    
    recommendations = []
    
    # Por cada comercio frecuente, buscar alternativas más baratas en la misma categoría
    for _, merchant in frequent_merchants_data.iterrows():
        # Buscamos alternativas en la misma categoría
        same_category = merchants[
            (merchants['merchant_category'] == merchant['merchant_category']) &
            (merchants['merchant_name'] != merchant['merchant_name']) &
            (merchants['avg_amount'] < merchant['avg_amount'])
        ]
        
        if not same_category.empty:
            # Calculamos distancias desde el comercio original a las alternativas
            same_category['distance'] = same_category.apply(
                lambda row: calculate_distance(
                    (merchant['avg_lat'], merchant['avg_lon']),
                    (row['avg_lat'], row['avg_lon'])
                ),
                axis=1
            )
            
            # Filtrar por distancia máxima (2 km) y ordenar por precio
            nearby_alternatives = same_category[same_category['distance'] <= 2].sort_values('avg_amount')
            
            if not nearby_alternatives.empty:
                best_alternative = nearby_alternatives.iloc[0]
                savings_percent = ((merchant['avg_amount'] - best_alternative['avg_amount']) / merchant['avg_amount']) * 100
                
                # Solo recomendar si el ahorro es significativo (>10%)
                if savings_percent > 10:
                    recommendations.append({
                        'original_merchant': merchant['merchant_name'],
                        'alternative': best_alternative['merchant_name'],
                        'category': merchant['merchant_category'],
                        'original_price': merchant['avg_amount'],
                        'alternative_price': best_alternative['avg_amount'],
                        'savings_percent': savings_percent,
                        'distance_km': best_alternative['distance'],
                        'original_coords': (merchant['avg_lat'], merchant['avg_lon']),
                        'alternative_coords': (best_alternative['avg_lat'], best_alternative['avg_lon'])
                    })
    
    # Ordenar por porcentaje de ahorro
    recommendations.sort(key=lambda x: x['savings_percent'], reverse=True)
    
    return recommendations

def find_closest_alternatives(df, user_lat, user_lon, category=None, max_distance=5.0):
    """
    Encuentra los comercios más cercanos al usuario, opcionalmente filtrado por categoría,
    y los ordena considerando tanto distancia como precio
    
    Args:
        df: DataFrame con datos de transacciones
        user_lat: Latitud del usuario
        user_lon: Longitud del usuario
        category: Categoría opcional para filtrar (default: None)
        max_distance: Distancia máxima en km a considerar
        
    Returns:
        DataFrame con comercios cercanos ordenados por un score combinado de precio y distancia
    """
    # Verificar que tenemos datos suficientes
    if df.empty:
        return pd.DataFrame()
        
    # Filtrar por categoría si se especifica
    category_df = df if category is None else df[df['merchant_category'] == category]
    
    if category_df.empty:
        return pd.DataFrame()
    
    # Agrupar por comercio para obtener precio promedio y ubicación
    merchants = category_df.groupby(['merchant_name', 'merchant_category']).agg(
        avg_price=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        merchant_lat=('merchant_latitude', 'mean'),
        merchant_lon=('merchant_longitude', 'mean')
    ).reset_index()
    
    # Calcular distancia desde la ubicación del usuario
    merchants['distance_from_user'] = merchants.apply(
        lambda row: calculate_distance(
            (user_lat, user_lon),
            (row['merchant_lat'], row['merchant_lon'])
        ),
        axis=1
    )
    
    # Filtrar por distancia máxima
    nearby = merchants[merchants['distance_from_user'] <= max_distance]
    
    if nearby.empty:
        return pd.DataFrame()
    
    # Normalizar precio y distancia para combinarlos en un score
    if len(nearby) > 1:  # Solo normalizar si hay más de un comercio
        nearby['price_norm'] = (nearby['avg_price'] - nearby['avg_price'].min()) / (nearby['avg_price'].max() - nearby['avg_price'].min())
        nearby['distance_norm'] = (nearby['distance_from_user'] - nearby['distance_from_user'].min()) / (nearby['distance_from_user'].max() - nearby['distance_from_user'].min())
        # Score combinado (60% precio, 40% distancia)
        nearby['combined_score'] = 0.6 * nearby['price_norm'] + 0.4 * nearby['distance_norm']
    else:
        # Si solo hay un comercio, asignar score 0
        nearby['price_norm'] = 0
        nearby['distance_norm'] = 0
        nearby['combined_score'] = 0
    
    # Ordenar por score combinado (menor es mejor)
    result = nearby.sort_values('combined_score')
    
    return result