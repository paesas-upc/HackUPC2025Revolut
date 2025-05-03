# src/app.py

import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import numpy as np
import random
import colorsys
from datetime import datetime, timedelta
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import load_and_clean_data, aggregate_by_location, aggregate_by_time_location
from geocluster import cluster_locations, analyze_clusters
from insights import identify_spending_patterns, generate_user_insights, find_alternative_merchants

# Definir la función get_distinct_colors a nivel global para que esté disponible en todo el script
def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2)  # Alternar entre 0.7 y 1.0
        lightness = 0.4 + 0.1 * (i % 3)   # Variar entre 0.4 y 0.6
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append('#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)))
    return colors

st.set_page_config(layout="wide", page_title="Spending Map AI", page_icon="🗺️")

# Sidebar - configuraciones
st.sidebar.title("🔍 Spending Map AI")
st.sidebar.info("Visualiza y analiza tus gastos geográficamente")

# Opciones de visualización
view_option = st.sidebar.radio(
    "Selecciona una vista:",
    ["Mapa de gastos", "Análisis por zona", "Recomendaciones", "Predicciones"]
)

# Carga de datos desde data/transactions.csv
df = load_and_clean_data()

# Aplicar filtros si se desea
# Filtros de tiempo
st.sidebar.subheader("Filtros")
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
    min_value=df["timestamp"].min().date(),
    max_value=df["timestamp"].max().date()
)

# Filtrar por rango de fechas
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered_df = df[mask]
else:
    filtered_df = df

# Filtrar por categoría
if "merchant_category" in filtered_df.columns:
    all_categories = ["Todas"] + sorted(filtered_df["merchant_category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Categoría", all_categories)
    
    if selected_category != "Todas":
        filtered_df = filtered_df[filtered_df["merchant_category"] == selected_category]

# Aplicar clustering geográfico
clustered_df = cluster_locations(filtered_df, min_cluster_size=3)
cluster_stats = analyze_clusters(clustered_df)

# Vista principal según selección
if view_option == "Mapa de gastos":
    st.title("🗺️ Mapa de gastos")
    
    # Tabs para diferentes tipos de visualización
    map_tab, heatmap_tab, stats_tab = st.tabs(["Mapa de marcadores", "Mapa de calor", "Estadísticas"])
    
    with map_tab:
        # Agregación de transacciones por ubicación y categoría
        agg = aggregate_by_location(filtered_df)
        
        # Mapa base
        center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
        m = folium.Map(location=center, zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)
        
        # Colores por categoría
        categories = filtered_df['merchant_category'].unique()
        
        cat_colors = {cat: color for cat, color in zip(categories, get_distinct_colors(len(categories)))}
        
        # Añadir marcadores por grupo
        for _, row in agg.iterrows():
            popup = f"""
            <b>Categoría:</b> {row['merchant_category']}<br>
            <b>Total gastado:</b> ${row['total_spent']:.2f}<br>
            <b>Gasto promedio:</b> ${row['avg_spent']:.2f}<br>
            <b>Transacciones:</b> {row['transaction_count']}
            """
            
            # Determinar color basado en categoría
            color = cat_colors.get(row['merchant_category'], 'blue')
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup,
                icon=folium.Icon(color="blue", icon="info-sign"),
                tooltip=f"{row['merchant_category']}: ${row['total_spent']:.2f}"
            ).add_to(marker_cluster)
        
        # Mostrar el mapa
        st.write("Cada marcador representa un punto de gasto. Haz clic para ver más detalles.")
        folium_static(m)
    
    with heatmap_tab:
        st.write("Mapa de calor que muestra la intensidad de gasto por zona")
        
        # Mapa de calor basado en cantidad de gasto
        heat_data = [[row['latitude'], row['longitude'], row['amount']] for _, row in filtered_df.iterrows()]
        
        # Mapa base
        center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
        heat_map = folium.Map(location=center, zoom_start=13)
        
        # Añadir capa de calor
        HeatMap(heat_data, radius=15).add_to(heat_map)
        
        # Mostrar el mapa
        folium_static(heat_map)
    
    with stats_tab:
        st.subheader("Estadísticas de gastos")
        
        # Gráfico de barras por categoría
        cat_spending = filtered_df.groupby('merchant_category')['amount'].sum().reset_index()
        cat_spending = cat_spending.sort_values('amount', ascending=False)
        
        fig = px.bar(
            cat_spending,
            x='merchant_category',
            y='amount',
            title='Gasto total por categoría',
            labels={'merchant_category': 'Categoría', 'amount': 'Monto total ($)'}
        )
        st.plotly_chart(fig)
        
        # Gráfico de líneas por tiempo (si hay suficientes datos)
        if len(filtered_df['date'].unique()) > 1:
            time_spending = filtered_df.groupby(['date'])['amount'].sum().reset_index()
            
            fig_time = px.line(
                time_spending,
                x='date',
                y='amount',
                title='Evolución temporal del gasto',
                labels={'date': 'Fecha', 'amount': 'Monto total ($)'}
            )
            st.plotly_chart(fig_time)

elif view_option == "Análisis por zona":
    st.title("🔍 Análisis por zona geográfica")
    
    if clustered_df['cluster'].nunique() > 1:
        st.write(f"Se han identificado {clustered_df[clustered_df['cluster'] >= 0]['cluster'].nunique()} zonas distintas de gasto.")
        
        # Mostrar mapa con clusters coloreados
        center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
        cluster_map = folium.Map(location=center, zoom_start=12)
        
        # Colores para los clusters
        cluster_ids = sorted([c for c in clustered_df['cluster'].unique() if c >= 0])
        n_clusters = len(cluster_ids)
        colors = get_distinct_colors(n_clusters)  # Ahora usamos la función definida a nivel global
        cluster_colors = {cid: colors[i] for i, cid in enumerate(cluster_ids)}
        
        # Agrupar puntos por cluster
        for cluster_id in cluster_ids:
            cluster_points = clustered_df[clustered_df['cluster'] == cluster_id]
            
            # Calcular centro del cluster
            cluster_center = [
                cluster_points['latitude'].mean(), 
                cluster_points['longitude'].mean()
            ]
            
            # Estadísticas del cluster
            total_spent = cluster_points['amount'].sum()
            avg_spent = cluster_points['amount'].mean()
            transaction_count = len(cluster_points)
            top_category = cluster_points['merchant_category'].value_counts().index[0]
            
            # Añadir círculo para el cluster
            folium.Circle(
                location=cluster_center,
                radius=100,  # metros
                color=cluster_colors.get(cluster_id, 'gray'),
                fill=True,
                fill_opacity=0.4,
                tooltip=f"Zona {cluster_id}: ${total_spent:.2f} ({transaction_count} transacciones)"
            ).add_to(cluster_map)
            
            # Añadir etiqueta
            folium.Marker(
                location=cluster_center,
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">Zona {cluster_id}</div>'
                )
            ).add_to(cluster_map)
        
        # Mostrar el mapa
        st.write("Zonas identificadas mediante agrupamiento geográfico:")
        folium_static(cluster_map)
        
        # Análisis comparativo entre zonas
        st.subheader("Comparativa entre zonas")
        
        # Preparar datos para el gráfico
        compare_data = cluster_stats[['cluster', 'avg_spent', 'total_spent', 'transaction_count', 'dominant_category']]
        compare_data = compare_data.sort_values('total_spent', ascending=False)
        
        # Gráfico de barras comparativo
        fig = px.bar(
            compare_data,
            x='cluster',
            y='total_spent',
            color='dominant_category',
            title='Gasto total por zona',
            labels={'cluster': 'Zona', 'total_spent': 'Monto total ($)', 'dominant_category': 'Categoría dominante'}
        )
        st.plotly_chart(fig)
        
        # Análisis detallado de una zona específica
        st.subheader("Análisis detallado por zona")
        
        selected_cluster = st.selectbox(
            "Selecciona una zona para analizar en detalle:",
            options=cluster_ids,
            format_func=lambda x: f"Zona {x}"
        )
        
        # Filtrar datos para el cluster seleccionado
        cluster_data = clustered_df[clustered_df['cluster'] == selected_cluster]
        
        # Estadísticas del cluster
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total gastado", f"${cluster_data['amount'].sum():.2f}")
        with col2:
            st.metric("Transacciones", f"{len(cluster_data)}")
        with col3:
            st.metric("Gasto promedio", f"${cluster_data['amount'].mean():.2f}")
        
        # Distribución de categorías en el cluster
        cat_dist = cluster_data['merchant_category'].value_counts().reset_index()
        cat_dist.columns = ['merchant_category', 'count']
        
        fig_pie = px.pie(
            cat_dist,
            names='merchant_category',
            values='count',
            title=f'Distribución de categorías en Zona {selected_cluster}'
        )
        st.plotly_chart(fig_pie)
        
        # Evolución temporal si hay suficientes datos
        if len(cluster_data['date'].unique()) > 1:
            time_data = cluster_data.groupby('date')['amount'].sum().reset_index()
            
            fig_time = px.line(
                time_data,
                x='date',
                y='amount',
                title=f'Evolución temporal del gasto en Zona {selected_cluster}',
                labels={'date': 'Fecha', 'amount': 'Monto total ($)'}
            )
            st.plotly_chart(fig_time)
        
        # Insights sobre la zona
        st.subheader("Insights de la zona")
        
        # Comparar con el promedio global
        avg_global = filtered_df['amount'].mean()
        avg_cluster = cluster_data['amount'].mean()
        diff_pct = ((avg_cluster - avg_global) / avg_global) * 100
        
        if diff_pct > 0:
            st.info(f"En esta zona gastas en promedio un {diff_pct:.1f}% más que tu promedio global.")
        else:
            st.info(f"En esta zona gastas en promedio un {abs(diff_pct):.1f}% menos que tu promedio global.")
        
        # Identificar momento del día con mayor gasto en esta zona
        if 'time_of_day' in cluster_data.columns:
            time_spending = cluster_data.groupby('time_of_day')['amount'].mean()
            peak_time = time_spending.idxmax()
            st.info(f"En esta zona, tiendes a gastar más durante la {peak_time}.")

elif view_option == "Recomendaciones":
    st.title("💡 Recomendaciones personalizadas")
    
    # Generar insights generales
    insights = generate_user_insights(filtered_df)
    
    st.subheader("Insights sobre tus hábitos de gasto")
    for insight in insights:
        st.info(insight)
    
    # Buscar alternativas más baratas
    st.subheader("Alternativas de ahorro")
    recommendations = find_alternative_merchants(filtered_df)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.write(f"**Recomendación {i+1}:** En lugar de {rec['original_merchant']}, considera ir a {rec['alternative']} que está a {rec['distance_km']:.1f} km y podrías ahorrar aproximadamente un {rec['savings_percent']:.1f}%.")
    else:
        st.write("No se encontraron alternativas de ahorro basadas en tus datos actuales.")
    
    # Consejos basados en patrones
    st.subheader("Consejos basados en tus patrones")
    
    # Día de la semana más caro
    if 'day_of_week' in filtered_df.columns:
        day_spending = filtered_df.groupby('day_of_week')['amount'].mean()
        expensive_day = day_spending.idxmax()
        cheap_day = day_spending.idxmin()
        
        days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        
        st.write(f"Tu día más caro de la semana es el **{days[expensive_day]}** mientras que en el **{days[cheap_day]}** tiendes a gastar menos.")
        
        # Categorías por día
        expensive_day_cats = filtered_df[filtered_df['day_of_week'] == expensive_day]['merchant_category'].value_counts().head(2).index.tolist()
        st.write(f"El {days[expensive_day]} sueles gastar principalmente en: {', '.join(expensive_day_cats)}")

elif view_option == "Predicciones":
    st.title("🔮 Predicciones de gasto")
    
    st.info("Basado en tus datos históricos, podemos predecir patrones de gasto futuros.")
    
    # Simulaciones
    st.subheader("Simulaciones de escenarios")
    
    # Simulador de cambio de ubicación
    st.write("**¿Y si cambias tu ubicación habitual?**")
    
    # Lista de clusters/zonas detectadas
    if 'cluster' in clustered_df.columns and clustered_df['cluster'].nunique() > 1:
        valid_clusters = [c for c in clustered_df['cluster'].unique() if c >= 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_zone = st.selectbox(
                "Tu zona actual:",
                options=valid_clusters,
                format_func=lambda x: f"Zona {x}"
            )
        
        with col2:
            new_zone = st.selectbox(
                "Nueva zona potencial:",
                options=[z for z in valid_clusters if z != current_zone],
                format_func=lambda x: f"Zona {x}"
            )
        
        # Comparativa de gastos entre zonas
        if current_zone != new_zone:
            current_data = clustered_df[clustered_df['cluster'] == current_zone]
            new_data = clustered_df[clustered_df['cluster'] == new_zone]
            
            # Gastos promedio
            current_avg = current_data['amount'].mean()
            new_avg = new_data['amount'].mean()
            
            diff_pct = ((new_avg - current_avg) / current_avg) * 100
            
            # Categorías principales
            current_cats = current_data['merchant_category'].value_counts().head(3)
            new_cats = new_data['merchant_category'].value_counts().head(3)
            
            # Visualización comparativa
            if diff_pct > 0:
                st.warning(f"Si te mudaras de la Zona {current_zone} a la Zona {new_zone}, tu gasto promedio aumentaría aproximadamente un {diff_pct:.1f}%.")
            else:
                st.success(f"Si te mudaras de la Zona {current_zone} a la Zona {new_zone}, tu gasto promedio se reduciría aproximadamente un {abs(diff_pct):.1f}%.")
            
            # Comparativa de categorías
            st.write("**Comparativa de gastos por categoría:**")
            
            # Preparar datos para el gráfico
            current_cat_avg = current_data.groupby('merchant_category')['amount'].mean().reset_index()
            current_cat_avg['zone'] = f'Zona {current_zone} (actual)'
            
            new_cat_avg = new_data.groupby('merchant_category')['amount'].mean().reset_index()
            new_cat_avg['zone'] = f'Zona {new_zone} (potencial)'
            
            # Unir los datos
            compare_df = pd.concat([current_cat_avg, new_cat_avg])
            
            # Gráfico comparativo
            fig = px.bar(
                compare_df,
                x='merchant_category',
                y='amount',
                color='zone',
                barmode='group',
                title='Comparativa de gasto promedio por categoría entre zonas',
                labels={'merchant_category': 'Categoría', 'amount': 'Gasto promedio ($)', 'zone': 'Zona'}
            )
            st.plotly_chart(fig)
    else:
        st.write("No se han detectado suficientes zonas diferentes para hacer predicciones geográficas.")