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
from insights import identify_spending_patterns, generate_user_insights, find_alternative_merchants, find_closest_alternatives

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

def main(user_first_name=None, user_last_name=None):
    """
    Función principal de la aplicación que puede ser llamada con información de usuario
    
    Args:
        user_first_name: Nombre del usuario
        user_last_name: Apellido del usuario
    """
    # La configuración de página se debe mover al bloque principal al final del archivo
    # y eliminarla de aquí

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
    
    # Filtrar datos por usuario si se especificó
    if user_first_name and user_last_name:
        # Filtrar por nombre y apellido
        user_mask = (
            (df["first"].str.lower() == user_first_name.lower()) & 
            (df["last"].str.lower() == user_last_name.lower())
        )
        user_df = df[user_mask]
        
        # Verificar si hay datos para este usuario
        if len(user_df) == 0:
            st.error(f"No se encontraron transacciones para {user_first_name} {user_last_name}")
            return
            
        # Mostrar información de perfil
        st.title(f"👋 Hola, {user_first_name.title()} {user_last_name.title()}")
        
        # Guardamos todos los datos para la sección de predicciones
        all_data_df = df.copy()
        
        # Para el análisis normal, usamos solo los datos del usuario
        df = user_df
    else:
        # Si no hay usuario especificado, trabajamos con todos los datos
        all_data_df = df.copy()
        st.title("🗺️ Spending Map AI")

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
    if len(filtered_df) < 3:  # O cualquier umbral mínimo que quieras establecer
        st.warning(f"No hay suficientes datos para realizar un análisis detallado. Se encontraron {len(filtered_df)} transacciones.")
        # Asignar todos los puntos al mismo cluster
        filtered_df['cluster'] = 0
        clustered_df = filtered_df
    else:
        # Intentar clustering con una configuración más permisiva
        try:
            clustered_df = cluster_locations(filtered_df, min_cluster_size=min(3, len(filtered_df)))
        except Exception as e:
            st.error(f"Error al procesar clusters: {str(e)}")
            filtered_df['cluster'] = 0
            clustered_df = filtered_df
            
    cluster_stats = analyze_clusters(clustered_df)

    # Vista principal según selección
    if view_option == "Mapa de gastos":
        st.header("🗺️ Mapa de gastos")
        
        # Tabs para diferentes tipos de visualización
        map_tab, heatmap_tab, stats_tab = st.tabs(["Mapa de marcadores", "Mapa de calor", "Estadísticas"])
        
        with map_tab:
            st.subheader("Mapa de transacciones")
            
            # Opción para mostrar ubicaciones de usuario, comercio
            show_option = st.radio(
                "Mostrar ubicaciones de:",
                ["Usuario", "Comercios"],
                horizontal=True
            )
            
            # Verificar si tenemos coordenadas de comerciantes
            has_merchant_coords = ('merchant_latitude' in filtered_df.columns and 
                                filtered_df['merchant_latitude'].notna().any() and
                                'merchant_longitude' in filtered_df.columns and 
                                filtered_df['merchant_longitude'].notna().any())
            
            if not has_merchant_coords and show_option in ["Comercios"]:
                st.warning("No hay coordenadas de comercios disponibles en los datos.")
                show_option = "Usuario"
            
            # Mapa base
            center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            m = folium.Map(location=center, zoom_start=12)
            
            # Colores por categoría
            categories = filtered_df['merchant_category'].unique()
            cat_colors = {cat: color for cat, color in zip(categories, get_distinct_colors(len(categories)))}
            
            # Crear grupos de marcadores para usuario y comercios
            user_marker_cluster = MarkerCluster(name="Ubicaciones de usuario").add_to(m)
            merchant_marker_cluster = MarkerCluster(name="Ubicaciones de comercios").add_to(m)
            
            # Mostrar ubicaciones de usuario
            if show_option in ["Usuario"]:
                # Agregación por ubicación de usuario
                user_agg = aggregate_by_location(filtered_df, include_merchant=False)
                
                for _, row in user_agg.iterrows():
                    popup = f"""
                    <b>Categoría:</b> {row['merchant_category']}<br>
                    <b>Total gastado:</b> ${row['total_spent']:.2f}<br>
                    <b>Gasto promedio:</b> ${row['avg_spent']:.2f}<br>
                    <b>Transacciones:</b> {row['transaction_count']}
                    """
                    
                    # Color basado en categoría
                    color = cat_colors.get(row['merchant_category'], 'blue')
                    
                    # Usar un ícono de persona para el usuario
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=popup,
                        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
                        tooltip=f"Usuario - {row['merchant_category']}: ${row['total_spent']:.2f}"
                    ).add_to(user_marker_cluster)
            
            # Mostrar ubicaciones de comercios
            if show_option in ["Comercios"] and has_merchant_coords:
                # Agregación por ubicación de comercio
                merchant_agg = aggregate_by_location(filtered_df, include_merchant=True)
                
                for _, row in merchant_agg.iterrows():
                    popup = f"""
                    <b>Categoría:</b> {row['merchant_category']}<br>
                    <b>Total gastado:</b> ${row['total_spent']:.2f}<br>
                    <b>Gasto promedio:</b> ${row['avg_spent']:.2f}<br>
                    <b>Transacciones:</b> {row['transaction_count']}
                    """
                    
                    # Color basado en categoría
                    color = cat_colors.get(row['merchant_category'], 'red')
                    
                    # Usar un ícono de tienda para los comercios
                    folium.Marker(
                        location=[row['merchant_latitude'], row['merchant_longitude']],
                        popup=popup,
                        icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
                        tooltip=f"Comercio - {row['merchant_category']}: ${row['total_spent']:.2f}"
                    ).add_to(merchant_marker_cluster)
            
            # Añadir control de capas para activar/desactivar grupo de marcadores
            folium.LayerControl().add_to(m)
            
            # Mostrar el mapa
            st.write("El mapa muestra ubicaciones de usuarios (azul) y comercios (rojo). Las líneas conectan al usuario con los comercios donde realizó transacciones.")
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
        st.header("🔍 Análisis por zona geográfica")
        
        # Para esta sección, utilizamos todos los datos para el clustering
        # pero destacamos la información del usuario actual
        if user_first_name and user_last_name:
            # Aplicar clustering a todos los datos
            all_clustered_df = cluster_locations(all_data_df, min_cluster_size=5)
            
            # Identificar los clusters a los que pertenece el usuario
            user_mask = (
                (all_data_df["first"].str.lower() == user_first_name.lower()) & 
                (all_data_df["last"].str.lower() == user_last_name.lower())
            )
            user_raw_df = all_data_df[user_mask]
            
            # Asignar usuarios a clusters basados en la proximidad
            user_points = user_raw_df[['latitude', 'longitude']].values
            all_clusters = all_clustered_df[all_clustered_df['cluster'] >= 0]
            
            if len(user_points) > 0 and not all_clusters.empty:
                user_clusters = set()
                for _, user_point in enumerate(user_points):
                    # Buscar el cluster más cercano para cada punto del usuario
                    min_dist = float('inf')
                    closest_cluster = None
                    
                    for cluster_id in all_clusters['cluster'].unique():
                        cluster_points = all_clustered_df[all_clustered_df['cluster'] == cluster_id]
                        cluster_center = (
                            cluster_points['latitude'].mean(),
                            cluster_points['longitude'].mean()
                        )
                        
                        # Calcular distancia euclidiana simple
                        dist = np.sqrt(
                            (user_point[0] - cluster_center[0])**2 + 
                            (user_point[1] - cluster_center[1])**2
                        )
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_cluster = cluster_id
                    
                    if closest_cluster is not None:
                        user_clusters.add(closest_cluster)
                
                # Convertir a lista para usar más adelante
                user_clusters = sorted(list(user_clusters))
            else:
                user_clusters = []
            
            # Analizar los clusters
            cluster_stats = analyze_clusters(all_clustered_df)
            
            st.write(f"Se han identificado {all_clustered_df[all_clustered_df['cluster'] >= 0]['cluster'].nunique()} zonas distintas de gasto en toda la base de datos.")
            
            if user_clusters:
                st.write(f"Tus transacciones se concentran principalmente en {len(user_clusters)} zonas: {', '.join([f'Zona {c}' for c in user_clusters])}")
            
            # Mostrar mapa con clusters coloreados y datos del usuario destacados
            center = [all_data_df['latitude'].mean(), all_data_df['longitude'].mean()]
            cluster_map = folium.Map(location=center, zoom_start=12)
            
            # Colores para los clusters
            cluster_ids = sorted([c for c in all_clustered_df['cluster'].unique() if c >= 0])
            n_clusters = len(cluster_ids)
            colors = get_distinct_colors(n_clusters)
            cluster_colors = {cid: colors[i] for i, cid in enumerate(cluster_ids)}
            
            # Agrupar puntos por cluster
            for cluster_id in cluster_ids:
                cluster_points = all_clustered_df[all_clustered_df['cluster'] == cluster_id]
                
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
                
                # Obtener el color: más brillante si el usuario está en ese cluster
                cluster_color = cluster_colors.get(cluster_id, 'gray')
                fill_opacity = 0.6 if cluster_id in user_clusters else 0.3
                
                # Añadir círculo para el cluster
                folium.Circle(
                    location=cluster_center,
                    radius=100,  # metros
                    color=cluster_color,
                    fill=True,
                    fill_opacity=fill_opacity,
                    tooltip=f"Zona {cluster_id}: ${total_spent:.2f} ({transaction_count} transacciones)"
                ).add_to(cluster_map)
                
                # Añadir etiqueta
                folium.Marker(
                    location=cluster_center,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">Zona {cluster_id}</div>'
                    )
                ).add_to(cluster_map)
            
            # Si hay usuario, añadir sus puntos al mapa
            if user_first_name and len(user_raw_df) > 0:
                user_points_group = folium.FeatureGroup(name="Tus ubicaciones")
                
                for _, row in user_raw_df.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=4,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.7,
                        tooltip=f"Tu transacción: ${row['amount']:.2f} en {row['merchant_category']}"
                    ).add_to(user_points_group)
                
                user_points_group.add_to(cluster_map)
            
            # Añadir control de capas
            folium.LayerControl().add_to(cluster_map)
            
            # Mostrar el mapa
            st.write("Zonas identificadas mediante agrupamiento geográfico:")
            st.write("Las zonas donde tienes actividad están resaltadas con mayor opacidad. Los puntos azules representan tus ubicaciones.")
            folium_static(cluster_map)
            
            # Análisis comparativo entre zonas
            st.subheader("Comparativa entre zonas")
            
            # Preparar datos para el gráfico
            compare_data = cluster_stats[['cluster', 'avg_spent', 'total_spent', 'transaction_count', 'dominant_category']]
            compare_data = compare_data.sort_values('total_spent', ascending=False)
            
            # Añadir columna que indique si el usuario está en esa zona
            compare_data['usuario_presente'] = compare_data['cluster'].apply(lambda x: "Sí" if x in user_clusters else "No")
            
            # Gráfico de barras comparativo
            fig = px.bar(
                compare_data,
                x='cluster',
                y='total_spent',
                color='dominant_category',
                pattern_shape='usuario_presente',
                title='Gasto total por zona',
                labels={
                    'cluster': 'Zona', 
                    'total_spent': 'Monto total ($)', 
                    'dominant_category': 'Categoría dominante',
                    'usuario_presente': 'Tu actividad'
                }
            )
            st.plotly_chart(fig)
            
            # Análisis detallado de una zona específica
            st.subheader("Análisis detallado por zona")
            
            # Priorizar las zonas donde el usuario tiene actividad para el selectbox
            if user_clusters:
                all_cluster_options = user_clusters + [c for c in cluster_ids if c not in user_clusters]
                default_cluster = user_clusters[0] if user_clusters else cluster_ids[0]
            else:
                all_cluster_options = cluster_ids
                default_cluster = cluster_ids[0] if cluster_ids else 0
            
            selected_cluster = st.selectbox(
                "Selecciona una zona para analizar en detalle:",
                options=all_cluster_options,
                index=0,
                format_func=lambda x: f"Zona {x}" + (" (con tu actividad)" if x in user_clusters else "")
            )
            
            # Filtrar datos para el cluster seleccionado de todos los usuarios
            cluster_data = all_clustered_df[all_clustered_df['cluster'] == selected_cluster]
            
            # Datos del usuario en este cluster (si existe)
            if user_first_name:
                user_cluster_data = cluster_data[
                    (cluster_data['first'].str.lower() == user_first_name.lower()) &
                    (cluster_data['last'].str.lower() == user_last_name.lower())
                ]
                has_user_data = not user_cluster_data.empty
            else:
                user_cluster_data = pd.DataFrame()
                has_user_data = False
            
            # Estadísticas del cluster
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total gastado (todos)", f"${cluster_data['amount'].sum():.2f}")
            with col2:
                st.metric("Transacciones (todos)", f"{len(cluster_data)}")
            with col3:
                st.metric("Gasto promedio (todos)", f"${cluster_data['amount'].mean():.2f}")
            
            # Si hay datos del usuario, mostrar comparativa
            if has_user_data:
                st.subheader(f"Tu actividad en la Zona {selected_cluster}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    user_total = user_cluster_data['amount'].sum()
                    pct_total = (user_total / cluster_data['amount'].sum()) * 100
                    st.metric("Tu gasto total", f"${user_total:.2f}", f"{pct_total:.1f}% del total")
                    
                with col2:
                    user_count = len(user_cluster_data)
                    pct_count = (user_count / len(cluster_data)) * 100
                    st.metric("Tus transacciones", f"{user_count}", f"{pct_count:.1f}% del total")
                    
                with col3:
                    user_avg = user_cluster_data['amount'].mean()
                    avg_diff = ((user_avg / cluster_data['amount'].mean()) - 1) * 100
                    st.metric("Tu gasto promedio", f"${user_avg:.2f}", f"{avg_diff:+.1f}% vs promedio")
            
            # Distribución de categorías en el cluster
            st.subheader(f"Categorías en la Zona {selected_cluster}")
            
            # Para todos los usuarios
            cat_dist = cluster_data['merchant_category'].value_counts().reset_index()
            cat_dist.columns = ['merchant_category', 'count']
            
            # Para el usuario actual (si hay datos)
            if has_user_data:
                user_cat_dist = user_cluster_data['merchant_category'].value_counts().reset_index()
                user_cat_dist.columns = ['merchant_category', 'count']
                
                # Combinar los dataframes
                cat_dist['tipo'] = 'Todos los usuarios'
                user_cat_dist['tipo'] = 'Tus transacciones'
                combined_cat_dist = pd.concat([cat_dist, user_cat_dist])
                
                # Gráfico combinado
                fig_pie = px.pie(
                    combined_cat_dist,
                    names='merchant_category',
                    values='count',
                    facet_col='tipo',
                    title=f'Distribución de categorías en Zona {selected_cluster}'
                )
                st.plotly_chart(fig_pie)
            else:
                # Solo datos generales
                fig_pie = px.pie(
                    cat_dist,
                    names='merchant_category',
                    values='count',
                    title=f'Distribución de categorías en Zona {selected_cluster}'
                )
                st.plotly_chart(fig_pie)
            
            # Evolución temporal si hay suficientes datos
            if len(cluster_data['date'].unique()) > 1:
                st.subheader(f"Evolución temporal del gasto en Zona {selected_cluster}")
                
                # Datos de todos
                time_data = cluster_data.groupby('date')['amount'].sum().reset_index()
                time_data['tipo'] = 'Todos los usuarios'
                
                if has_user_data and len(user_cluster_data['date'].unique()) > 1:
                    # Datos del usuario
                    user_time_data = user_cluster_data.groupby('date')['amount'].sum().reset_index()
                    user_time_data['tipo'] = 'Tus transacciones'
                    
                    # Combinar
                    combined_time_data = pd.concat([time_data, user_time_data])
                    
                    fig_time = px.line(
                        combined_time_data,
                        x='date',
                        y='amount',
                        color='tipo',
                        title=f'Evolución temporal del gasto en Zona {selected_cluster}',
                        labels={'date': 'Fecha', 'amount': 'Monto total ($)', 'tipo': ''}
                    )
                    st.plotly_chart(fig_time)
                else:
                    # Solo datos generales
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
            avg_global = all_data_df['amount'].mean()
            avg_cluster = cluster_data['amount'].mean()
            diff_pct = ((avg_cluster - avg_global) / avg_global) * 100
            
            if diff_pct > 0:
                st.info(f"En esta zona se gasta en promedio un {diff_pct:.1f}% más que el promedio global.")
            else:
                st.info(f"En esta zona se gasta en promedio un {abs(diff_pct):.1f}% menos que el promedio global.")
            
            # Si hay datos del usuario, comparar con el promedio de la zona
            if has_user_data:
                user_avg = user_cluster_data['amount'].mean()
                user_zone_diff = ((user_avg - avg_cluster) / avg_cluster) * 100
                
                if user_zone_diff > 0:
                    st.info(f"Tu gasto promedio en esta zona es un {user_zone_diff:.1f}% mayor que el de otros usuarios.")
                else:
                    st.info(f"Tu gasto promedio en esta zona es un {abs(user_zone_diff):.1f}% menor que el de otros usuarios.")
            
            # Identificar momento del día con mayor gasto en esta zona
            if 'time_of_day' in cluster_data.columns:
                time_spending = cluster_data.groupby('time_of_day')['amount'].mean()
                peak_time = time_spending.idxmax()
                st.info(f"En esta zona, los usuarios tienden a gastar más durante la {peak_time}.")
                
                if has_user_data and 'time_of_day' in user_cluster_data.columns:
                    user_time_spending = user_cluster_data.groupby('time_of_day')['amount'].mean()
                    if not user_time_spending.empty:
                        user_peak_time = user_time_spending.idxmax()
                        st.info(f"Tú en particular tiendes a gastar más en esta zona durante la {user_peak_time}.")
                    
            # Análisis de distancia si hay datos disponibles
            if 'distance_to_merchant' in all_clustered_df.columns:
                st.subheader("Análisis de desplazamiento")
                
                # Calcular distancia promedio por cluster
                if 'distance_to_merchant' in cluster_data.columns:
                    avg_distance = cluster_data['distance_to_merchant'].mean()
                    if not pd.isna(avg_distance):
                        st.metric("Distancia promedio a comercios", f"{avg_distance:.2f} km")
                        
                        if has_user_data and 'distance_to_merchant' in user_cluster_data.columns:
                            user_avg_distance = user_cluster_data['distance_to_merchant'].mean()
                            if not pd.isna(user_avg_distance):
                                dist_diff = user_avg_distance - avg_distance
                                st.metric(
                                    "Tu distancia promedio", 
                                    f"{user_avg_distance:.2f} km",
                                    f"{dist_diff:+.2f} km vs promedio"
                                )
                    
                # Análisis de distancia por categoría
                st.write("**Distancia promedio por categoría:**")
                
                cat_distances = cluster_data.groupby('merchant_category')['distance_to_merchant'].mean().reset_index()
                cat_distances = cat_distances.sort_values('distance_to_merchant', ascending=False)
                
                if not cat_distances.empty and not cat_distances['distance_to_merchant'].isna().all():
                    fig_cat_dist = px.bar(
                        cat_distances,
                        x='merchant_category',
                        y='distance_to_merchant',
                        title='Distancia promedio recorrida por categoría',
                        labels={'merchant_category': 'Categoría', 'distance_to_merchant': 'Distancia promedio (km)'}
                    )
                    st.plotly_chart(fig_cat_dist)
        else:
            # Si no hay usuario específico, usamos el clustering normal
            if clustered_df['cluster'].nunique() > 1:
                st.write(f"Se han identificado {clustered_df[clustered_df['cluster'] >= 0]['cluster'].nunique()} zonas distintas de gasto.")
                
                # Mostrar mapa con clusters coloreados
                center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
                cluster_map = folium.Map(location=center, zoom_start=12)
                
                # Colores para los clusters
                cluster_ids = sorted([c for c in clustered_df['cluster'].unique() if c >= 0])
                n_clusters = len(cluster_ids)
                colors = get_distinct_colors(n_clusters)
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
                
                # Resto del código original para el análisis detallado de zonas...
                # (mantener el mismo código que ya tenías para el caso sin usuario específico)
                
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
                    
                # Análisis de distancia por zona
                if 'cluster' in clustered_df.columns and clustered_df['cluster'].nunique() > 1 and 'distance_to_merchant' in filtered_df.columns:
                    st.subheader("Análisis de desplazamiento")
                    
                    # Calcular distancia promedio por cluster
                    cluster_distances = clustered_df.groupby('cluster')['distance_to_merchant'].mean().reset_index()
                    cluster_distances = cluster_distances[cluster_distances['cluster'] >= 0]  # Filtrar ruido (-1)
                    
                    if not cluster_distances.empty and not cluster_distances['distance_to_merchant'].isna().all():
                        # Gráfico de barras de distancia por zona
                        fig_dist = px.bar(
                            cluster_distances,
                            x='cluster',
                            y='distance_to_merchant',
                            title='Distancia promedio a comercios por zona',
                            labels={'cluster': 'Zona', 'distance_to_merchant': 'Distancia promedio (km)'}
                        )
                        st.plotly_chart(fig_dist)
                        
                        # Identificar zona con mayor y menor distancia
                        max_dist_zone = cluster_distances.loc[cluster_distances['distance_to_merchant'].idxmax()]
                        min_dist_zone = cluster_distances.loc[cluster_distances['distance_to_merchant'].idxmin()]
                        
                        st.info(f"En la Zona {max_dist_zone['cluster']}, te desplazas en promedio {max_dist_zone['distance_to_merchant']:.2f} km para comprar, mientras que en la Zona {min_dist_zone['cluster']} solo te desplazas {min_dist_zone['distance_to_merchant']:.2f} km.")
                    
                    # Análisis de distancia por categoría
                    st.write("**Distancia promedio por categoría:**")
                    
                    cat_distances = filtered_df.groupby('merchant_category')['distance_to_merchant'].mean().reset_index()
                    cat_distances = cat_distances.sort_values('distance_to_merchant', ascending=False)
                    
                    if not cat_distances.empty and not cat_distances['distance_to_merchant'].isna().all():
                        fig_cat_dist = px.bar(
                            cat_distances,
                            x='merchant_category',
                            y='distance_to_merchant',
                            title='Distancia promedio recorrida por categoría',
                            labels={'merchant_category': 'Categoría', 'distance_to_merchant': 'Distancia promedio (km)'}
                        )
                        st.plotly_chart(fig_cat_dist)
            else:
                st.warning("No se han detectado suficientes zonas distintas con tus datos actuales. Prueba a ajustar los filtros o añadir más transacciones.")

    elif view_option == "Recomendaciones":
        st.header("💡 Recomendaciones personalizadas")
        
        # Generar insights generales
        insights = generate_user_insights(filtered_df)
        
        st.subheader("Insights sobre tus hábitos de gasto")
        for insight in insights:
            st.info(insight)
        
        st.subheader("Alternativas de ahorro por cercanía")

        # Obtenemos la ubicación promedio del usuario para las recomendaciones
        user_avg_lat = filtered_df['latitude'].mean()
        user_avg_lon = filtered_df['longitude'].mean()

        # Verificamos si tenemos datos de ubicación de comercios
        has_merchant_coords = ('merchant_latitude' in filtered_df.columns and 
                            'merchant_longitude' in filtered_df.columns and
                            filtered_df['merchant_latitude'].notna().any() and
                            filtered_df['merchant_longitude'].notna().any())

        if not has_merchant_coords:
            st.warning("No hay coordenadas de comercios disponibles en los datos para generar recomendaciones.")
        else:
            # Selector de categoría para buscar alternativas
            categories = ["Todas"] + sorted(filtered_df['merchant_category'].unique().tolist())
            selected_category = st.selectbox("Selecciona una categoría para encontrar alternativas:", categories)
            
            # Slider para distancia máxima
            max_distance = st.slider("Distancia máxima (km):", 0.5, 10.0, 5.0, 0.5)
            
            # Botón para buscar alternativas
            if st.button("Buscar alternativas cercanas"):
                with st.spinner("Buscando las mejores opciones..."):
                    if selected_category == "Todas":
                        # Para cada categoría, buscar la mejor alternativa
                        all_alternatives = []
                        for category in filtered_df['merchant_category'].unique():
                            alternatives = find_closest_alternatives(
                                filtered_df, user_avg_lat, user_avg_lon, 
                                category=category, max_distance=max_distance
                            )
                            if not alternatives.empty:
                                all_alternatives.append(alternatives.iloc[0])
                        
                        # Crear un DataFrame con todas las mejores alternativas
                        if all_alternatives:
                            results = pd.DataFrame(all_alternatives)
                        else:
                            results = pd.DataFrame()
                    else:
                        # Buscar alternativas para la categoría seleccionada
                        results = find_closest_alternatives(
                            filtered_df, user_avg_lat, user_avg_lon, 
                            category=selected_category, max_distance=max_distance
                        )
                    
                    if results.empty:
                        st.info(f"No se encontraron alternativas para {selected_category} dentro de {max_distance} km.")
                    else:
                        st.success(f"¡Encontradas {len(results)} alternativas!")
                        
                        # Mostrar resultados en un mapa
                        alt_map = folium.Map(location=[user_avg_lat, user_avg_lon], zoom_start=13)
                        
                        # Marcar la ubicación del usuario
                        folium.Marker(
                            [user_avg_lat, user_avg_lon],
                            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
                            tooltip="Tu ubicación"
                        ).add_to(alt_map)
                        
                        # Añadir marcadores para cada alternativa
                        for _, alt in results.iterrows():
                            popup = f"""
                            <b>{alt['merchant_name']}</b><br>
                            <b>Categoría:</b> {alt['merchant_category']}<br>
                            <b>Precio promedio:</b> ${alt['avg_price']:.2f}<br>
                            <b>Distancia:</b> {alt['distance_from_user']:.2f} km
                            """
                            
                            # Usar un ícono de tienda con color según score (verde=mejor, rojo=peor)
                            if 'combined_score' in alt:
                                # Normalizar score entre 0 y 1 (0 es mejor)
                                score_norm = min(1.0, max(0.0, alt['combined_score']))
                                # Convertir a color: verde (0) a rojo (1)
                                r = int(255 * score_norm)
                                g = int(255 * (1 - score_norm))
                                b = 0
                                color = f'#{r:02x}{g:02x}{b:02x}'
                            else:
                                color = "green"
                            
                            folium.Marker(
                                [alt['merchant_lat'], alt['merchant_lon']],
                                popup=popup,
                                icon=folium.Icon(color="green" if 'combined_score' not in alt else None, 
                                                icon_color=color if 'combined_score' in alt else None,
                                                icon="shopping-cart", prefix="fa"),
                                tooltip=f"{alt['merchant_name']} - ${alt['avg_price']:.2f}"
                            ).add_to(alt_map)
                            
                            # Dibujar línea desde usuario a comercio
                            folium.PolyLine(
                                [(user_avg_lat, user_avg_lon), (alt['merchant_lat'], alt['merchant_lon'])],
                                color=color,
                                weight=2,
                                opacity=0.7,
                                tooltip=f"Distancia: {alt['distance_from_user']:.2f} km"
                            ).add_to(alt_map)
                        
                        # Mostrar el mapa
                        st.write("Mapa de alternativas cercanas (color verde = mejor relación precio/distancia):")
                        folium_static(alt_map)
                        
                        # Mostrar tabla de resultados
                        st.write("Alternativas ordenadas por mejor relación precio/distancia:")
                        display_results = results[['merchant_name', 'merchant_category', 'avg_price', 'distance_from_user', 'transaction_count']]
                        display_results.columns = ['Comercio', 'Categoría', 'Precio promedio ($)', 'Distancia (km)', 'Popularidad']
                        st.dataframe(display_results)

        # Buscar alternativas tradicionales (solo precio)
        st.subheader("Alternativas de ahorro tradicionales")
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
            if expensive_day_cats:
                st.write(f"El {days[expensive_day]} sueles gastar principalmente en: {', '.join(expensive_day_cats)}")

    elif view_option == "Predicciones":
        st.header("🔮 Predicciones de gasto")
        
        st.info("Basado en datos históricos, podemos predecir patrones de gasto futuros.")
        
        # IMPORTANTE: Para predicciones usamos todos los datos, no solo los del usuario
        # Aquí usamos all_data_df en lugar de filtered_df para tener una visión completa
        
        # Si tenemos datos de usuario, aplicamos clustering a todos los datos
        if user_first_name and user_last_name:
            st.write("Para las predicciones, analizamos tendencias generales de todos los usuarios y las aplicamos a tu perfil personal.")
            
            # Aplicar clustering a todos los datos
            all_clustered_df = cluster_locations(all_data_df, min_cluster_size=5)
            
            # Pero filtramos para el usuario actual para las comparaciones personalizadas
            user_mask = (
                (all_clustered_df["first"].str.lower() == user_first_name.lower()) & 
                (all_clustered_df["last"].str.lower() == user_last_name.lower())
            )
            user_clustered_df = all_clustered_df[user_mask]
        else:
            # Sin usuario específico, usamos todos los datos
            all_clustered_df = cluster_locations(all_data_df, min_cluster_size=5)
            user_clustered_df = all_clustered_df
        
        # Simulaciones
        st.subheader("Simulaciones de escenarios")
        
        # Simulador de cambio de ubicación
        st.write("**¿Y si cambias tu ubicación habitual?**")
        
        # Lista de clusters/zonas detectadas
        if 'cluster' in all_clustered_df.columns and all_clustered_df['cluster'].nunique() > 1:
            valid_clusters = [c for c in all_clustered_df['cluster'].unique() if c >= 0]
            
            # Determinar zonas actuales del usuario
            if user_first_name and user_last_name:
                user_zones = set(user_clustered_df['cluster'].unique())
                other_zones = [z for z in valid_clusters if z not in user_zones]
                
                if not user_zones or len(user_zones) == 0:
                    st.warning("No tenemos suficientes datos para determinar tus zonas habituales.")
                    return
                    
                col1, col2 = st.columns(2)
                
                with col1:
                    current_zone = st.selectbox(
                        "Tu zona actual:",
                        options=sorted(list(user_zones)),
                        format_func=lambda x: f"Zona {x}"
                    )
                
                with col2:
                    new_zone = st.selectbox(
                        "Nueva zona potencial:",
                        options=sorted([z for z in valid_clusters if z != current_zone]),
                        format_func=lambda x: f"Zona {x}"
                    )
            else:
                # Sin usuario específico
                col1, col2 = st.columns(2)
                
                with col1:
                    current_zone = st.selectbox(
                        "Zona actual:",
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
                # Datos para la zona actual
                current_data = all_clustered_df[all_clustered_df['cluster'] == current_zone]
                
                # Si hay usuario, filtramos los datos actuales para el usuario
                if user_first_name and user_last_name:
                    user_current_data = current_data[
                        (current_data["first"].str.lower() == user_first_name.lower()) & 
                        (current_data["last"].str.lower() == user_last_name.lower())
                    ]
                    if not user_current_data.empty:
                        current_data = user_current_data
                    
                # Datos para la nueva zona (siempre todos los usuarios)
                new_data = all_clustered_df[all_clustered_df['cluster'] == new_zone]
                
                # Gastos promedio
                current_avg = current_data['amount'].mean()
                new_avg = new_data['amount'].mean()
                
                diff_pct = ((new_avg - current_avg) / current_avg) * 100
                
                # Categorías principales
                current_cats = current_data['merchant_category'].value_counts().head(3)
                new_cats = new_data['merchant_category'].value_counts().head(3)
                
                # Personalización del mensaje
                personal_prefix = "tu" if user_first_name else "el"
                
                # Visualización comparativa
                if diff_pct > 0:
                    st.warning(f"Si te mudaras de la Zona {current_zone} a la Zona {new_zone}, {personal_prefix} gasto promedio aumentaría aproximadamente un {diff_pct:.1f}%.")
                else:
                    st.success(f"Si te mudaras de la Zona {current_zone} a la Zona {new_zone}, {personal_prefix} gasto promedio se reduciría aproximadamente un {abs(diff_pct):.1f}%.")
                
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

                # Análisis de distancia si hay datos disponibles
                if 'distance_to_merchant' in all_clustered_df.columns:
                    st.write("**Comparativa de distancia a comercios:**")
                    
                    # Calcular distancia promedio por zona
                    current_dist = current_data['distance_to_merchant'].mean()
                    new_dist = new_data['distance_to_merchant'].mean()
                    
                    if not pd.isna(current_dist) and not pd.isna(new_dist):
                        dist_diff = new_dist - current_dist
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Distancia actual promedio", f"{current_dist:.2f} km")
                        with col2:
                            st.metric("Distancia nueva promedio", f"{new_dist:.2f} km", f"{dist_diff:+.2f} km")
                        
                        if dist_diff > 0:
                            st.info(f"En la nueva zona, tendrías que desplazarte en promedio {dist_diff:.2f} km más para hacer tus compras.")
                        else:
                            st.info(f"En la nueva zona, te desplazarías en promedio {abs(dist_diff):.2f} km menos para hacer tus compras.")
        else:
            st.write("No se han detectado suficientes zonas diferentes para hacer predicciones geográficas.")

# Esta parte solo se ejecuta si app.py es llamado directamente
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Spending Map AI", page_icon="🗺️")
    main()