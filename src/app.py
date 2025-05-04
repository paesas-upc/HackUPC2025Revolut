# src/app.py

import streamlit as st
import folium
from folium.plugins import MarkerCluster
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
from insights import record_merchant_feedback, identify_spending_patterns, generate_user_insights, find_alternative_merchants, find_closest_alternatives

def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2)
        lightness = 0.4 + 0.1 * (i % 3)
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append('#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)))
    return colors

def main(user_first_name=None, user_last_name=None):
    """
    Main application function that can be called with user information
    
    Args:
        user_first_name: User's first name
        user_last_name: User's last name
    """
    st.sidebar.title("🔍 Spending Map AI")
    st.sidebar.info("Visualize and analyze your expenses geographically")

    view_option = st.sidebar.radio(
        "Select a view:",
        ["Expense Map", "Zone Analysis", "Recommendations", "Predictions"]
    )

    df = load_and_clean_data()
    
    if user_first_name and user_last_name:
        user_mask = (
            (df["first"].str.lower() == user_first_name.lower()) & 
            (df["last"].str.lower() == user_last_name.lower())
        )
        user_df = df[user_mask]
        
        if len(user_df) == 0:
            st.error(f"No transactions found for {user_first_name} {user_last_name}")
            return
            
        st.title(f"👋 Hello, {user_first_name.title()} {user_last_name.title()}")
        
        all_data_df = df.copy()
        df = user_df
    else:
        all_data_df = df.copy()
        st.title("🗺️ Spending Map AI")

    st.sidebar.subheader("Filters")
    date_range = st.sidebar.date_input(
        "Date range",
        value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
        min_value=df["timestamp"].min().date(),
        max_value=df["timestamp"].max().date()
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        filtered_df = df[mask]
    else:
        filtered_df = df

    if "merchant_category" in filtered_df.columns:
        all_categories = ["All"] + sorted(filtered_df["merchant_category"].unique().tolist())
        selected_category = st.sidebar.selectbox("Category", all_categories)
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["merchant_category"] == selected_category]

    if len(filtered_df) < 3:
        st.warning(f"Not enough data for detailed analysis. Found {len(filtered_df)} transactions.")
        filtered_df['cluster'] = 0
        clustered_df = filtered_df
    else:
        try:
            clustered_df = cluster_locations(filtered_df, min_cluster_size=5)
        except Exception as e:
            st.error(f"Error processing clusters: {str(e)}")
            filtered_df['cluster'] = 0
            clustered_df = filtered_df
            
    cluster_stats = analyze_clusters(clustered_df)

    if view_option == "Expense Map":
        st.header("🗺️ Expense Map")
        
        map_tab, stats_tab = st.tabs(["Marker Map", "Statistics"])
        
        with map_tab:
            st.subheader("Transaction Map")
            
            has_merchant_coords = ('merchant_latitude' in filtered_df.columns and 
                                filtered_df['merchant_latitude'].notna().any() and
                                'merchant_longitude' in filtered_df.columns and 
                                filtered_df['merchant_longitude'].notna().any())
            
            if not has_merchant_coords:
                st.warning("No merchant coordinates available in the data.")
            
            center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            m = folium.Map(location=center, zoom_start=12)
            
            categories = filtered_df['merchant_category'].unique()
            cat_colors = {cat: color for cat, color in zip(categories, get_distinct_colors(len(categories)))}
            
            user_marker = folium.FeatureGroup(name="User location").add_to(m)
            merchant_marker_cluster = MarkerCluster(name="Merchant locations").add_to(m)
            
            user_location = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            
            folium.Marker(
                location=user_location,
                popup="<b>User</b>",
                icon=folium.Icon(color="blue", icon="user", prefix="fa"),
                tooltip="User location"
            ).add_to(user_marker)
            
            if has_merchant_coords:
                merchant_df = filtered_df.dropna(subset=['merchant_latitude', 'merchant_longitude'])
                
                merchant_agg = merchant_df.groupby(['merchant_latitude', 'merchant_longitude', 'merchant_category']).agg(
                    total_spent=('amount', 'sum'),
                    avg_spent=('amount', 'mean'),
                    transaction_count=('amount', 'count')
                ).reset_index()
                
                for _, row in merchant_agg.iterrows():
                    popup = f"""
                    <b>Category:</b> {row['merchant_category']}<br>
                    <b>Total spent:</b> ${row['total_spent']:.2f}<br>
                    <b>Average spent:</b> ${row['avg_spent']:.2f}<br>
                    <b>Transactions:</b> {row['transaction_count']}
                    """
                    
                    color = cat_colors.get(row['merchant_category'], 'red')
                    
                    merchant_location = [row['merchant_latitude'], row['merchant_longitude']]
                    
                    folium.Marker(
                        location=merchant_location,
                        popup=popup,
                        icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
                        tooltip=f"Merchant - {row['merchant_category']}: ${row['total_spent']:.2f}"
                    ).add_to(merchant_marker_cluster)
            
            folium.LayerControl().add_to(m)
            
            st.write("The map shows the user's unique location (blue) and the merchants where transactions were made (red).")
            folium_static(m)
        
        with stats_tab:
            st.subheader("Expense Statistics")
            
            cat_spending = filtered_df.groupby('merchant_category')['amount'].sum().reset_index()
            cat_spending = cat_spending.sort_values('amount', ascending=False)
            
            fig = px.bar(
                cat_spending,
                x='merchant_category',
                y='amount',
                title='Total spending by category',
                labels={'merchant_category': 'Category', 'amount': 'Total amount ($)'}
            )
            st.plotly_chart(fig)
            
            if len(filtered_df['date'].unique()) > 1:
                time_spending = filtered_df.groupby(['date'])['amount'].sum().reset_index()
                
                fig_time = px.line(
                    time_spending,
                    x='date',
                    y='amount',
                    title='Temporal evolution of expenses',
                    labels={'date': 'Date', 'amount': 'Total amount ($)'}
                )
                st.plotly_chart(fig_time)

    elif view_option == "Zone Analysis":
        st.header("🔍 Geographic Zone Analysis")
        
        if user_first_name and user_last_name:
            all_clustered_df = cluster_locations(all_data_df, min_cluster_size=5)
            
            user_mask = (
                (all_data_df["first"].str.lower() == user_first_name.lower()) & 
                (all_data_df["last"].str.lower() == user_last_name.lower())
            )
            user_raw_df = all_data_df[user_mask].copy()
            
            user_points = user_raw_df[['latitude', 'longitude']].values
            all_clusters = all_clustered_df[all_clustered_df['cluster'] >= 0]
            
            if 'cluster' not in user_raw_df.columns:
                user_raw_df['cluster'] = -1
            
            user_clusters = set()
            for i, row in user_raw_df.iterrows():
                user_point = [row['latitude'], row['longitude']]
                
                min_dist = float('inf')
                closest_cluster = None
                
                for cluster_id in all_clusters['cluster'].unique():
                    cluster_points = all_clustered_df[all_clustered_df['cluster'] == cluster_id]
                    cluster_center = (
                        cluster_points['latitude'].mean(),
                        cluster_points['longitude'].mean()
                    )
                    
                    dist = np.sqrt(
                        (user_point[0] - cluster_center[0])**2 + 
                        (user_point[1] - cluster_center[1])**2
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = cluster_id
                
                if closest_cluster is not None:
                    user_raw_df.at[i, 'cluster'] = closest_cluster
                    user_clusters.add(closest_cluster)
            
            user_clusters = sorted(list(user_clusters))
            
            cluster_stats = analyze_clusters(all_clustered_df)
            
            st.write(f"Identified {all_clustered_df[all_clustered_df['cluster'] >= 0]['cluster'].nunique()} distinct spending zones in the entire database.")
            
            if user_clusters:
                st.write(f"Your transactions are mainly concentrated in {len(user_clusters)} zones: {', '.join([f'Zone {c}' for c in user_clusters])}")
            
            center = [all_data_df['latitude'].mean(), all_data_df['longitude'].mean()]
            cluster_map = folium.Map(location=center, zoom_start=12)
            
            cluster_ids = sorted([c for c in all_clustered_df['cluster'].unique() if c >= 0])
            n_clusters = len(cluster_ids)
            colors = get_distinct_colors(n_clusters)
            cluster_colors = {cid: colors[i] for i, cid in enumerate(cluster_ids)}
            
            for cluster_id in cluster_ids:
                cluster_points = all_clustered_df[all_clustered_df['cluster'] == cluster_id]
                
                cluster_center = [
                    cluster_points['latitude'].mean(), 
                    cluster_points['longitude'].mean()
                ]
                
                total_spent = cluster_points['amount'].sum()
                avg_spent = cluster_points['amount'].mean()
                transaction_count = len(cluster_points)
                top_category = cluster_points['merchant_category'].value_counts().index[0]
                
                cluster_color = cluster_colors.get(cluster_id, 'gray')
                fill_opacity = 0.6 if cluster_id in user_clusters else 0.3
                
                folium.Circle(
                    location=cluster_center,
                    radius=100,
                    color=cluster_color,
                    fill=True,
                    fill_opacity=fill_opacity,
                    tooltip=f"Zone {cluster_id}: ${total_spent:.2f} ({transaction_count} transactions)"
                ).add_to(cluster_map)
                
                folium.Marker(
                    location=cluster_center,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">Zone {cluster_id}</div>'
                    )
                ).add_to(cluster_map)
            
            if user_first_name and len(user_raw_df) > 0:
                user_points_group = folium.FeatureGroup(name="Your locations")
                
                for _, row in user_raw_df.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=4,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.7,
                        tooltip=f"Your transaction: ${row['amount']:.2f} at {row['merchant_category']}"
                    ).add_to(user_points_group)
                
                user_points_group.add_to(cluster_map)
            
            folium.LayerControl().add_to(cluster_map)
            
            st.write("Zones identified through geographic clustering:")
            st.write("The blue dot represents your location.")
            folium_static(cluster_map)
            
            st.subheader("Zone Comparison")
            
            compare_data = cluster_stats[['cluster', 'avg_spent', 'total_spent', 'transaction_count', 'dominant_category']]
            compare_data = compare_data.sort_values('total_spent', ascending=False)
            
            compare_data['user_present'] = compare_data['cluster'].apply(lambda x: "Yes" if x in user_clusters else "No")
            
            fig = px.bar(
                compare_data,
                x='cluster',
                y='total_spent',
                color='dominant_category',
                pattern_shape='user_present',
                title='Total spending by zone',
                labels={
                    'cluster': 'Zone', 
                    'total_spent': 'Total amount ($)', 
                    'dominant_category': 'Dominant category',
                    'user_present': 'Your activity'
                }
            )
            st.plotly_chart(fig)
            
            st.subheader("Detailed Zone Analysis")
            
            if user_clusters:
                all_cluster_options = user_clusters + [c for c in cluster_ids if c not in user_clusters]
                default_cluster = user_clusters[0] if user_clusters else cluster_ids[0]
            else:
                all_cluster_options = cluster_ids
                default_cluster = cluster_ids[0] if cluster_ids else 0
            
            selected_cluster = st.selectbox(
                "Select a zone to analyze in detail:",
                options=all_cluster_options,
                index=0,
                format_func=lambda x: f"Zone {x}" + (" (with your activity)" if x in user_clusters else "")
            )
            
            cluster_data = all_clustered_df[all_clustered_df['cluster'] == selected_cluster]
            
            if user_first_name:
                user_cluster_data = user_raw_df[user_raw_df['cluster'] == selected_cluster]
                has_user_data = not user_cluster_data.empty
            else:
                user_cluster_data = pd.DataFrame()
                has_user_data = False
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total spent (all)", f"${cluster_data['amount'].sum():.2f}")
            with col2:
                st.metric("Transactions (all)", f"{len(cluster_data)}")
            with col3:
                st.metric("Average spent (all)", f"${cluster_data['amount'].mean():.2f}")
            
            if has_user_data:
                st.subheader(f"Your activity in Zone {selected_cluster}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    user_total = user_cluster_data['amount'].sum()
                    pct_total = (user_total / cluster_data['amount'].sum()) * 100
                    st.metric("Your total spending", f"${user_total:.2f}", f"{pct_total:.1f}% of total")
                    
                with col2:
                    user_count = len(user_cluster_data)
                    pct_count = (user_count / len(cluster_data)) * 100
                    st.metric("Your transactions", f"{user_count}", f"{pct_count:.1f}% of total")
                    
                with col3:
                    user_avg = user_cluster_data['amount'].mean()
                    avg_diff = ((user_avg / cluster_data['amount'].mean()) - 1) * 100
                    st.metric("Your average spending", f"${user_avg:.2f}", f"{avg_diff:+.1f}% vs average")

    elif view_option == "Recommendations":
        st.header("💡 Personalized Recommendations")
        
        insights = generate_user_insights(filtered_df)
        
        st.subheader("Insights about your spending habits")
        for insight in insights:
            st.info(insight)
        
        st.subheader("Savings alternatives by proximity")

        user_avg_lat = filtered_df['latitude'].mean()
        user_avg_lon = filtered_df['longitude'].mean()

        has_merchant_coords = ('merchant_latitude' in filtered_df.columns and 
                            'merchant_longitude' in filtered_df.columns and
                            filtered_df['merchant_latitude'].notna().any() and
                            filtered_df['merchant_longitude'].notna().any())

        if not has_merchant_coords:
            st.warning("No merchant coordinates available in the data to generate recommendations.")
        else:
            categories = ["All"] + sorted(filtered_df['merchant_category'].unique().tolist())
            selected_category = st.selectbox("Select a category to find alternatives:", categories)
            
            max_distance = st.slider("Maximum distance (km):", 0.5, 10.0, 5.0, 0.5)
            
            if st.button("Find nearby alternatives"):
                with st.spinner("Searching for the best options..."):
                    if selected_category == "All":
                        all_alternatives = []
                        for category in filtered_df['merchant_category'].unique():
                            alternatives = find_closest_alternatives(
                                filtered_df, user_avg_lat, user_avg_lon, 
                                category=category, max_distance=max_distance,
                                use_linucb=True
                            )
                            if not alternatives.empty:
                                all_alternatives.append(alternatives.iloc[0])
                        
                        if all_alternatives:
                            results = pd.DataFrame(all_alternatives)
                        else:
                            results = pd.DataFrame()
                    else:
                        results = find_closest_alternatives(
                            filtered_df, user_avg_lat, user_avg_lon, 
                            category=selected_category, max_distance=max_distance,
                            use_linucb=True
                        )
                    
                    if results.empty:
                        st.info(f"No alternatives found for {selected_category} within {max_distance} km.")
                    else:
                        st.success(f"Found {len(results)} alternatives!")
                        
                        st.session_state['recommendation_results'] = results
                        
                        alt_map = folium.Map(location=[user_avg_lat, user_avg_lon], zoom_start=13)
                        
                        folium.Marker(
                            [user_avg_lat, user_avg_lon],
                            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
                            tooltip="Your location"
                        ).add_to(alt_map)
                        
                        for idx, alt in results.iterrows():
                            if 'ucb_score' in alt:
                                score_norm = 1.0 - min(1.0, max(0.0, alt['ucb_score'] / results['ucb_score'].max()))
                            elif 'combined_score' in alt:
                                score_norm = min(1.0, max(0.0, alt['combined_score']))
                            else:
                                score_norm = 0.5
                            
                            r = int(255 * score_norm)
                            g = int(255 * (1 - score_norm))
                            b = 0
                            color = f'#{r:02x}{g:02x}{b:02x}'
                            
                            score_info = ""
                            if 'ucb_score' in alt:
                                score_info = f"<b>LinUCB Score:</b> {alt['ucb_score']:.2f}<br>"
                            elif 'combined_score' in alt:
                                score_info = f"<b>Combined score:</b> {alt['combined_score']:.2f}<br>"
                            
                            popup = f"""
                            <b>{alt['merchant_name']}</b><br>
                            <b>Category:</b> {alt['merchant_category']}<br>
                            <b>Average price:</b> ${alt['avg_price']:.2f}<br>
                            <b>Distance:</b> {alt['distance_from_user']:.2f} km<br>
                            {score_info}
                            """
                            
                            folium.Marker(
                                [alt['merchant_lat'], alt['merchant_lon']],
                                popup=popup,
                                icon=folium.Icon(color="green" if score_norm < 0.3 else None, 
                                                icon_color=color,
                                                icon="shopping-cart", prefix="fa"),
                                tooltip=f"{alt['merchant_name']} - ${alt['avg_price']:.2f}"
                            ).add_to(alt_map)
                            
                            folium.PolyLine(
                                [(user_avg_lat, user_avg_lon), (alt['merchant_lat'], alt['merchant_lon'])],
                                color=color,
                                weight=2,
                                opacity=0.7,
                                tooltip=f"Distance: {alt['distance_from_user']:.2f} km"
                            ).add_to(alt_map)
                        
                        st.write("Map of nearby alternatives (green = best according to LinUCB algorithm):")
                        folium_static(alt_map)
                        
                        st.write("Alternatives recommended by LinUCB:")
                        
                        display_cols = ['merchant_name', 'merchant_category', 'avg_price', 'distance_from_user', 'transaction_count']
                        
                        if 'ucb_score' in results.columns:
                            display_cols.extend(['ucb_score', 'expected_reward', 'exploration_bonus'])
                        elif 'combined_score' in results.columns:
                            display_cols.append('combined_score')
                            
                        display_results = results[display_cols].copy()
                        
                        column_renames = {
                            'merchant_name': 'Merchant', 
                            'merchant_category': 'Category', 
                            'avg_price': 'Average price ($)', 
                            'distance_from_user': 'Distance (km)', 
                            'transaction_count': 'Popularity',
                            'ucb_score': 'LinUCB Score',
                            'expected_reward': 'Expected reward',
                            'exploration_bonus': 'Exploration bonus',
                            'combined_score': 'Combined score'
                        }
                        display_results.columns = [column_renames.get(col, col) for col in display_results.columns]
                        
                        st.dataframe(display_results)
                        
                        st.subheader("Do you like this recommendation?")
                        st.write("Your feedback helps us improve our suggestions.")

            if 'recommendation_results' in st.session_state and not st.session_state['recommendation_results'].empty:
                results = st.session_state['recommendation_results']
                
                if 'selected_merchant' not in st.session_state:
                    st.session_state['selected_merchant'] = results['merchant_name'].iloc[0]
                
                if 'feedback_score' not in st.session_state:
                    st.session_state['feedback_score'] = 0.5
                    
                selected_merchant = st.selectbox(
                    "Select a merchant to rate:",
                    options=results['merchant_name'].tolist(),
                    key='merchant_selection',
                    index=list(results['merchant_name']).index(st.session_state['selected_merchant'])
                )
                st.session_state['selected_merchant'] = selected_merchant
                
                feedback_score = st.slider(
                    "How useful was this recommendation?",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state['feedback_score'],
                    step=0.1,
                    key='feedback_slider',
                    help="0 = Not useful, 1 = Very useful"
                )
                st.session_state['feedback_score'] = feedback_score
                
                if st.button("Submit rating", key='submit_feedback'):
                    merchant_data = results[results['merchant_name'] == selected_merchant].iloc[0].to_dict()
                    
                    user_avg_lat = filtered_df['latitude'].mean()
                    user_avg_lon = filtered_df['longitude'].mean()
                    
                    success = record_merchant_feedback(
                        selected_merchant,
                        user_avg_lat,
                        user_avg_lon,
                        feedback_score,
                        merchant_data
                    )
                    
                    if success:
                        st.success(f"Thank you for your feedback on {selected_merchant}!")
                        st.info("Your ratings help improve our future recommendations.")
                        
                        st.session_state['feedback_score'] = 0.5
                    else:
                        st.error("Could not process feedback. Please try again.")
                        
        st.subheader("Traditional Savings Alternatives")
        recommendations = find_alternative_merchants(filtered_df)
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                st.write(f"**Recommendation {i+1}:** Instead of {rec['original_merchant']}, consider going to {rec['alternative']} which is {rec['distance_km']:.1f} km away and you could save approximately {rec['savings_percent']:.1f}%.")
        else:
            st.write("No savings alternatives found based on your current data.")
        
        st.subheader("Tips based on your patterns")
        
        if 'day_of_week' in filtered_df.columns:
            day_spending = filtered_df.groupby('day_of_week')['amount'].mean()
            expensive_day = day_spending.idxmax()
            cheap_day = day_spending.idxmin()
            
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            st.write(f"Your most expensive day of the week is **{days[expensive_day]}** while on **{days[cheap_day]}** you tend to spend less.")
            
            expensive_day_cats = filtered_df[filtered_df['day_of_week'] == expensive_day]['merchant_category'].value_counts().head(2).index.tolist()
            if expensive_day_cats:
                st.write(f"On {days[expensive_day]} you mainly spend on: {', '.join(expensive_day_cats)}")

    elif view_option == "Predictions":
        st.header("🔮 Spending Predictions")
        
        st.info("Based on historical data, we can predict future spending patterns.")
        
        if user_first_name and user_last_name:
            all_clustered_df = cluster_locations(all_data_df, min_cluster_size=5)
            
            user_mask = (
                (all_data_df["first"].str.lower() == user_first_name.lower()) & 
                (all_data_df["last"].str.lower() == user_last_name.lower())
            )
            user_raw_df = all_data_df[user_mask].copy()
            
            user_points = user_raw_df[['latitude', 'longitude']].values
            all_clusters = all_clustered_df[all_clustered_df['cluster'] >= 0]
            
            if 'cluster' not in user_raw_df.columns:
                user_raw_df['cluster'] = -1
            
            user_clusters = set()
            for i, row in user_raw_df.iterrows():
                user_point = [row['latitude'], row['longitude']]
                
                min_dist = float('inf')
                closest_cluster = None
                
                for cluster_id in all_clusters['cluster'].unique():
                    cluster_points = all_clustered_df[all_clustered_df['cluster'] == cluster_id]
                    cluster_center = (
                        cluster_points['latitude'].mean(),
                        cluster_points['longitude'].mean()
                    )
                    
                    dist = np.sqrt(
                        (user_point[0] - cluster_center[0])**2 + 
                        (user_point[1] - cluster_center[1])**2
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = cluster_id
                
                if closest_cluster is not None:
                    user_raw_df.at[i, 'cluster'] = closest_cluster
                    user_clusters.add(closest_cluster)
            
            user_clusters = sorted(list(user_clusters))
        else:
            all_clustered_df = cluster_locations(all_data_df, min_cluster_size=5)
            user_clusters = []
        
        st.subheader("Scenario Simulations")
        
        st.write("**What if you change your usual location?**")
        
        if 'cluster' in all_clustered_df.columns and all_clustered_df['cluster'].nunique() > 1:
            valid_clusters = [c for c in all_clustered_df['cluster'].unique() if c >= 0]
            
            if user_first_name and user_last_name and user_clusters:
                other_zones = [z for z in valid_clusters if z not in user_clusters]
                
                if not user_clusters:
                    st.warning("Not enough data to determine your usual zones.")
                    return
                    
                col1, col2 = st.columns(2)
                
                with col1:
                    current_zone = st.selectbox(
                        "Your current zone:",
                        options=user_clusters,
                        format_func=lambda x: f"Zone {x}"
                    )
                
                with col2:
                    new_zone = st.selectbox(
                        "Potential new zone:",
                        options=sorted([z for z in valid_clusters if z != current_zone]),
                        format_func=lambda x: f"Zone {x}"
                    )
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    current_zone = st.selectbox(
                        "Current zone:",
                        options=valid_clusters,
                        format_func=lambda x: f"Zone {x}"
                    )
                
                with col2:
                    new_zone = st.selectbox(
                        "Potential new zone:",
                        options=[z for z in valid_clusters if z != current_zone],
                        format_func=lambda x: f"Zone {x}"
                    )
            
            if current_zone != new_zone:
                current_data = all_clustered_df[all_clustered_df['cluster'] == current_zone]
                
                if user_first_name and user_last_name:
                    user_current_data = user_raw_df[user_raw_df['cluster'] == current_zone]
                    if not user_current_data.empty:
                        current_data = user_current_data
                    
                new_data = all_clustered_df[all_clustered_df['cluster'] == new_zone]
                
                current_avg = current_data['amount'].mean()
                new_avg = new_data['amount'].mean()
                
                diff_pct = ((new_avg - current_avg) / current_avg) * 100
                
                current_cats = current_data['merchant_category'].value_counts().head(3)
                new_cats = new_data['merchant_category'].value_counts().head(3)
                
                personal_prefix = "your" if user_first_name else "the"
                
                if diff_pct > 0:
                    st.warning(f"If you moved from Zone {current_zone} to Zone {new_zone}, {personal_prefix} average spending would increase approximately {diff_pct:.1f}%.")
                else:
                    st.success(f"If you moved from Zone {current_zone} to Zone {new_zone}, {personal_prefix} average spending would decrease approximately {abs(diff_pct):.1f}%.")
                
                st.write("**Spending comparison by category:**")
                
                current_cat_avg = current_data.groupby('merchant_category')['amount'].mean().reset_index()
                current_cat_avg['zone'] = f'Zone {current_zone} (current)'
                
                new_cat_avg = new_data.groupby('merchant_category')['amount'].mean().reset_index()
                new_cat_avg['zone'] = f'Zone {new_zone} (potential)'
                
                compare_df = pd.concat([current_cat_avg, new_cat_avg])
                
                fig = px.bar(
                    compare_df,
                    x='merchant_category',
                    y='amount',
                    color='zone',
                    barmode='group',
                    title='Average spending comparison by category between zones',
                    labels={'merchant_category': 'Category', 'amount': 'Average spending ($)', 'zone': 'Zone'}
                )
                st.plotly_chart(fig)

                if 'distance_to_merchant' in all_clustered_df.columns:
                    st.write("**Distance to merchants comparison:**")
                    
                    current_dist = current_data['distance_to_merchant'].mean()
                    new_dist = new_data['distance_to_merchant'].mean()
                    
                    if not pd.isna(current_dist) and not pd.isna(new_dist):
                        dist_diff = new_dist - current_dist
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current average distance", f"{current_dist:.2f} km")
                        with col2:
                            st.metric("New average distance", f"{new_dist:.2f} km", f"{dist_diff:+.2f} km")
                        
                        if dist_diff > 0:
                            st.info(f"In the new zone, you would need to travel on average {dist_diff:.2f} km more for your purchases.")
                        else:
                            st.info(f"In the new zone, you would travel on average {abs(dist_diff):.2f} km less for your purchases.")
        else:
            st.write("Not enough distinct zones detected to make geographic predictions.")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Spending Map AI", page_icon="🗺️")
    main()