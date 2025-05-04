import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import numpy as np
import os
from datetime import datetime

# Import the LinUCB algorithm
from linucb import LinUCBAlgorithm, extract_features_from_merchant

def calculate_distance(coords1, coords2):
    """
    Calculates the distance in kilometers between two coordinate pairs
    using the Haversine formula (geodesic)
    """
    try:
        distance = geodesic(coords1, coords2).kilometers
        return distance
    except:
        return None

def identify_spending_patterns(df):
    # Patterns by category and time of day
    time_patterns = df.groupby(['merchant_category', 'time_of_day']).agg(
        avg_transaction=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        avg_distance=('distance_to_merchant', 'mean')
    ).reset_index()
    
    # Patterns by category and day of the week
    week_patterns = df.groupby(['merchant_category', 'day_of_week']).agg(
        avg_transaction=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        avg_distance=('distance_to_merchant', 'mean')
    ).reset_index()
    
    # Identify categories with higher price variation
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
    Generates personalized insights for the user based on their data
    """
    insights = []
    
    # Verify there's enough data
    if len(df) < 5:
        insights.append("We don't have enough data to generate personalized insights. Keep using the app to get more accurate recommendations!")
        return insights
    
    # Insight about distances
    if 'distance_to_merchant' in df.columns:
        avg_distance = df['distance_to_merchant'].mean()
        if not pd.isna(avg_distance):
            insights.append(f"On average, you travel {avg_distance:.1f} km to make your purchases.")
            
            # Categories with greater distance
            cat_dist = df.groupby('merchant_category')['distance_to_merchant'].mean().sort_values(ascending=False)
            if not cat_dist.empty:
                furthest_cat = cat_dist.index[0]
                furthest_dist = cat_dist.iloc[0]
                if not pd.isna(furthest_dist) and furthest_dist > 0:
                    insights.append(f"You travel further for '{furthest_cat}', with an average of {furthest_dist:.1f} km.")
    
    # Insight about most frequent categories
    if len(df) >= 10:
        top_categories = df['merchant_category'].value_counts().head(3)
        if not top_categories.empty:
            top_cat = top_categories.index[0]
            top_count = top_categories.iloc[0]
            insights.append(f"Your most frequent spending category is '{top_cat}' with {top_count} transactions.")
    
    # Insight about zones if there are clusters
    if 'cluster' in df.columns:
        valid_clusters = df[df['cluster'] != -1]
        if not valid_clusters.empty:
            expensive_zone = valid_clusters.groupby('cluster')['amount'].mean().idxmax()
            cheapest_zone = valid_clusters.groupby('cluster')['amount'].mean().idxmin()
            
            if expensive_zone != cheapest_zone:
                exp_zone_avg = valid_clusters[valid_clusters['cluster'] == expensive_zone]['amount'].mean()
                cheap_zone_avg = valid_clusters[valid_clusters['cluster'] == cheapest_zone]['amount'].mean()
                
                diff_percentage = ((exp_zone_avg - cheap_zone_avg) / cheap_zone_avg) * 100
                
                insights.append(f"You spend on average {diff_percentage:.1f}% more in zone {expensive_zone} compared to zone {cheapest_zone}.")
    
    # Insight about most expensive days
    if 'day_of_week' in df.columns:
        day_spending = df.groupby('day_of_week')['amount'].mean()
        expensive_day = day_spending.idxmax()
        cheapest_day = day_spending.idxmin()
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        diff_day_percentage = ((day_spending.max() - day_spending.min()) / day_spending.min()) * 100
        
        if diff_day_percentage > 20:  # Only if the difference is significant
            insights.append(f"Your expenses are on average {diff_day_percentage:.1f}% higher on {days[expensive_day]} compared to {days[cheapest_day]}.")
    
    # Insight about time of day
    if 'time_of_day' in df.columns:
        time_spending = df.groupby('time_of_day')['amount'].mean()
        expensive_time = time_spending.idxmax()
        
        insights.append(f"You tend to spend more during the {expensive_time}.")
    
    return insights

def find_alternative_merchants(df):
    """
    Finds alternative merchants that might offer better prices
    """
    if len(df) < 10:
        return []
    
    # Make sure we have the necessary columns
    required_cols = ['merchant_name', 'merchant_category', 'amount', 
                     'merchant_latitude', 'merchant_longitude',
                     'latitude', 'longitude']
    
    if not all(col in df.columns for col in required_cols):
        return []
    
    # Find frequent merchants (at least 2 visits)
    merchant_counts = df['merchant_name'].value_counts()
    frequent_merchants = merchant_counts[merchant_counts >= 2].index.tolist()
    
    # If there aren't enough frequent merchants, we can't make recommendations
    if len(frequent_merchants) < 2:
        return []
    
    # Group data by merchant to get average price and location
    merchants = df.groupby(['merchant_name', 'merchant_category']).agg(
        avg_amount=('amount', 'mean'),
        visit_count=('amount', 'count'),
        avg_lat=('merchant_latitude', 'mean'),
        avg_lon=('merchant_longitude', 'mean')
    ).reset_index()
    
    # Filter to get only frequent merchants
    frequent_merchants_data = merchants[merchants['merchant_name'].isin(frequent_merchants)]
    
    recommendations = []
    
    # For each frequent merchant, look for cheaper alternatives in the same category
    for _, merchant in frequent_merchants_data.iterrows():
        # Look for alternatives in the same category
        same_category = merchants[
            (merchants['merchant_category'] == merchant['merchant_category']) &
            (merchants['merchant_name'] != merchant['merchant_name']) &
            (merchants['avg_amount'] < merchant['avg_amount'])
        ]
        
        if not same_category.empty:
            # Calculate distances from the original merchant to the alternatives
            same_category['distance'] = same_category.apply(
                lambda row: calculate_distance(
                    (merchant['avg_lat'], merchant['avg_lon']),
                    (row['avg_lat'], row['avg_lon'])
                ),
                axis=1
            )
            
            # Filter by maximum distance (2 km) and sort by price
            nearby_alternatives = same_category[same_category['distance'] <= 2].sort_values('avg_amount')
            
            if not nearby_alternatives.empty:
                best_alternative = nearby_alternatives.iloc[0]
                savings_percent = ((merchant['avg_amount'] - best_alternative['avg_amount']) / merchant['avg_amount']) * 100
                
                # Only recommend if the savings are significant (>10%)
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
    
    # Sort by savings percentage
    recommendations.sort(key=lambda x: x['savings_percent'], reverse=True)
    
    return recommendations

def find_closest_alternatives(df, user_lat, user_lon, category=None, max_distance=5.0, use_linucb=True):
    """
    Finds the closest merchants to the user, optionally filtered by category,
    and orders them considering both distance and price using LinUCB if available
    
    Args:
        df: DataFrame with transaction data
        user_lat: User's latitude
        user_lon: User's longitude
        category: Optional category to filter (default: None)
        max_distance: Maximum distance in km to consider
        use_linucb: If True, uses LinUCB to optimize recommendations
        
    Returns:
        DataFrame with nearby merchants ordered by recommendation score
    """
    # Verify we have enough data
    if df.empty:
        return pd.DataFrame()
        
    # Filter by category if specified
    category_df = df if category is None else df[df['merchant_category'] == category]
    
    if category_df.empty:
        return pd.DataFrame()
    
    # Group by merchant to get average price and location
    merchants = category_df.groupby(['merchant_name', 'merchant_category']).agg(
        avg_price=('amount', 'mean'),
        transaction_count=('amount', 'count'),
        merchant_lat=('merchant_latitude', 'mean'),
        merchant_lon=('merchant_longitude', 'mean')
    ).reset_index()
    
    # Calculate distance from the user's location
    merchants['distance_from_user'] = merchants.apply(
        lambda row: calculate_distance(
            (user_lat, user_lon),
            (row['merchant_lat'], row['merchant_lon'])
        ),
        axis=1
    )
    
    # Filter by maximum distance
    nearby = merchants[merchants['distance_from_user'] <= max_distance]
    
    if nearby.empty:
        return pd.DataFrame()
    
    # If LinUCB is not used, go back to the original method
    if not use_linucb:
        # Normalize price and distance to combine them into a score
        if len(nearby) > 1:  # Only normalize if there's more than one merchant
            nearby['price_norm'] = (nearby['avg_price'] - nearby['avg_price'].min()) / (nearby['avg_price'].max() - nearby['avg_price'].min() + 1e-10)
            nearby['distance_norm'] = (nearby['distance_from_user'] - nearby['distance_from_user'].min()) / (nearby['distance_from_user'].max() - nearby['distance_from_user'].min() + 1e-10)
            # Combined score (60% price, 40% distance)
            nearby['combined_score'] = 0.6 * nearby['price_norm'] + 0.4 * nearby['distance_norm']
        else:
            # If there's only one merchant, assign score 0
            nearby['price_norm'] = 0
            nearby['distance_norm'] = 0
            nearby['combined_score'] = 0
        
        # Sort by combined score (lower is better)
        result = nearby.sort_values('combined_score')
        
        return result
    
    # Use LinUCB for recommendations
    try:
        # Load or create a LinUCB model
        linucb_model = LinUCBAlgorithm.load_model()
        
        # Prepare the data for LinUCB
        # 1. Extract features from each merchant
        context_features = {}
        merchants_dict = {}
        
        for idx, merchant in nearby.iterrows():
            merchant_id = merchant['merchant_name']
            
            # Add the arm to the model if it's new
            linucb_model.add_arm(merchant_id)
            
            # Extract features
            features = extract_features_from_merchant(merchant, user_lat, user_lon)
            context_features[merchant_id] = features
            merchants_dict[merchant_id] = merchant
        
        # 2. Select merchants with LinUCB
        selected_arm, all_scores = linucb_model.select_arm(context_features)
        
        # 3. Add UCB score to the results
        for merchant_id, scores in all_scores.items():
            merchant_idx = nearby.index[nearby['merchant_name'] == merchant_id].tolist()
            if merchant_idx:
                nearby.at[merchant_idx[0], 'ucb_score'] = scores[0]
                nearby.at[merchant_idx[0], 'expected_reward'] = scores[1]
                nearby.at[merchant_idx[0], 'exploration_bonus'] = scores[2]
        
        # Sort by UCB score (higher is better)
        nearby['linucb_rank'] = nearby['ucb_score'].rank(ascending=False)
        result = nearby.sort_values('linucb_rank')
        
        # Save the model for future reference
        linucb_model.save_model()
        
        return result
    
    except Exception as e:
        print(f"Error using LinUCB: {str(e)}. Falling back to standard method.")
        
        # If there's an error, go back to the original method
        if len(nearby) > 1:
            nearby['price_norm'] = (nearby['avg_price'] - nearby['avg_price'].min()) / (nearby['avg_price'].max() - nearby['avg_price'].min() + 1e-10)
            nearby['distance_norm'] = (nearby['distance_from_user'] - nearby['distance_from_user'].min()) / (nearby['distance_from_user'].max() - nearby['distance_from_user'].min() + 1e-10)
            nearby['combined_score'] = 0.6 * nearby['price_norm'] + 0.4 * nearby['distance_norm']
        else:
            nearby['price_norm'] = 0
            nearby['distance_norm'] = 0
            nearby['combined_score'] = 0
            
        result = nearby.sort_values('combined_score')
        return result

def record_merchant_feedback(merchant_id, user_lat, user_lon, feedback_score, merchant_data=None):
    """
    Records user feedback for a merchant and updates the LinUCB model
    
    Args:
        merchant_id: Merchant ID selected
        user_lat: User's latitude
        user_lon: User's longitude
        feedback_score: Score from 0 to 1 (where 1 is the best)
        merchant_data: Merchant data to extract features
    
    Returns:
        Success flag
    """
    try:
        # Load LinUCB model
        linucb_model = LinUCBAlgorithm.load_model()
        
        if merchant_data:
            # Extract merchant features
            features = extract_features_from_merchant(merchant_data, user_lat, user_lon)
            
            # Update the model with the feedback
            linucb_model.update(merchant_id, features, feedback_score)
            
            # Save the updated model
            linucb_model.save_model()
            
            return True
        else:
            print("No merchant data provided for feedback")
            return False
    
    except Exception as e:
        print(f"Error recording feedback: {str(e)}")
        return False