# src/linucb.py
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

class LinUCBAlgorithm:
    """
    LinUCB Algorithm for Contextual Bandits
    
    Attributes:
        alpha (float): Exploration parameter
        context_dim (int): Dimension of context features
        arms (dict): Dictionary of arms where keys are arm IDs and values are arm information
    """
    def __init__(self, alpha=1.0, context_dim=4):
        self.alpha = alpha
        self.context_dim = context_dim
        self.arms = {}  # Dictionary to store arm info
        self.rewards_history = {}  # Dictionary to track rewards history
        
    def _arm_init(self, arm_id, context_dim):
        """Initialize an arm with given id and context dimension"""
        arm = {
            'A': np.identity(context_dim),  # A matrix (d x d)
            'b': np.zeros(context_dim),     # b vector (d x 1)
            'theta': np.zeros(context_dim), # theta vector (d x 1)
            'pulls': 0                       # number of times this arm has been pulled
        }
        return arm
        
    def add_arm(self, arm_id):
        """Add a new arm to the algorithm"""
        if arm_id not in self.arms:
            self.arms[arm_id] = self._arm_init(arm_id, self.context_dim)
            self.rewards_history[arm_id] = []
            
    def select_arm(self, context_features):
        """
        Select an arm based on context features using LinUCB algorithm
        
        Args:
            context_features (dict): Dictionary with arm_id as keys and feature vectors as values
            
        Returns:
            The arm with the highest UCB score
        """
        max_ucb = -np.inf
        selected_arm = None
        all_scores = {}
        
        for arm_id, arm in self.arms.items():
            if arm_id not in context_features:
                continue
                
            # Get context for this arm
            x = context_features[arm_id]
            
            # Calculate UCB
            A_inv = np.linalg.inv(arm['A'])
            theta = arm['theta']
            
            # Calculate expected reward (theta^T * x)
            expected_reward = np.dot(theta, x)
            
            # Calculate exploration bonus (alpha * sqrt(x^T * A_inv * x))
            exploration_bonus = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
            
            # UCB score
            ucb = expected_reward + exploration_bonus
            all_scores[arm_id] = (ucb, expected_reward, exploration_bonus)
            
            # Keep track of the arm with highest UCB
            if ucb > max_ucb:
                max_ucb = ucb
                selected_arm = arm_id
                
        return selected_arm, all_scores
    
    def update(self, arm_id, context, reward):
        """
        Update the algorithm based on selected arm, observed context and reward
        
        Args:
            arm_id: The ID of the selected arm
            context: The context vector observed
            reward: The reward observed
        """
        if arm_id not in self.arms:
            self.add_arm(arm_id)
            
        # Get the arm
        arm = self.arms[arm_id]
        
        # Update A (A = A + x * x^T)
        arm['A'] += np.outer(context, context)
        
        # Update b (b = b + r * x)
        arm['b'] += reward * context
        
        # Update theta (theta = A^(-1) * b)
        arm['theta'] = np.linalg.solve(arm['A'], arm['b'])
        
        # Update pulls count
        arm['pulls'] += 1
        
        # Record reward
        self.rewards_history[arm_id].append(reward)
        
    def get_arm_data(self, arm_id):
        """Get data for a specific arm"""
        if arm_id in self.arms:
            arm = self.arms[arm_id]
            return {
                'theta': arm['theta'],
                'pulls': arm['pulls'],
                'avg_reward': np.mean(self.rewards_history[arm_id]) if self.rewards_history[arm_id] else 0
            }
        return None
    
    def save_model(self, filepath='data/linucb_model.pkl'):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'context_dim': self.context_dim,
                'arms': self.arms,
                'rewards_history': self.rewards_history,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
            
    @classmethod
    def load_model(cls, filepath='data/linucb_model.pkl'):
        """Load the model from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Create instance with saved parameters
            instance = cls(alpha=data['alpha'], context_dim=data['context_dim'])
            instance.arms = data['arms']
            instance.rewards_history = data['rewards_history']
            return instance
        except FileNotFoundError:
            # Return a new instance if file doesn't exist
            return cls()
        except Exception as e:
            print(f"Error loading LinUCB model: {e}")
            return cls()


def extract_features_from_merchant(merchant, user_lat, user_lon, normalize_features=True):
    """
    Extract features from merchant data to use with LinUCB
    
    Returns:
        numpy array of features: [price_feature, distance_feature, popularity_feature, category_feature]
    """
    # Basic features
    features = np.zeros(4)
    
    # Price feature
    features[0] = merchant['avg_price']
    
    # Distance feature (if available)
    if 'distance_from_user' in merchant:
        features[1] = merchant['distance_from_user']
    elif 'merchant_lat' in merchant and 'merchant_lon' in merchant:
        from geopy.distance import geodesic
        features[1] = geodesic(
            (user_lat, user_lon), 
            (merchant['merchant_lat'], merchant['merchant_lon'])
        ).kilometers
    
    # Popularity feature (if available)
    if 'transaction_count' in merchant:
        features[2] = merchant['transaction_count']
    elif 'visit_count' in merchant:
        features[2] = merchant['visit_count']
    
    # Category feature - simple encoding
    # In production, you would use a proper encoding for categories
    # Here we just use a hash of the category string modulo 10
    if 'merchant_category' in merchant:
        features[3] = hash(merchant['merchant_category']) % 10
    
    # Normalize features if requested
    if normalize_features:
        # Apply simple min-max normalization for the first 3 features
        # Assuming reasonable value ranges
        features[0] = min(1.0, features[0] / 1000)  # Price normalized to 0-1
        features[1] = min(1.0, features[1] / 10)    # Distance normalized to 0-10km
        features[2] = min(1.0, features[2] / 100)   # Popularity capped at 100
    
    return features