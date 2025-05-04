# Spending Map AI - HackUPC2025Revolut

## Overview

Spending Map AI is a sophisticated financial analytics platform developed for HackUPC2025 Revolut challenge. This application helps users visualize and optimize their spending habits geographically, providing personalized recommendations and insights.

The platform applies advanced machine learning (LinUCB algorithm) to deliver tailored recommendations based on user preferences, spending patterns, and location data.

## Key Features

- **Geospatial Spending Visualization**: View your spending patterns on interactive maps
- **Zone Analysis**: Understand where you spend more and identify savings opportunities
- **AI-Powered Recommendations**: Get personalized merchant recommendations using LinUCB algorithm
- **Spending Pattern Predictions**: Forecast your spending trends and simulate financial scenarios
- **Secure User Profiles**: Personal data management with user authentication

## LinUCB Implementation

The repository now implements the LinUCB (Linear Upper Confidence Bound) algorithm for contextual bandit-based recommendations. This algorithm balances exploration and exploitation to provide personalized merchant recommendations.

### How LinUCB Works

LinUCB improves upon traditional recommendation systems by:

1. **Contextual Awareness**: Takes into account user context and merchant features
2. **Adaptive Learning**: Improves recommendations over time based on feedback
3. **Exploration vs. Exploitation**: Balances trying new merchants vs. recommending known good options
4. **Personalization**: Adapts to each user's unique preferences

### Features Used in the Model

Our implementation utilizes these features for merchant recommendations:
- Price levels
- Distance from user
- Merchant popularity
- Category relevance

## Installation

```bash
# Clone repository
git clone https://github.com/paesas-upc/HackUPC2025Revolut.git
cd HackUPC2025Revolut

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/main.py
