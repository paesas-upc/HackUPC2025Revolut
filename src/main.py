# src/main.py

import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Import app.py as a module to maintain code separation
import app

# Page configuration
st.set_page_config(
    page_title="Spending Map AI - Login", 
    page_icon="🗺️",
    layout="centered"
)

# Session initialization if it doesn't exist
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_first_name = ""
    st.session_state.user_last_name = ""
    st.session_state.user_full_name = ""

# Function to verify if the user exists in the dataset
def verify_user(first_name, last_name):
    try:
        # Load only the necessary columns to verify users
        df = pd.read_csv("data/transactions.csv", usecols=["first", "last"])
        
        # Convert to lowercase for case-insensitive comparison
        df["first"] = df["first"].str.lower()
        df["last"] = df["last"].str.lower()
        
        # Verify if the combination exists
        exists = ((df["first"] == first_name.lower()) & 
                 (df["last"] == last_name.lower())).any()
        
        return exists
    except Exception as e:
        st.error(f"Error verifying user: {str(e)}")
        return False

# Login function
def login_user():
    if st.session_state.first_name and st.session_state.last_name:
        if verify_user(st.session_state.first_name, st.session_state.last_name):
            st.session_state.authenticated = True
            st.session_state.user_first_name = st.session_state.first_name
            st.session_state.user_last_name = st.session_state.last_name
            st.session_state.user_full_name = f"{st.session_state.first_name} {st.session_state.last_name}"
            st.success("Access granted!")
            # We need to rerun to update the UI
            st.rerun()
        else:
            st.error("User not found. Please verify your first and last name.")

# Logout function
def logout_user():
    for key in ["authenticated", "user_first_name", "user_last_name", "user_full_name"]:
        st.session_state[key] = ""
    st.session_state.authenticated = False
    st.rerun()

# Main interface
if not st.session_state.authenticated:
    # Login screen
    st.title("🗺️ Spending Map AI")
    st.subheader("Visualize and analyze your spending geographically")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("First Name", key="first_name", placeholder="Enter your first name")
    
    with col2:
        st.text_input("Last Name", key="last_name", placeholder="Enter your last name")
    
    st.button("Login", on_click=login_user)
    
    # Additional information
    with st.expander("ℹ️ Information"):
        st.write("""
        **Spending Map AI** allows you to visualize your transactions on an interactive map,
        analyze your spending patterns by location, and get personalized recommendations
        to optimize your finances.
        
        To access, enter your first and last name associated with your account.
        """)
    
    # Demo users
    with st.expander("👤 Demo Users"):
        st.info("""
        To test the application, you can use any of the following users:
        
        - First Name: John, Last Name: Doe
        - First Name: Jane, Last Name: Smith
        - First Name: Michael, Last Name: Johnson
        
        (Note: these users will only work if they exist in your dataset)
        """)
        
    # Footer
    st.markdown("---")
    st.caption("HackUPC 2025 - Revolut Challenge | paesas-upc")
    
else:
    # If authenticated, show the main application
    st.sidebar.success(f"👤 Active session: {st.session_state.user_full_name.title()}")
    st.sidebar.button("Logout", on_click=logout_user)
    
    # Run the main application passing the user information
    app.main(
        user_first_name=st.session_state.user_first_name,
        user_last_name=st.session_state.user_last_name
    )