# src/main.py

import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Importamos app.py como módulo para mantener la separación de código
import app

# Configuración de la página
st.set_page_config(
    page_title="Spending Map AI - Login", 
    page_icon="🗺️",
    layout="centered"
)

# Inicialización de la sesión si no existe
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_first_name = ""
    st.session_state.user_last_name = ""
    st.session_state.user_full_name = ""

# Función para verificar si el usuario existe en el dataset
def verify_user(first_name, last_name):
    try:
        # Cargamos solo las columnas necesarias para verificar usuarios
        df = pd.read_csv("data/transactions.csv", usecols=["first", "last"])
        
        # Convertimos a minúsculas para comparación insensible a mayúsculas
        df["first"] = df["first"].str.lower()
        df["last"] = df["last"].str.lower()
        
        # Verificamos si la combinación existe
        exists = ((df["first"] == first_name.lower()) & 
                 (df["last"] == last_name.lower())).any()
        
        return exists
    except Exception as e:
        st.error(f"Error al verificar usuario: {str(e)}")
        return False

# Función de login
def login_user():
    if st.session_state.first_name and st.session_state.last_name:
        if verify_user(st.session_state.first_name, st.session_state.last_name):
            st.session_state.authenticated = True
            st.session_state.user_first_name = st.session_state.first_name
            st.session_state.user_last_name = st.session_state.last_name
            st.session_state.user_full_name = f"{st.session_state.first_name} {st.session_state.last_name}"
            st.success("¡Acceso concedido!")
            # Necesitamos rerun para actualizar la UI
            st.rerun()
        else:
            st.error("Usuario no encontrado. Por favor, verifica tu nombre y apellido.")

# Función de cierre de sesión
def logout_user():
    for key in ["authenticated", "user_first_name", "user_last_name", "user_full_name"]:
        st.session_state[key] = ""
    st.session_state.authenticated = False
    st.rerun()

# Interfaz principal
if not st.session_state.authenticated:
    # Pantalla de login
    st.title("🗺️ Spending Map AI")
    st.subheader("Visualiza y analiza tus gastos geográficamente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Nombre", key="first_name", placeholder="Ingresa tu nombre")
    
    with col2:
        st.text_input("Apellido", key="last_name", placeholder="Ingresa tu apellido")
    
    st.button("Acceder", on_click=login_user)
    
    # Información adicional
    with st.expander("ℹ️ Información"):
        st.write("""
        **Spending Map AI** te permite visualizar tus transacciones en un mapa interactivo,
        analizar tus patrones de gasto por ubicación, y obtener recomendaciones personalizadas
        para optimizar tus finanzas.
        
        Para acceder, ingresa tu nombre y apellido asociados a tu cuenta.
        """)
    
    # Usuarios de demostración
    with st.expander("👤 Usuarios de demostración"):
        st.info("""
        Para probar la aplicación, puedes usar cualquiera de los siguientes usuarios:
        
        - Nombre: John, Apellido: Doe
        - Nombre: Jane, Apellido: Smith
        - Nombre: Michael, Apellido: Johnson
        
        (Nota: estos usuarios solo funcionarán si existen en tu dataset)
        """)
        
    # Footer
    st.markdown("---")
    st.caption("HackUPC 2025 - Revolut Challenge | paesas-upc")
    
else:
    # Si está autenticado, mostrar la aplicación principal
    st.sidebar.success(f"👤 Sesión activa: {st.session_state.user_full_name.title()}")
    st.sidebar.button("Cerrar sesión", on_click=logout_user)
    
    # Ejecutamos la aplicación principal pasando la información de usuario
    app.main(
        user_first_name=st.session_state.user_first_name,
        user_last_name=st.session_state.user_last_name
    )