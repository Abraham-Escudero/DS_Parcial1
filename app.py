import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('First Data Model')

# Cargar modelo previamente entrenado
modelo = joblib.load("final_model.pkl")

# Crear inputs para cada atributo
longitude = st.number_input("Longitude", value=0.0)
latitude = st.number_input("Latitude", value=0.0)
housing_median_age = st.number_input("Housing Median Age", value=0)
total_rooms = st.number_input("Total Rooms", value=0)
total_bedrooms = st.number_input("Total Bedrooms", value=0)
population = st.number_input("Population", value=0)
households = st.number_input("Households", value=0)
median_income = st.number_input("Median Income", value=0.0)
ocean_proximity = st.selectbox("Ocean Proximity", options=[0, 1, 2, 3, 4])

# Crear DataFrame con los valores ingresados
data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Botón para realizar predicción
if st.button("Predecir"):
    prediccion = modelo.predict(data)
    st.write("### Predicción del Modelo")
    st.write(prediccion[0])