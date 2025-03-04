import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('First Data Model')

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
    self.add_bedrooms_per_room = add_bedrooms_per_room
  def fit(self, X, y=None):
    return self # nothing else to do
  def transform(self, X, y=None):
    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
    population_per_household = X[:, population_ix] / X[:, households_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
      return np.c_[X, rooms_per_household, population_per_household,
                   bedrooms_per_room]

    else:
      return np.c_[X, rooms_per_household, population_per_household]
# Cargar modelo previamente entrenado
full_pipeline = joblib.load("full_pipeline.pkl")
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
ocean_proximity = st.selectbox("Proximidad al Océano", ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"])

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

# Botón para predecir
if st.button("Predecir"):
    # Aplicar el pipeline para transformar los datos
    data_prepared = full_pipeline.transform(data)

    prediccion = modelo.predict(data_prepared)
    st.write("### Predicción del Modelo")
    st.write(prediccion[0])