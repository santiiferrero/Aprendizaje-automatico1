import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from datetime import date, datetime
from funciones import season,divide_estaciones,rellenar,eliminar,rellenar_con_cols,eliminar_viento,dummies_elim,rellenar_ultimos,estandarizacion,preprocesamiento, agrego_columnas
from clases import NeuralNetworkClass, NeuralNetworkReg, MiniBatchGradientDescentRegressor
import pandas as pd 
import numpy as np 
from tensorflow_addons.metrics import RSquare


pipe = joblib.load('regresion_logistica (2).pkl')
pipe_reg = joblib.load('gradiente-mini-batch (2).pkl')
#pipe_red_reg = joblib.load('red_neuronal_regresion1.pkl')
#pipe_red_clas = joblib.load('red_neuronal_clasificacion.pkl')

def get_user_input():
    with st.form(key='my_form'):
        submit_button = st.form_submit_button(label='Submit')
    return submit_button

st.title('TP Aprendizaje Automatico')

#st.title('Cargar CSV')

# Widget para cargar un archivo CSV
#file = st.file_uploader('Cargar archivo CSV', type=['csv'])

#if file is not None:
    # Leer el archivo CSV y mostrar los datos
#    df = pd.read_csv(file)
#    st.write('Datos del archivo CSV:')
#    st.write(df)

selected_date = st.date_input('Date', date.today())
location = st.text_input('Location (Sydney, SydneyAirport, Canberra, Melbourne y MelbourneAirport)', 'Sidney')
a1 = st.slider('MinTemp', -20.0, 40.0, 5.0)
a2 = st.slider('MaxTemp', -10.0, 60.0, 5.0)
a3 = st.slider('Rainfall', 0.0, 500.0, 5.0)
a4 = st.slider('Evaporation', 0.0, 30.0, 5.0)
a5 = st.slider('Sunshine', 0.0, 14.0, 5.0)
dir = st.text_input('WindGustDir', 'N')
a6 = st.slider('WindGustSpeed', 0.0, 130.0, 5.0)
dir9 = st.text_input('WindDir9am', 'N')
dir3 = st.text_input('WindDir3pm', 'N')
a7 = st.slider('WindSpeed9am', 0.0, 130.0, 5.0)
a8 = st.slider('WindSpeed3pm', 0.0, 130.0, 5.0)
a9 = st.slider('Humidity9am', 0.0, 100.0, 5.0)
a10 = st.slider('Humidity3pm', 0.0, 100.0, 5.0)
a11 = st.slider('Pressure9am', 500.0, 1500.0, 5.0)
a12 = st.slider('Pressure3pm', 500.0, 1500.0, 5.0)
a13 = st.slider('Cloud9am', 0.0, 10.0, 5.0)
a14 = st.slider('Cloud3pm', 0.0, 10.0, 5.0)
a15 = st.slider('Temp9am', -20.0, 40.0, 5.0)
a16 = st.slider('Temp3pm', -10.0, 60.0, 5.0)
rain = st.text_input('RainToday (Yes, No)', 'No')


submit_button = get_user_input()
# When the 'Submit' button is pressed, perform the prediction
if submit_button:

    # Crear el diccionario
    data_para_predecir = {
    'Date': selected_date.strftime('%Y-%m-%d'),
    'Location': location,
    'MinTemp': a1,
    'MaxTemp': a2,
    'Rainfall': a3,
    'Evaporation': a4,
    'Sunshine': a5,
    'WindGustDir': dir,
    'WindGustSpeed': a6,
    'WindDir9am': dir9,
    'WindDir3pm': dir3,
    'WindSpeed9am': a7,
    'WindSpeed3pm': a8,
    'Humidity9am': a9,
    'Humidity3pm': a10,
    'Pressure9am': a11,
    'Pressure3pm': a12,
    'Cloud9am': a13,
    'Cloud3pm': a14,
    'Temp9am': a15,
    'Temp3pm': a16,
    'RainToday': rain
}

    data = pd.DataFrame([data_para_predecir])

    # Predicciones lluvia
    prediction = pipe.predict(data)
    prediction_clas = prediction[0]
    # Display the prediction
    st.header("Rain Tomorrow?")
    if prediction_clas == 1:
        st.write('Yes')
    else: 
        st.write('No')
    

    prediction = pipe_reg.predict(data)
    prediction_reg = prediction[0]
    # Display the prediction
    st.header("Rainfall Tomorrow?")
    st.write(prediction_reg)

    #prediction = pipe_red_reg.predict(data)
    #prediction_red_reg = prediction[0]
    # Display the prediction
    #st.header("Regresion de una red neuronal")
    #st.write(prediction_red_reg)

    #prediction = pipe_red_clas.predict(pd.DataFrame([data_para_predecir]))
    #prediction_red_clas = prediction[0]
    #Display the prediction
    #st.header("Clasificacion de una red neuronal")
    #st.write(prediction_red_clas)

