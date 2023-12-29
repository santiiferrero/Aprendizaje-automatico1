import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from funciones import season,divide_estaciones,rellenar,eliminar,rellenar_con_cols,eliminar_viento,dummies_elim,rellenar_ultimos,estandarizacion,preprocesamiento, agrego_columnas, MiniBatchGradientDescentRegressor, CustomStandardScaler, NeuralNetworkReg, NeuralNetworkClass
from datetime import date, datetime
from keras.models import load_model
from sklearn.pipeline import Pipeline, make_pipeline
from tensorflow_addons.metrics import RSquare
from keras.models import model_from_json



pipeline1 = joblib.load('pipeline.pkl')
pipeline2 = joblib.load('pipeline.pkl')


pipe = joblib.load('regresion_logistica (2).pkl')
pipe_reg = joblib.load('gradiente-mini-batch (3).pkl')

json_file = open("model_reg.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("red_regresion.h5")

pipeline_reg = pipeline1.set_params(Model = model)

json_file = open("model_clas.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_clas = model_from_json(loaded_model_json)
model_clas.load_weights("red_clasificacion (3).h5")

pipeline_clas = pipeline2.set_params(Model = model_clas)


def get_user_input():
    input_dict = {}

    with st.form(key='my_form'):
        
        selected_date = st.date_input('Date', date.today())
        location = st.text_input('Location (Sydney, SydneyAirport, Canberra, Melbourne y MelbourneAirport)', 'Sidney')
        a1 = st.slider('MinTemp', -20.0, 40.0, 22.0)
        a2 = st.slider('MaxTemp', -10.0, 60.0, 27.0)
        a3 = st.slider('Rainfall', 0.0, 500.0, 5.0)
        a4 = st.slider('Evaporation', 0.0, 30.0, 15.0)
        a5 = st.slider('Sunshine', 0.0, 14.0, 3.6)
        wind_gust_dir = st.text_input('WindGustDir', 'S')
        a6 = st.slider('WindGustSpeed', 0.0, 130.0, 54.0)
        wind_dir_9am = st.text_input('WindDir9am', 'SE')
        wind_dir_3pm = st.text_input('WindDir3pm', 'S')
        a7 = st.slider('WindSpeed9am', 0.0, 130.0, 17.0)
        a8 = st.slider('WindSpeed3pm', 0.0, 130.0, 43.0)
        a9 = st.slider('Humidity9am', 0.0, 100.0, 56.0)
        a10 = st.slider('Humidity3pm', 0.0, 100.0, 92.0)
        a11 = st.slider('Pressure9am', 500.0, 1500.0, 1014.0)
        a12 = st.slider('Pressure3pm', 500.0, 1500.0, 1017.0)
        a13 = st.slider('Cloud9am', 0.0, 10.0, 6.0)
        a14 = st.slider('Cloud3pm', 0.0, 10.0, 8.0)
        a15 = st.slider('Temp9am', -20.0, 40.0, 26.4)
        a16 = st.slider('Temp3pm', -10.0, 60.0, 20.7)
        rain_today = st.text_input('RainToday (Yes, No)', 'No')

        input_dict = {
            'Date': selected_date.strftime('%Y-%m-%d'),
            'Location': location,
            'MinTemp': a1,
            'MaxTemp': a2,
            'Rainfall': a3,
            'Evaporation': a4,
            'Sunshine': a5,
            'WindGustDir': wind_gust_dir,
            'WindGustSpeed': a6,
            'WindDir9am': wind_dir_9am,
            'WindDir3pm': wind_dir_3pm,
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
            'RainToday': rain_today,
            'RainTomorrow': np.nan,
            'RainfallTomorrow': np.nan
        }
        
        user_imput = pd.DataFrame([input_dict])
        
        submit_button = st.form_submit_button(label='Submit')
        st.write(user_imput)


    return user_imput, submit_button


st.title('TP Aprendizaje Automatico')

st.title('Cargar CSV')

 #Widget para cargar un archivo CSV
file = st.file_uploader('Cargar archivo CSV', type=['csv'])

if file is not None:
    # Leer el archivo CSV y mostrar los datos
    df = pd.read_csv(file)
    st.write('Datos del archivo CSV:')
    # Lista de opciones
    options = range(len(df)) 
    # Selección del usuario
    selected_option = st.selectbox('Seleccione un número', options, 0)
    st.write(df.iloc[selected_option, : ])
    
    # Predict csv
    prediction = pipe.predict(df)
    prediction_clas_csv = prediction[selected_option]
    # Display the prediction
    st.header("Rain Tomorrow?")
    if prediction_clas_csv == 1:
        st.write('Yes')
    else: 
        st.write('No')
    
    
    prediction = pipe_reg.predict(df)
    prediction_reg_csv = prediction[selected_option]
    # Display the prediction
    st.header("Rainfall Tomorrow?")
    st.write(str(prediction_reg_csv[0]), ' mm')


    prediction = pipeline_clas.predict(df)
    prediction_red_clas = prediction[selected_option]
    #Display the prediction
    st.header("Clasificacion de la red neuronal")
    if prediction_red_clas > 0.5:
        st.write('Llovera')
    else: 
        st.write('No llovera')

    
    prediction = pipeline_reg.predict(df)
    prediction_red_reg = prediction[selected_option]
    #Display the prediction
    st.header("Regresion de la red neuronal")
    st.write(str(prediction_red_reg[0]), ' mm')


#USUARIO
user_imput, submit_button = get_user_input()

numeric = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm','Pressure3pm','Sunshine','WindGustSpeed','Cloud3pm','Evaporation']

if submit_button:  
    #st.write("Preprocesamiento:")
    #df2 = preprocesamiento(user_imput)
    #df2[numeric] = scaler.transform(df2[numeric])
    #st.write(df2)
    
# Prediccion
    prediction = pipe.predict(user_imput)
    prediction_clas = prediction[0] 
    
    prediction = pipe_reg.predict(user_imput)
    prediction_reg = prediction[0] 

    prediction = pipeline_clas.predict(user_imput)
    prediction_red_clas = prediction[0]
    
    prediction = pipeline_reg.predict(user_imput)
    prediction_red_reg = prediction[0]

    
# Display the prediction
    st.header("Rain Tomorrow?")
    if prediction_clas == 1:
        st.write('Yes')
    else: 
        st.write('No')
    
    # Display the prediction
    st.header("Rainfall Tomorrow?")
    st.write(str(prediction_reg[0]), ' mm')

    
    #Display the prediction
    st.header("Clasificacion de la red neuronal")
    if prediction_red_clas > 0.5:
        st.write('Llovera')
    else: 
        st.write('No llovera')

    #Display the prediction
    st.header("Regresion de la red neuronal")
    st.write(str(prediction_red_reg[0]), ' mm')

