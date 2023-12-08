# funciones
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

def season(date):
    date = datetime.strptime(date, '%Y-%m-%d')
    if date.month in [12, 1, 2]:
        return 'Verano'
    elif date.month in [3, 4, 5]:
        return 'Otoño'
    elif date.month in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'

def divide_estaciones(df):
  # Aplicamos la funcion para crear la columna season
  df['season'] = df['Date'].apply(season)
  return df

def rellenar(df):
    #Rellenemos valores faltanes de MaxTemp con Temp3pm
    df['MaxTemp'] = df['MaxTemp'].fillna(df['Temp3pm'])

    #Rellenemos valores faltanes de MinTemp con Temp9am
    df['MinTemp'] = df['MinTemp'].fillna(df['Temp9am'])

    #Rellenemos valores faltanes de Pressure3pm con Pressure9am
    df['Pressure3pm'] = df['Pressure3pm'].fillna(df['Pressure9am'])

    #Rellenemos valores faltanes de Cloud3pm con Cloud9am
    df['Cloud3pm'] = df['Cloud9am'].fillna(df['Cloud9am'])

    #Rellenemos valores faltanes de WindGustSpeed con WindSpeed9am y WindSpeed3pm
    df['WindGustSpeed'] = df['WindGustSpeed'].fillna(df['WindSpeed9am'])
    df['WindGustSpeed'] = df['WindGustSpeed'].fillna(df['WindSpeed3pm'])

    return df

def eliminar(df):
    #Eliminamos Temp9am y Temp3pm
    df = df.drop(['Temp9am','Temp3pm',], axis=1)

    #Eliminamos Pressure9am
    df = df.drop(['Pressure9am'], axis=1)

    #Eliminamos WindSpeed9am y WindSpeed3pm
    df = df.drop(['WindSpeed9am','WindSpeed3pm'], axis=1)

    #Eliminamos Cloud9am
    df = df.drop(['Cloud9am'], axis=1)

    #Eliminamos Humidity9am
    df = df.drop(['Humidity9am'], axis=1)

    return df

def rellenar_con_cols(df):
    #Debido a que las variables Sunshine y Cloud3pm tienen una correlacion negativa, rellenaremos los valores nulos que podamos con el negativo de la otra
    df['Sunshine'].fillna(-df['Cloud3pm'], inplace=True)

    df['Cloud3pm'].fillna(-df['Sunshine'], inplace=True)

    #Lo mismo haremos con Evaporation y MaxTemp, solo que estas se correlacionan positivamente
    df['Evaporation'].fillna(df['MaxTemp'], inplace=True)

    df['MaxTemp'].fillna(df['Evaporation'], inplace=True)

    return df

def eliminar_viento(df):
  #Eliminamos WindDir9am y WindDir3pm
  df = df.drop(['WindDir9am','WindDir3pm'], axis=1)
  return df

def dummies_elim(df):
  #Aplicamos Dummies para las variables categoricas que tendremos en cuenta para nuestra prediccion

  df = pd.get_dummies(df, columns=['season'])

  df = pd.get_dummies(df, columns=['WindGustDir'])

  df = pd.get_dummies(df, columns=['RainToday'])

  #Eliminamos Date y Location ya que son variables que no son de nuestro interes
  df = df.drop(['Date', 'Location'], axis=1)

  return df

def rellenar_ultimos(df):
  # Lista de columnas numéricas y categóricas
  columnas_numericas = ['Sunshine', 'WindGustSpeed', 'Humidity3pm', 'Pressure3pm','Rainfall', 'MinTemp',  'Cloud3pm', 'MaxTemp', 'Evaporation']
  columnas_categoricas = ['WindGustDir']

  # Imputador para rellenar valores nulos en columnas numéricas con la media
  numeric_imputer = SimpleImputer(strategy='mean')
  df[columnas_numericas] = numeric_imputer.fit_transform(df[columnas_numericas])

  # Imputador para rellenar valores nulos en columnas categóricas con 'Sin dato'
  categorical_imputer = SimpleImputer(strategy='constant', fill_value='Sin dato')
  df[columnas_categoricas] = categorical_imputer.fit_transform(df[columnas_categoricas])
  return df

def agrego_columnas(df):
    columns_to_add = ['season_Invierno', 'season_Otoño', 'season_Primavera', 'season_Verano',
                      'WindGustDir_E', 'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N',
                      'WindGustDir_NE', 'WindGustDir_NNE', 'WindGustDir_NNW',
                      'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE', 'WindGustDir_SSE',
                      'WindGustDir_SSW', 'WindGustDir_SW', 'WindGustDir_Sin dato',
                      'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW', 'RainToday_Yes','RainToday_No']
    # Obtener las columnas que no se encuentran en el DataFrame
    columns_not_in_df = [col for col in columns_to_add if col not in df.columns]
    
    # Agregar las columnas faltantes con valor 0
    for new_column in columns_not_in_df:
        df[new_column] = 0
        
    df = df.drop(['RainToday_No'], axis=1)

    df_deseado = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                  'WindGustSpeed', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm',
                  'season_Invierno', 'season_Otoño', 'season_Primavera', 'season_Verano',
                  'WindGustDir_E', 'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N',
                  'WindGustDir_NE', 'WindGustDir_NNE', 'WindGustDir_NNW',
                  'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE', 'WindGustDir_SSE',
                  'WindGustDir_SSW', 'WindGustDir_SW', 'WindGustDir_Sin dato',
                  'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW', 'RainToday_Yes']


    df = df[df_deseado]
    return df



def estandarizacion(df):
  #Columnas numericas que estandarizaremos
  numeric = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm','Pressure3pm','Sunshine','WindGustSpeed','Cloud3pm','Evaporation']

  # Estandarizamos utilizando StandardScaler de sklearn
  scaler = StandardScaler()

  # estandarizamos las columnas numericas
  df_scaled = df.copy()
  df_scaled[numeric] = scaler.fit_transform(df[numeric])
  return df_scaled


def preprocesamiento(df):
    #Dvidimos las estaciones
    df = divide_estaciones(df)

    #Rellenamos columnas con nulos
    df = rellenar(df)

    #Eliminamos features
    df = eliminar(df)

    #Rellenamos
    df = rellenar_con_cols(df)

    #Elimiamos viento
    df = eliminar_viento(df)

    #Por ultimo, rellenamos los valores que nos quedaron Nulos
    df = rellenar_ultimos(df)

    #Aplicamos dummies
    df = dummies_elim(df)

    #Agregamos columnas faltante
    df = agrego_columnas(df)

    return df
