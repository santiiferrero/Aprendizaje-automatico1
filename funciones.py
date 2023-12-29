# funciones
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

    #Establezco todo a numerico
    df = df.astype(float)

    return df

class MiniBatchGradientDescentRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lr=0.001, epochs=100, batch_size=600):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None

    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]

        # Poner columna de unos a la matriz X
        X = np.hstack((np.ones((n, 1)), X))

        # Inicializar pesos aleatorios
        self.W = np.random.randn(m + 1).reshape(-1, 1)

        train_errors = []
        test_errors = []

        for i in range(self.epochs):

            # Permutación aleatoria de los datos
            permutation = np.random.permutation(n)
            X = X[permutation]
            y = y[permutation]

            for j in range(0, n, self.batch_size):
                # Obtener un lote (mini-batch) de datos
                x_batch = X[j:j+self.batch_size, :]
                y_batch = y[j:j+self.batch_size].reshape(-1, 1)

                prediction = np.matmul(x_batch, self.W)
                error = y_batch - prediction
                train_mse = np.mean(error ** 2)
                train_errors.append(train_mse)

                gradient = -2 * np.matmul(x_batch.T, error) / self.batch_size

                self.W = self.W - (self.lr * gradient)

        return self

    def predict(self, X):
        # Poner columna de unos a la matriz X
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calcula la predicción
        prediction = np.matmul(X, self.W)

        return prediction


#Crearemos una clase para la implementacion de nuestra red neuronal de regresión.
class NeuralNetworkReg(BaseEstimator, TransformerMixin):
    def __init__(self, num_layers=None,
                 n_units_layer_0=None, n_units_layer_1=None,n_units_layer_2=None,n_units_layer_3 =None ,
                 n_units_layer_4=None,n_units_layer_5=None,n_units_layer_6 =None ,
                 n_units_layer_7=None,n_units_layer_8=None,n_units_layer_9 =None,
                epochs=None, batch_size=None):

        self.num_layers = num_layers            # Número de capas ocultas

        self.n_units_layer_0 = n_units_layer_0  # -----------------------------------
        self.n_units_layer_1 = n_units_layer_1
        self.n_units_layer_2 = n_units_layer_2
        self.n_units_layer_3 = n_units_layer_3
        self.n_units_layer_4 = n_units_layer_4    # Número de neuronas en la resp. capa oculta.
        self.n_units_layer_5 = n_units_layer_5
        self.n_units_layer_6 = n_units_layer_6
        self.n_units_layer_7 = n_units_layer_7
        self.n_units_layer_8 = n_units_layer_8
        self.n_units_layer_9 = n_units_layer_9  #------------------------------------------------

        self.epochs = epochs                    # Número de épocas de entrenamiento
        self.batch_size = batch_size            # Tamaño del batch para el entrenamiento
        self.model = None                       # Variable para almacenar el modelo entrenado
        self.loss_history = None                # Variable para almacenar el historial de pérdida durante el entrenamiento

    def fit(self, X, y):
        # Convierte los datos a arrays NumPy si no lo están
        X = np.array(X)
        y = np.array(y)

        # Creamos el modelo de red neuronal
        model_nn = tf.keras.Sequential()

        # Añadimos la primera capa oculta con la cantidad de unidades y función de activación especificadas
        model_nn.add(tf.keras.layers.Dense(self.n_units_layer_0, activation='relu', input_shape=(31,)))

        # Añadimos capas ocultas adicionales según el número de capas especificado
        for i in range(0, self.num_layers + 1):
            # Usamos getattr para obtener el valor de n_units_layer_i dinámicamente
            n_units_layer_i = getattr(self, f'n_units_layer_{i}', None)
            if n_units_layer_i is not None:
                model_nn.add(tf.keras.layers.Dense(n_units_layer_i, activation='relu'))

        # Capa de salida con 1 neurona y función de activación lineal
        model_nn.add(tf.keras.layers.Dense(1, activation='linear'))

        # Compilamos el modelo con el optimizador 'adam', la función de pérdida 'mean_squared_error' y la métrica 'R2'
        model_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=[RSquare()])

        # Entrenamos el modelo y guardamos el historial de pérdida
        history = model_nn.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model_nn                        # Almacenamos el modelo entrenado
        self.loss_history = history.history['loss']  # Almacenamos el historial de pérdida

        return self

    def predict(self, X):
        # Método para hacer predicciones con el modelo entrenado
        predictions = self.model.predict(X)
        return predictions


class NeuralNetworkClass(BaseEstimator, TransformerMixin):
    def __init__(self, num_layers=None,
                 n_units_layer_0=None, n_units_layer_1=None,n_units_layer_2=None,n_units_layer_3 =None ,
                 n_units_layer_4=None,n_units_layer_5=None,n_units_layer_6 =None ,
                 n_units_layer_7=None,n_units_layer_8=None,n_units_layer_9 =None,
                epochs=None, batch_size=None):

        self.num_layers = num_layers            # Número de capas ocultas

        self.n_units_layer_0 = n_units_layer_0  # -----------------------------------
        self.n_units_layer_1 = n_units_layer_1
        self.n_units_layer_2 = n_units_layer_2
        self.n_units_layer_3 = n_units_layer_3
        self.n_units_layer_4 = n_units_layer_4    # Número de neuronas en la resp. capa oculta
        self.n_units_layer_5 = n_units_layer_5
        self.n_units_layer_6 = n_units_layer_6
        self.n_units_layer_7 = n_units_layer_7
        self.n_units_layer_8 = n_units_layer_8
        self.n_units_layer_9 = n_units_layer_9  #------------------------------------------------

        self.epochs = epochs                    # Número de épocas de entrenamiento
        self.batch_size = batch_size            # Tamaño del batch
        self.model = None                       # Variable para almacenar el modelo entrenado
        self.loss_history = None                # Variable para almacenar el historial de pérdida durante el entrenamiento

    def fit(self, X, y):
        #Convierte los datos en float si no lo estan
        X = X.astype('float32')
        y = y.astype('float32')

        # Convierte los datos a arrays NumPy si no lo están
        X = np.array(X)
        y = np.array(y)

        # Creamos el modelo de red neuronal
        model_nn = tf.keras.Sequential()

        # Añadimos la primera capa oculta con la cantidad de unidades y función de activación especificadas
        model_nn.add(tf.keras.layers.Dense(self.n_units_layer_0, activation='sigmoid', input_shape=(31,)))

       # Añadimos capas ocultas adicionales según el número de capas especificado
        for i in range(0, self.num_layers + 1):
            # Usamos getattr para obtener el valor de n_units_layer_i dinámicamente
            n_units_layer_i = getattr(self, f'n_units_layer_{i}', None)
            if n_units_layer_i is not None:
                model_nn.add(tf.keras.layers.Dense(n_units_layer_i, activation='sigmoid'))

        # Capa de salida con 1 neurona y función de activación sigmoide.
        model_nn.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))

        # Compilamos el modelo con el optimizador 'adam', la función de pérdida 'binary_crossentropy' y la métrica 'F1-Score'.
        model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score(name="f1_metric")])

        # Entrenamos el modelo y guardamos el historial de pérdida
        history = model_nn.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model_nn                        # Almacenamos el modelo entrenado
        self.loss_history = history.history['loss']  # Almacenamos el historial de pérdida

        return self

    def predict(self, X):
        # Método para hacer predicciones con el modelo entrenado
        predictions = self.model.predict(X)
        return predictions


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train, y=None):
        self.scaler.fit(X_train)
        return self

    def transform(self, X):
        X_transformed = self.scaler.transform(X)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        return X

