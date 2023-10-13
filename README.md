# Trabajo Práctico Aprendizaje Automatico 1

En este repo encontraremos el trabajo práctico de la asignatura Aprendizaje Automatico 1, de la Tecnicatura Univesitaria en Inteligencia Artificial.

Contiene 2 archivos:
 - `WeatherAUS.csv` : dataset que utilizaremos
 - `TrabajoPrácticoIntegradorAA1.ipynb` : Colab con el codigo python y explicacion del proyecto.


El dataset contiene aproximadamente 10 años de observaciones diarias de variables climáticas de Australia: temperatura, dirección y velocidad del viento, humedad, presión, nubosidad, y cantidad de lluvia en mm.
Tras observar los datos del día de hoy, el objetivo es predecir las variables target:
                                                                               
-   `RainFallTomorrow`: cantidad de lluvia del día posterior a la observación.   **Problema de Regresión**.
-   `RainTomorrow`: si el día siguiente llueve o no llueve. **Problema de Clasificación**.
