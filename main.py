import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn import linear_model


df = pd.read_csv("bitacora.csv")

x_multiple = df.iloc[: , 2:5]



y_multiple = df.iloc[:,1]
#La columna 1 (C) son las calorías que es la Y
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split (x_multiple, y_multiple,
test_size=0.6)


lr_multiple = linear_model.LinearRegression (fit_intercept=False)
lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)
print()
print()
print()
print ("Datos del modelo de regresión lineal múltiple")
print()

print("Valores de lo coeficientes B1, B2....:")
print(lr_multiple.coef_)
print()

print("Valor de la intersección o coeficiente 'b0': " )
print ( lr_multiple.intercept_)
print()
print("Precisión del modelo")
print(lr_multiple.score(X_train, y_train))
print()
print()
print()



reg = smf.ols(" C1~ C2 + G + P ", data = df)
res = reg.fit()
print(res.summary())


import pandas as pd


beta_zero = True  # Cambiar esto a False para que el Coeficiente Beta sea 0
nombre_archivo = 'datos.csv'  # Cambiar esto por el nombre de su archivo
columna_calorias = 1
columna = ['Carbohidratos', 2]  # Cambiar esto por el nombre y número de columna de su variable independiente

    # Cargar los datos en un DataFrame
df = pd.read_csv("bitacora.csv")

datos_x = df.iloc[:, columna[1]]  # Columna de la variable independiente
datos_y = df.iloc[:, columna_calorias]  # Columna de la variable dependiente

# Crear el modelo de regresión lineal sencilla
regresion_sencilla = LinearRegression(fit_intercept=beta_zero)

    # Entrenar el modelo
regresion_sencilla.fit(datos_x.values.reshape(-1, 1), datos_y)

    # Imprimir los coeficientes
print('Coeficientes: ', regresion_sencilla.coef_)

    # Imprimir intercepción
print('Intercepción: ', regresion_sencilla.intercept_)



    # Graficar los datos y la recta de regresión
plt.scatter(datos_x, datos_y)
plt.plot(datos_x, regresion_sencilla.predict(datos_x.values.reshape(-1, 1)), color='red')

    # Agregar título y nombres de ejes
plt.title(f'Regresión lineal sencilla entre Calorías y {columna[0]}')
plt.xlabel(columna[0])
plt.ylabel('Calorías')

    # Mostrar la gráfica
plt.show()






