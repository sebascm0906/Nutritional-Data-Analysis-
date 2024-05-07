import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Configuraciones
beta_zero = True  # Cambiar esto a False para que el Coeficiente Beta sea 0
nombre_archivo = 'bitacora.csv'  # Cambiar esto por el nombre de su archivo
columna_calorias = 1
columna = ['C2', 1]  # Cambiar esto por el nombre y número de columna de su variable independiente


def main():
    # Cargar los datos en un DataFrame
    df = pd.read_csv(nombre_archivo)

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

    # Imprimir la precisión del modelo
    print('Precisión del modelo: ', regresion_sencilla.score(datos_x.values.reshape(-1, 1), datos_y))

    # Graficar los datos y la recta de regresión
    plt.scatter(datos_x, datos_y)
    plt.plot(datos_x, regresion_sencilla.predict(datos_x.values.reshape(-1, 1)), color='red')

    # Agregar título y nombres de ejes
    plt.title(f'Regresión lineal sencilla entre Calorías y {columna[0]}')
    plt.xlabel(columna[0])
    plt.ylabel('Calorías')

    # Mostrar la gráfica
    plt.show()


if __name__ == '__main__':
    main()