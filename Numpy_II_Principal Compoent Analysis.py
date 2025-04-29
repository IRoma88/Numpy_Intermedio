# -*- coding: utf-8 -*-

#Principal Compoent Analysis (PCA) con Numpy

El analisis de comoponentes principales es una técnica para reducción lineal de la dimensionalidad; es decir, conseguir una representación que conserve la mayor cantidad de información posible utilizando mucho menos datos para obtenerla.

##Obtención de datos y visualización.

Para poder realizar un analisis de componentes principales es fundamental tener unos datos con los que trabajar, en este caso utilizaremos una base de datos de 3 tipos distintos de lirios. Donde tenemos 50 muestras de cada uno de los 3 tipos.

Dado que utilizaremos pandas y matploblib, pertenecientes a cursos siguientes, este paso te lo daremos hecho.

Para que la curiosidad no mate al gato, hemos comentado para que sirve cada linea  utilizada. De momento basta con que ejecutes ambas celdas, no hace falta que las comprendas todavía.
"""

#@title Esta celda carga los datos desde UCI Machine Learning Repository desde un CSV y lo convierte a Dataframe. Puedes ejecutarla de nuevo si quieres.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None) #Mediante el metodo read_csv de pandas convertimos a dataframe la base de datos
iris.columns = ["sepal_length","sepal_width",
                'petal_length','petal_width','species'] #Nombramos cada columna con los datos que contienen
iris.dropna(how='all', inplace=True) #Eliminamos de la tabla los valores vacíos
iris.head() #Mostramos los 5 primeros datos de la tabla para comprobar que se está ejecutando correctamente

#@title Esta celda permite visualizar los datos obtenidos en la celda anterior. Puedes ejecutarla de nuevo si quieres.
plt.style.use("ggplot") #Estilo de gráfico a usar, en este caso de dispersión
plt.rcParams["figure.figsize"] = (12,8) #Tamaño de la ventana que mostrará el gráfico
sns.scatterplot(x = iris.sepal_length, y=iris.sepal_width,
               hue = iris.species, style=iris.species) #Nombre de los ejes y estilo de los puntos

#@title Transformar los datos del dataframe a listas, una con los datos numéricos y otra que almacene a que especie pertenecen. Puedes ejecutarla de nuevo si quieres.
X = iris.iloc[:, 0:4].values #Convertimos los valores (de la longitud del sépalo, ancho de sépalo, longitud de pétalo y ancho de pétalo) del dataframe a una lista de listas de 4 valores
y = iris.species.values #Creamos una lista que identifique a que especie pertenece la lista de 4 valores de la lista anterior
#print(y,'\n'X)

print('Datos preparados correctamente')
#Descomenta el print anterior si deseas ver los datos de la lista.

"""## Estandarizar los datos.

Antes de aplicar PCA es necesario estandarizar los datos para que tengan una media de 0 y una variación estándar de 1. De esta forma centramos los datos en el eje de coordenadas, dejando el punto (0,0) en el centro.

Crea la clase PCA que contenga el método estandarizar_datos.

Este método debe:

* Recibir un array como parámetro.

* Obtener la media y la derivación estandar de cada columna.

* Crear un nuevo array en el que a cada elemento se le reste la media de su columna y el resultado se divida entre la desviación estándar de su columna.

* Devolver el nuevo array una vez estandarizados todos los valores.
"""

class PCA:

  def estandarizar_datos(self,arr):
      self.arr=arr

      # Calcula la media y la desviación estándar de cada columna
      mean = np.mean(arr, axis=0)
      std = np.std(arr, axis=0)

      # Estandariza los datos restando la media y dividiendo por la desviación estándar
      standardized_arr = (arr - mean) / std

      return standardized_arr

result=PCA().estandarizar_datos(X)
print(result)

# @title Test para comprobar el método estandarizar datos
def test_pca_estandarizar_datos():
    # Crear una instancia de la clase PCA
    pca = PCA()

    # Datos de ejemplo
    datos = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # Resultado esperado después de la estandarización
    resultado_esperado = np.array([[-1.22474487, -1.22474487, -1.22474487],
                                   [ 0.        ,  0.        ,  0.        ],
                                   [ 1.22474487,  1.22474487,  1.22474487]])

    # Aplicar la estandarización
    datos_estandarizados = pca.estandarizar_datos(datos)

    # Comprobar si el resultado es igual al esperado con cierta tolerancia
    tolerancia = 1e-6
    if np.allclose(datos_estandarizados, resultado_esperado, atol=tolerancia):
        print('Correcto')
    else:
        print('Incorrecto')

# Ejecutar el test
test_pca_estandarizar_datos()

"""##Cálculo de los vectores propios y los valores propios (Eigenvectors and Eigenvalues)

Vamos a añadir el método descomp_prop a la clase PCA creada en el paso anterior.

Este método descomp_prop debe:

* Recibir como parámetro un array

* Estandarizar sus valores

* Calcular la matriz transpuesta

* Calcular sus vectores propios y valores propios

* Devolver los vectores propios y valores propios


"""

class PCA:

  def estandarizar_datos(self,arr):
      self.arr=arr
      mean = np.mean(arr, axis=0)
      std = np.std(arr, axis=0)

      # Estandariza los datos restando la media y dividiendo por la desviación estándar
      standardized_arr = (arr - mean) / std

      return standardized_arr


  def descomp_prop(self, datos):
      # Estandarizar los datos
      datos_estandarizados = self.estandarizar_datos(datos)


      # estand_T = np.transpose(datos_estandarizados)
      # Calcular la matriz de covarianza
      covarianza_matrix = np.cov(datos_estandarizados, rowvar=False)

      # Calcular los vectores propios y valores propios
      valores_propios, vectores_propios = np.linalg.eig(covarianza_matrix)

      return vectores_propios, valores_propios


PCA().descomp_prop(X)

datos = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
PCA().descomp_prop(datos)

# @title Test para comprobar el método eigenvectors y eigenvalues

def test_pca_descomp_prop():
    # Crear una instancia de la clase PCA
    pca = PCA()

    # Datos de ejemplo
    datos = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # Resultado esperado para vectores propios y valores propios
    vectores_propios_esperados = np. array([[-0.81649658,  0.57735027,  0.        ],
                                          [ 0.40824829,  0.57735027, -0.70710678],
                                          [ 0.40824829,  0.57735027,  0.70710678]])

    valores_propios_esperados = np.array([0. , 4.5, 0. ])

    # Aplicar la descomposición de valores propios y vectores propios
    vectores_propios, valores_propios = pca.descomp_prop(datos)

    # Comprobar si los resultados son iguales a los esperados con cierta tolerancia
    tolerancia = 1e-6
    if np.allclose(vectores_propios, vectores_propios_esperados, atol=tolerancia) and \
       np.allclose(valores_propios, valores_propios_esperados, atol=tolerancia):
        print('Correcto')
    else:
        print('Incorrecto')

# Ejecutar el test
test_pca_descomp_prop()

"""##Transformación PCA

Por último debes añadir el método analisis_comp_princ que reciba por parámetro un array y devuelva otro array de menor dimension lineal, que conserve el 95% de los datos originales (el 95% de la varianza).

Para ello el método debe:

* Calcular la varianza explicada por cada componente
* Calcular la varianza acumulada para conocer con cuales de los componentes se explica el 95%.
* Calcular la matriz de proyección, es la matriz de los vectores propios con tantas columnas como valores propios sena necesarios para representar el 95% de los datos.
* Calcular la transformación PCA de la matriz original mediante la multiplicación matricial de la matriz original y la matriz de proyección.

* Devolver la transformación PCA de la matriz original.
"""

import numpy as np

class PCA:

    def estandarizar_datos(self, arr):
        """
        Estandariza los datos para tener media cero y desviación estándar uno.

        Parameters:
        arr (numpy.ndarray): Array de datos a estandarizar.

        Returns:
        numpy.ndarray: Datos estandarizados.
        """
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        standardized_arr = (arr - mean) / std
        return standardized_arr

    def descomponer_matriz_covarianza(self, datos):
        """
        Calcula la matriz de covarianza y obtiene los valores propios y vectores propios.

        Parameters:
        datos_estandarizados (numpy.ndarray): Datos estandarizados.

        Returns:
        tuple: Vectores propios y valores propios.
        """
        # Estandarizar los datos
        datos_estandarizados = self.estandarizar_datos(datos)
        cov_matrix = np.cov(datos_estandarizados, rowvar=False)
        valores_propios, vectores_propios = np.linalg.eig(cov_matrix)
        return vectores_propios, valores_propios

    def analisis_comp_princ(self, datos_estandarizados, varianza_umbral=0.95):
        """
        Realiza el análisis de PCA y proyecta los datos en los componentes principales.

        Parameters:
        datos (numpy.ndarray): Datos originales.
        varianza_umbral (float): Umbral de varianza acumulada para seleccionar los componentes principales.

        Returns:
        numpy.ndarray: Datos proyectados en los componentes principales.
        """

        # Descomponer la matriz de covarianza
        vectores_propios, valores_propios = self.descomponer_matriz_covarianza(datos_estandarizados)

        # Calcular la varianza explicada por cada componente
        varianza_explicada = valores_propios / np.sum(valores_propios)
        varianza_acumulada = np.cumsum(varianza_explicada)

        # Determinar el número de componentes necesarios para el umbral de varianza
        num_componentes = np.argmax(varianza_acumulada >= varianza_umbral) + 1

        # Seleccionar los primeros 'num_componentes' vectores propios como matriz de proyección
        matriz_proyeccion = vectores_propios[:, :num_componentes]

        # Proyectar los datos estandarizados en los componentes principales
        transformacion_pca = np.dot(datos_estandarizados, matriz_proyeccion)

        # Devolver la varianza explicada acumulada para verificar el porcentaje de varianza explicado
        return np.round(transformacion_pca,4)

print(PCA().analisis_comp_princ(X))

#@title Comprueba el resultado de la transformación PCA
def check3():
  if list(PCA().analisis_comp_princ(X)[0,:])==[2.6692, -5.1809] and list(PCA().analisis_comp_princ(X)[-1,:])==[6.2744, -5.1987] and PCA().analisis_comp_princ(X).mean()==-0.024552333333333252 and len(PCA().analisis_comp_princ(X))==150:
    return 'Correcto'
  else:
    return 'Incorrecto'
check3()

# @title Test analisis PCA


def test_pca_analisis_comp_princ():
    # Crear una instancia de la clase PCA
    pca = PCA()

    # Datos de ejemplo
    datos = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # Aplicar el análisis de componentes principales
    transformacion_pca = pca.analisis_comp_princ(datos)

    # Comprobar si la transformación tiene la forma correcta
    assert transformacion_pca.shape[1] < datos.shape[1], "Error: El número de columnas después de la transformación debe ser menor."

    # Comprobar si se conserva el 95% de la varianza
    varianza_original = np.var(datos, axis=0)
    varianza_transformada = np.var(transformacion_pca, axis=0)
    varianza_explicada = 1 - np.sum(varianza_transformada) / np.sum(varianza_original)
    assert varianza_explicada <= 0.05, f"Error: Se esperaba conservar al menos el 95% de la varianza, pero se conservó {varianza_explicada * 100:.2f}%."

    print('Correcto')

# Ejecutar el test
test_pca_analisis_comp_princ()



# Mostrar el token
print("Token generado:")
print(token)
