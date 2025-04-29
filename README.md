# Análisis de Componentes Principales (PCA) con NumPy

Este proyecto implementa desde cero el algoritmo de PCA (Principal Component Analysis) usando únicamente `NumPy`, con apoyo de `pandas`, `matplotlib` y `seaborn` para carga y visualización de datos.

El objetivo es reducir la dimensionalidad de un conjunto de datos (Iris dataset) preservando el 95% de la varianza original.

## 📁 Estructura del proyecto

- `pca.py`: Contiene la clase `PCA` con los métodos:
  - `estandarizar_datos`: Estandariza los datos con media 0 y desviación estándar 1.
  - `descomponer_matriz_covarianza`: Calcula la matriz de covarianza y obtiene los eigenvalores y eigenvectores.
  - `analisis_comp_princ`: Reduce la dimensionalidad de los datos conservando el 95% de la varianza.

- `tests.py`: Pruebas automáticas para validar el funcionamiento correcto de cada método.

- `colab_pca.ipynb`: Notebook con el desarrollo paso a paso del análisis PCA y visualización.

## 📊 Dataset utilizado

Se usa el conjunto de datos de *Iris* de UCI Machine Learning Repository, con 150 muestras de 3 tipos distintos de lirios y 4 características por flor:

- Longitud del sépalo
- Ancho del sépalo
- Longitud del pétalo
- Ancho del pétalo

## 🔧 Requisitos

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn

Puedes instalar las dependencias con:

```bash
pip install numpy pandas matplotlib seaborn
````

## 🚀 Cómo ejecutar
Abre el notebook colab_pca.ipynb en Google Colab o Jupyter.

Ejecuta las celdas una por una para seguir el análisis PCA.

También puedes ejecutar los tests unitarios para validar el código:
````bash
python tests.py
````

## 🧪 Tests incluidos
El proyecto contiene pruebas automáticas para:

Verificar la estandarización de datos

Comprobar la descomposición en vectores y valores propios

Validar que se conserva el 95% de la varianza en la transformación PCA

## 📄 Licencia
Este proyecto está bajo licencia MIT. Puedes usar, modificar y distribuir libremente.


