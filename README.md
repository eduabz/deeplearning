# Clasificación del uso de mascarilla con Deep Learning

Proyecto académico de clasificación de rostros recortados en tres categorías: `with_mask` (mascarilla bien puesta), `without_mask` (sin mascarilla) y `mask_weared_incorrect` (mascarilla mal puesta). Se prepararon los datos y se compararon dos redes neuronales convolucionales (CNN) hechas con TensorFlow. La CNN básica se integró en una interfaz web con Streamlit.

**[Abrir el clasificador en línea](https://eduabz-mascarillas.streamlit.app/)**

**Alcance:** el modelo clasifica un rostro ya recortado y centrado. La aplicación no localiza rostros en una fotografía completa. El conjunto de prueba permanece reservado para una evaluación posterior.

## Archivos principales

| Archivo | Contenido |
|---|---|
| [`notebooks/01_etl_exploracion.ipynb`](notebooks/01_etl_exploracion.ipynb) | Exploración de imágenes y XML, decisiones de limpieza y división por fotografía. |
| [`notebooks/02_modelos_cnn.ipynb`](notebooks/02_modelos_cnn.ipynb) | Experimentos originales ejecutados en el equipo local, con sus salidas guardadas. |
| [`notebooks/02_modelos_cnn_colab.ipynb`](notebooks/02_modelos_cnn_colab.ipynb) | Versión autónoma para Google Colab: descarga el dataset, prepara recortes, entrena y compara ambas CNN. |
| [`preparar_dataset.py`](preparar_dataset.py) | ETL ejecutable desde la carpeta del proyecto. |
| [`app.py`](app.py) | Interfaz web para clasificar un rostro. |
| [`reports/reporte_deep_learning_mascarillas.pdf`](reports/reporte_deep_learning_mascarillas.pdf) | Reporte con metodología, resultados, gráficas y demostración de la interfaz. |

## Datos y preparación

La fuente es [Face Mask Detection de Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data). Contiene 853 imágenes PNG y 853 anotaciones XML. Los archivos originales y los 3,603 recortes procesados no se incluyen en el repositorio por su tamaño; el notebook de Colab los descarga y genera automáticamente.

El ETL ajusta las cajas que exceden el borde de la imagen, excluye 469 rostros con menos de 10 píxeles en alguna dimensión, convierte los recortes a RGB y los redimensiona a 128 × 128 píxeles. La división se hace por fotografía original para impedir que rostros de una misma imagen queden en conjuntos distintos:

| Conjunto | Fotografías | Rostros |
|---|---:|---:|
| Entrenamiento | 597 | 2,574 |
| Validación | 128 | 545 |
| Prueba | 128 | 484 |

## Ejecutar en Google Colab

1. Abre [`notebooks/02_modelos_cnn_colab.ipynb`](notebooks/02_modelos_cnn_colab.ipynb) en [Google Colab](https://colab.research.google.com/).
2. Selecciona **Entorno de ejecución → Ejecutar todo**. El notebook instala `kagglehub` si hace falta, descarga el dataset público, prepara los recortes y entrena las dos CNN. Necesita conexión a internet; puede ejecutarse en CPU o GPU.
3. Lee las métricas por clase, las matrices de confusión y las curvas. Los modelos `.keras` se guardan en `/content/face_mask_detection/models` dentro de esa sesión de Colab. Los archivos de una sesión de Colab no permanecen allí indefinidamente; descárgalos si deseas conservarlos.

El notebook publicado incluye salidas de una ejecución local completa. La ejecución en Colab vuelve a calcularlas y puede producir pequeñas diferencias numéricas respecto de la corrida local documentada en el reporte. El notebook no usa el conjunto de prueba para escoger la CNN.

## Ejecutar en el equipo local

Se utilizó Python 3.13.15 en macOS. Crea un entorno virtual e instala `requirements-dev.txt` para los notebooks y el entrenamiento. `requirements.txt` contiene solo las dependencias de la interfaz. Después, coloca el dataset descargado en `data/raw/images/` y `data/raw/annotations/`:

```bash
python -m pip install -r requirements-dev.txt
python preparar_dataset.py
```

Abre `notebooks/02_modelos_cnn.ipynb` para reproducir los experimentos locales. Para iniciar la interfaz con el modelo guardado:

```bash
python -m streamlit run app.py
```

La interfaz espera una imagen con un solo rostro centrado. El archivo `models/cnn_basica.keras` debe estar presente.

## Resultados de validación registrados

| Modelo | Accuracy | F1 macro | Casos de mascarilla mal puesta reconocidos |
|---|---:|---:|---:|
| CNN básica | 0.9468 | 0.7725 | 7 de 16 |
| CNN profunda | 0.9541 | 0.7472 | 4 de 16 |

La CNN profunda obtuvo mayor accuracy general; la básica reconoció más casos de uso incorrecto y obtuvo mejor F1 macro en esta corrida. Por eso se eligió provisionalmente la básica para la interfaz. La clase de mascarilla mal puesta tiene pocos ejemplos, por lo que sus resultados deben interpretarse con cuidado. Consulta el [reporte](reports/reporte_deep_learning_mascarillas.pdf) para el análisis completo.

## Preparación para GitHub

El repositorio incluye el código, los notebooks, el reporte, las figuras y los modelos `.keras` usados en los experimentos. `.gitignore` excluye el entorno virtual, las imágenes originales, los recortes procesados y los archivos internos de planificación. El notebook de Colab descarga y prepara los datos necesarios para reproducir el entrenamiento.
