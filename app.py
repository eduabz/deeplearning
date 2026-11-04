from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "models" / "cnn_basica.keras"

CLASES = [
    ("with_mask", "Mascarilla bien puesta"),
    ("without_mask", "Sin mascarilla"),
    ("mask_weared_incorrect", "Mascarilla mal puesta"),
]


@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(RUTA_MODELO, compile=False)


def preparar_imagen(imagen):
    # Mismo tratamiento de transparencia y tamaño que en el ETL.
    imagen_rgba = imagen.convert("RGBA")
    fondo = Image.new("RGBA", imagen_rgba.size, "white")

    imagen_rgb = Image.alpha_composite(
        fondo, imagen_rgba
    ).convert("RGB")

    entrada = imagen_rgb.resize(
        (128, 128),
        Image.Resampling.BILINEAR,
    )

    datos = np.asarray(entrada, dtype=np.float32)
    datos = np.expand_dims(datos, axis=0)

    # El propio modelo incluye la capa Rescaling(1/255).
    return imagen_rgb, datos


st.set_page_config(
    page_title="Clasificador de mascarillas",
    layout="centered",
)

st.title("Clasificador de uso de mascarilla")
st.write(
    "Sube una imagen con **un solo rostro centrado**. "
    "El modelo clasifica cómo se usa la mascarilla."
)
st.caption(
    "Prototipo académico: clasifica rostros; "
    "no localiza rostros automáticamente en fotografías completas."
)

if not RUTA_MODELO.exists():
    st.error("No se encontró el modelo guardado en models/cnn_basica.keras")
    st.stop()

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["png", "jpg", "jpeg"],
    help="Usa un recorte donde se vea claramente un rostro.",
)

if archivo is not None:
    try:
        with Image.open(archivo) as imagen:
            imagen_visible, entrada_modelo = preparar_imagen(imagen)
    except (UnidentifiedImageError, OSError):
        st.error("No se pudo leer la imagen. Prueba con otro archivo.")
        st.stop()

    st.image(
        imagen_visible,
        caption="Imagen seleccionada",
        width=300,
        alt="Rostro cargado para clasificar",
    )

    if st.button("Clasificar rostro", type="primary"):
        modelo = cargar_modelo()
        probabilidades = modelo.predict(
            entrada_modelo, verbose=0
        )[0]

        indice = int(np.argmax(probabilidades))
        nombre_resultado = CLASES[indice][1]

        st.subheader(f"Resultado: {nombre_resultado}")
        st.write(
            "Probabilidades estimadas por el modelo "
            "para cada clase:"
        )

        for numero, (_, nombre) in enumerate(CLASES):
            valor = float(probabilidades[numero])
            st.write(f"{nombre}: {valor:.1%}")
            st.progress(valor)