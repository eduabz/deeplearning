"""Entrena y evalúa la CNN básica sin depender de un notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


BASE = Path(__file__).resolve().parent
DATOS = BASE / "data" / "processed"
MODELO = BASE / "models" / "cnn_basica.keras"
FIGURAS = BASE / "outputs" / "figuras"

CLASES = ["with_mask", "without_mask", "mask_weared_incorrect"]
NOMBRES = ["Bien puesta", "Sin mascarilla", "Mal puesta"]


def cargar_datos():
    for conjunto in ["entrenamiento", "validacion"]:
        if not (DATOS / conjunto).is_dir():
            raise FileNotFoundError(
                f"Falta data/processed/{conjunto}. "
                "Ejecuta primero: python preparar_dataset.py"
            )

    entrenamiento = tf.keras.utils.image_dataset_from_directory(
        DATOS / "entrenamiento",
        class_names=CLASES,
        image_size=(128, 128),
        batch_size=32,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    validacion = tf.keras.utils.image_dataset_from_directory(
        DATOS / "validacion",
        class_names=CLASES,
        image_size=(128, 128),
        batch_size=32,
        label_mode="int",
        shuffle=False,
    )
    return entrenamiento, validacion


def crear_modelo():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def guardar_curvas(historial):
    epocas = range(1, len(historial.history["loss"]) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

    for eje, metrica, titulo in zip(
        ejes, ["loss", "accuracy"], ["Pérdida", "Accuracy"]
    ):
        eje.plot(epocas, historial.history[metrica], "o-", label="Entrenamiento")
        eje.plot(
            epocas, historial.history["val_" + metrica],
            "s-", label="Validación"
        )
        eje.set_title(titulo)
        eje.set_xlabel("Época")
        eje.legend()

    fig.savefig(FIGURAS / "cnn_basica_py_curvas.png", dpi=160)
    plt.close(fig)


def guardar_matriz(matriz):
    porcentajes = matriz / matriz.sum(axis=1, keepdims=True)
    fig, eje = plt.subplots(figsize=(6, 5), layout="constrained")
    imagen = eje.imshow(porcentajes, cmap="Blues", vmin=0, vmax=1)

    for fila in range(3):
        for columna in range(3):
            porcentaje = porcentajes[fila, columna]
            eje.text(
                columna, fila,
                f"{matriz[fila, columna]}\n{porcentaje:.1%}",
                ha="center", va="center",
                color="white" if porcentaje > 0.5 else "black",
            )

    eje.set_xticks(range(3), NOMBRES, rotation=20, ha="right")
    eje.set_yticks(range(3), NOMBRES)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title("CNN básica: validación")
    fig.colorbar(imagen, ax=eje, label="Proporción de la clase real")
    fig.savefig(FIGURAS / "cnn_basica_py_matriz.png", dpi=160)
    plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(42)
    entrenamiento, validacion = cargar_datos()
    modelo = crear_modelo()
    modelo.summary()

    historial = modelo.fit(
        entrenamiento, validation_data=validacion, epochs=10
    )

    etiquetas_reales = np.concatenate([
        etiquetas.numpy() for _, etiquetas in validacion
    ])
    probabilidades = modelo.predict(validacion, verbose=0)
    predicciones = np.argmax(probabilidades, axis=1)
    matriz = confusion_matrix(
        etiquetas_reales, predicciones, labels=[0, 1, 2]
    )

    print("\nRESULTADOS EN VALIDACIÓN")
    print(classification_report(
        etiquetas_reales, predicciones,
        labels=[0, 1, 2], target_names=CLASES,
        digits=4, zero_division=0,
    ))
    print("Matriz de confusión:\n", matriz)

    MODELO.parent.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    modelo.save(MODELO)
    guardar_curvas(historial)
    guardar_matriz(matriz)

    print("Modelo guardado en:", MODELO)
    print("Gráficas guardadas en:", FIGURAS)
    print("El conjunto de prueba no se utilizó.")


if __name__ == "__main__":
    main()
