from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parent
IMAGENES = BASE / "data" / "raw" / "images"
ANOTACIONES = BASE / "data" / "raw" / "annotations"
SALIDA = BASE / "data" / "processed"

CLASES = {
    "with_mask",
    "without_mask",
    "mask_weared_incorrect",
}


def leer_anotaciones():
    registros = []

    for ruta_xml in sorted(ANOTACIONES.glob("*.xml")):
        raiz = ET.parse(ruta_xml).getroot()
        nombre_imagen = raiz.findtext("filename")
        ruta_imagen = IMAGENES / nombre_imagen

        if not ruta_imagen.exists():
            raise FileNotFoundError(ruta_imagen)

        ancho_xml = int(raiz.findtext("size/width"))
        alto_xml = int(raiz.findtext("size/height"))

        with Image.open(ruta_imagen) as imagen:
            if imagen.size != (ancho_xml, alto_xml):
                raise ValueError(
                    f"Dimensiones distintas en {nombre_imagen}"
                )

        for numero, objeto in enumerate(
            raiz.findall("object"), start=1
        ):
            clase = objeto.findtext("name")

            if clase not in CLASES:
                raise ValueError(
                    f"Clase desconocida en {ruta_xml.name}: {clase}"
                )

            caja = objeto.find("bndbox")
            xmin = int(caja.findtext("xmin"))
            ymin = int(caja.findtext("ymin"))
            xmax = int(caja.findtext("xmax"))
            ymax = int(caja.findtext("ymax"))

            # Ajustar la caja a los límites reales de la imagen.
            xmin_usado = max(0, xmin)
            ymin_usado = max(0, ymin)
            xmax_usado = min(ancho_xml, xmax)
            ymax_usado = min(alto_xml, ymax)

            ancho_rostro = xmax_usado - xmin_usado
            alto_rostro = ymax_usado - ymin_usado

            if ancho_rostro <= 0 or alto_rostro <= 0:
                raise ValueError(
                    f"Caja sin área en {ruta_xml.name}, objeto {numero}"
                )

            registros.append({
                "imagen_original": nombre_imagen,
                "numero_objeto": numero,
                "clase": clase,
                "xmin_original": xmin,
                "ymin_original": ymin,
                "xmax_original": xmax,
                "ymax_original": ymax,
                "xmin_usado": xmin_usado,
                "ymin_usado": ymin_usado,
                "xmax_usado": xmax_usado,
                "ymax_usado": ymax_usado,
                "ancho_rostro": ancho_rostro,
                "alto_rostro": alto_rostro,
            })

    return pd.DataFrame(registros)


def dividir_por_imagen(rostros):
    combinaciones = (
        rostros.groupby("imagen_original")["clase"]
        .apply(lambda clases: " + ".join(sorted(set(clases))))
    )

    imagenes = combinaciones.rename(
        "clases_presentes"
    ).reset_index()

    imagenes["grupo_division"] = imagenes[
        "clases_presentes"
    ].apply(
        lambda texto: (
            "contiene_incorrecta"
            if "mask_weared_incorrect" in texto
            else texto
        )
    )

    entrenamiento, temporal = train_test_split(
        imagenes,
        test_size=0.30,
        random_state=42,
        stratify=imagenes["grupo_division"],
    )

    validacion, prueba = train_test_split(
        temporal,
        test_size=0.50,
        random_state=42,
        stratify=temporal["grupo_division"],
    )

    return pd.concat([
        entrenamiento.assign(conjunto="entrenamiento"),
        validacion.assign(conjunto="validacion"),
        prueba.assign(conjunto="prueba"),
    ])


def guardar_recortes(rostros):
    registros = []

    for _, fila in rostros.iterrows():
        nombre = (
            f"{Path(fila['imagen_original']).stem}"
            f"_obj{int(fila['numero_objeto']):03d}.png"
        )

        carpeta = SALIDA / fila["conjunto"] / fila["clase"]
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_destino = carpeta / nombre

        caja = (
            int(fila["xmin_usado"]),
            int(fila["ymin_usado"]),
            int(fila["xmax_usado"]),
            int(fila["ymax_usado"]),
        )

        with Image.open(
            IMAGENES / fila["imagen_original"]
        ) as imagen:
            recorte = imagen.crop(caja).convert("RGBA")
            fondo = Image.new("RGBA", recorte.size, "white")

            recorte_final = Image.alpha_composite(
                fondo, recorte
            ).convert("RGB").resize(
                (128, 128),
                Image.Resampling.BILINEAR,
            )

            recorte_final.save(ruta_destino)

        registro = fila.to_dict()
        registro["ruta_recorte"] = str(ruta_destino)
        registros.append(registro)

    return pd.DataFrame(registros)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    anotaciones = leer_anotaciones()

    rostros = anotaciones[
        (anotaciones["ancho_rostro"] >= 10)
        & (anotaciones["alto_rostro"] >= 10)
    ].copy()

    asignaciones = dividir_por_imagen(rostros)

    rostros = rostros.merge(
        asignaciones[["imagen_original", "conjunto"]],
        on="imagen_original",
        how="left",
        validate="many_to_one",
    )

    if rostros["conjunto"].isna().any():
        raise ValueError("Hay rostros sin conjunto asignado")

    asignaciones.to_csv(
        SALIDA / "splits.csv", index=False
    )

    metadata = guardar_recortes(rostros)
    metadata.to_csv(
        SALIDA / "metadata.csv", index=False
    )

    print("Anotaciones originales:", len(anotaciones))
    print("Rostros excluidos por tamaño:", len(anotaciones) - len(rostros))
    print("Recortes generados:", len(metadata))
    print("Imágenes asignadas:", len(asignaciones))
    print(
        "Cajas conservadas con xmax corregido:",
        (metadata["xmax_original"] != metadata["xmax_usado"]).sum(),
    )
    print("\nRecortes por clase y conjunto:")
    print(pd.crosstab(metadata["clase"], metadata["conjunto"]))


if __name__ == "__main__":
    main()