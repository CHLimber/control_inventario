"""Árbol YOLO auxiliar para la biblioteca de envases de los sintéticos (solo train).

Los datasets que no entran al multiclase (porque traen cajas sin etiquetar o de marcas que no son las
nuestras: ``cold_drinks_2``, ``hccb_overall``) sí tienen envases de góndola útiles para recortar. Acá se
arma ``<DIR_DATOS>/fusion/_biblioteca/{images,labels}/train`` con SUS imágenes de train:
  - las cajas que el remapeo lleva a una de nuestras clases (incluido ``auto:<marca>``) con su id de clase;
  - el resto de las cajas con el id ``ID_OTRO`` (no son candidatas, pero cuentan como tapa de las vecinas).
Solo train: usar val o test filtraría evaluación al entrenamiento.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.datos.fusion import _enlazar, _tam_imagen
from src.datos.remapeo import DESCARTAR, destino, es_auto, resolver_auto
from src.datos.util import escribir_etiquetas_yolo, leer_etiquetas_yolo, ruta_etiqueta

ID_OTRO = 99


def construir_fuente_biblioteca(df: pd.DataFrame, dir_raw: Path, dir_salida: Path, remapeo: dict,
                                nombres_por_alias: dict[str, list[str]], clases: list[str],
                                aliases: set[str]) -> dict[str, int]:
    """Escribe el árbol auxiliar. Devuelve {clase: cantidad de cajas con clase propia}."""
    if dir_salida.exists():
        shutil.rmtree(dir_salida)
    conteo = {c: 0 for c in clases}
    sel = df[df["alias"].isin(aliases) & (df["split"] == "train") & (df["n_cajas"] > 0)]
    for r in sel.itertuples():
        txt = ruta_etiqueta(dir_raw / r.ruta)
        nombres = nombres_por_alias.get(r.alias, [])
        cajas, tam = [], None
        for cid, xc, yc, w, h in leer_etiquetas_yolo(txt):
            nombre = nombres[cid] if 0 <= cid < len(nombres) else str(cid)
            dest = destino(remapeo, r.alias, nombre)
            if dest == DESCARTAR:
                continue
            if es_auto(dest):
                tam = tam or _tam_imagen(txt) or (1, 1)
                dest = resolver_auto(dest, w * tam[0], h * tam[1], clases)
            if dest in clases:
                conteo[dest] += 1
                cajas.append((clases.index(dest), xc, yc, w, h))
            else:
                cajas.append((ID_OTRO, xc, yc, w, h))
        if not any(c[0] != ID_OTRO for c in cajas):
            continue
        nombre_img = f"{r.alias}__{Path(r.ruta).name}"
        _enlazar(dir_raw / r.ruta, dir_salida / "images" / "train" / nombre_img)
        escribir_etiquetas_yolo((dir_salida / "labels" / "train" / nombre_img).with_suffix(".txt"), cajas)
    return conteo
