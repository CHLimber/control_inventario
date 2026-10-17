"""Recortes por caja para el clasificador CNN: DIR_DATOS/recortes/<split>/<clase>/*.jpg.

Filtros (en este orden): caja chica -> color incoherente con la clase (scripts/datos_color.py) ->
borrosa (varianza del Laplaciano) -> duplicada (pHash).
Cada caja candidata deja una fila en el índice CSV, con el motivo de descarte si no se guardó.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.datos.fusion import cajas_remapeadas
from src.datos.util import SPLITS, a_pixeles, ruta_etiqueta

ALTO_NITIDEZ = 128  # el Laplaciano se mide sobre el recorte reescalado a este alto


def recortar(img: Image.Image, caja_norm: tuple[float, float, float, float], margen: float) -> tuple[Image.Image, int, int]:
    """Recorta con un margen proporcional al tamaño de la caja. Devuelve (recorte, ancho_caja, alto_caja)."""
    xc, yc, w, h = caja_norm
    x1, y1, x2, y2 = a_pixeles(xc, yc, w, h, img.width, img.height)
    bw, bh = x2 - x1, y2 - y1
    mx, my = round(bw * margen), round(bh * margen)
    caja = (max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my))
    return img.crop(caja), bw, bh


def nitidez(recorte: Image.Image) -> float:
    """Varianza del Laplaciano sobre el recorte en grises reescalado a ALTO_NITIDEZ de alto."""
    gris = np.asarray(recorte.convert("L"))
    if gris.size == 0:
        return 0.0
    escala = ALTO_NITIDEZ / max(1, gris.shape[0])
    ancho = max(8, round(gris.shape[1] * escala))
    gris = cv2.resize(gris, (ancho, ALTO_NITIDEZ), interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def generar(sel: pd.DataFrame, dir_raw: Path, dir_salida: Path, remapeo: dict,
            nombres_por_alias: dict[str, list[str]], clases: list[str], margen: float = 0.05,
            min_lado: int = 32, min_nitidez: float = 40.0, calidad_jpeg: int = 92,
            rechazos_color: dict[tuple[str, int], str] | None = None,
            topes: dict[str, int] | None = None, tope_por_imagen: int = 2) -> pd.DataFrame:
    """Genera los recortes de las imágenes seleccionadas (columnas: alias, ruta, grupo, split).

    ``topes`` = {alias: máximo de recortes por clase en train} para datasets enormes (una góndola trae
    más de 15 cajas por foto): en val y test el tope es el 12 %, y de una misma imagen salen a lo sumo
    ``tope_por_imagen`` recortes por clase, para que el subconjunto reparta las fotos y no se agote
    en las primeras. Los que no entran quedan en el índice con motivo ``tope_por_origen``.

    Se procesa train antes que val/test: si un recorte de val/test es idéntico (pHash) a uno ya
    guardado en otro split, se descarta para no filtrar información.
    """
    for split in SPLITS:
        shutil.rmtree(dir_salida / split, ignore_errors=True)
    vistos: dict[str, tuple[str, str]] = {}  # phash -> (split, clase)
    rechazos_color = rechazos_color or {}
    topes = topes or {}
    guardados: dict[tuple[str, str, str], int] = {}
    filas = []
    orden = {s: i for i, s in enumerate(SPLITS)}
    sel = sel.sort_values(["split", "alias", "ruta"], key=lambda c: c.map(orden) if c.name == "split" else c)
    for r in sel.itertuples():
        ruta_img = dir_raw / r.ruta
        cajas = cajas_remapeadas(ruta_etiqueta(ruta_img), nombres_por_alias.get(r.alias, []), remapeo,
                                 r.alias, "multiclase", clases)
        if not cajas:
            continue
        try:
            with Image.open(ruta_img) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:  # noqa: BLE001
            for i, c in enumerate(cajas):
                filas.append({"origen": r.alias, "imagen": r.ruta, "imagen_original": r.grupo, "split": r.split,
                              "clase": clases[c[0]], "caja": i, "motivo_descarte": "imagen_ilegible"})
            continue
        por_imagen: dict[str, int] = {}
        for i, (cid, xc, yc, w, h) in enumerate(cajas):
            clase = clases[cid]
            fila = {"origen": r.alias, "imagen": r.ruta, "imagen_original": r.grupo, "split": r.split,
                    "clase": clase, "caja": i, "ancho": 0, "alto": 0, "nitidez": np.nan, "phash": "",
                    "ruta_recorte": "", "motivo_descarte": ""}
            tope = topes.get(r.alias)
            if tope:
                limite = tope if r.split == "train" else max(1, round(tope * 0.12))
                if guardados.get((r.alias, clase, r.split), 0) >= limite or por_imagen.get(clase, 0) >= tope_por_imagen:
                    fila["motivo_descarte"] = "tope_por_origen"
                    filas.append(fila)
                    continue
            recorte, bw, bh = recortar(img, (xc, yc, w, h), margen)
            fila["ancho"], fila["alto"] = bw, bh
            if min(bw, bh) < min_lado:
                fila["motivo_descarte"] = "chica"
                filas.append(fila)
                continue
            if (r.ruta, i) in rechazos_color:
                fila["motivo_descarte"] = rechazos_color[(r.ruta, i)]
                filas.append(fila)
                continue
            fila["nitidez"] = round(nitidez(recorte), 1)
            if fila["nitidez"] < min_nitidez:
                fila["motivo_descarte"] = "borrosa"
                filas.append(fila)
                continue
            ph = str(imagehash.phash(recorte))
            fila["phash"] = ph
            if ph in vistos:
                split_previo, clase_previa = vistos[ph]
                if clase_previa != clase:
                    fila["motivo_descarte"] = "duplicada_otra_clase"
                elif split_previo != r.split:
                    fila["motivo_descarte"] = "duplicada_entre_splits"
                else:
                    fila["motivo_descarte"] = "duplicada"
                filas.append(fila)
                continue
            vistos[ph] = (r.split, clase)
            if tope:
                guardados[(r.alias, clase, r.split)] = guardados.get((r.alias, clase, r.split), 0) + 1
                por_imagen[clase] = por_imagen.get(clase, 0) + 1
            destino = dir_salida / r.split / clase / f"{r.alias}__{Path(r.ruta).stem}__{i}.jpg"
            destino.parent.mkdir(parents=True, exist_ok=True)
            recorte.save(destino, "JPEG", quality=calidad_jpeg)
            fila["ruta_recorte"] = destino.relative_to(dir_salida).as_posix()
            filas.append(fila)
    indice = pd.DataFrame(filas)
    if not indice.empty:
        indice.to_csv(dir_salida / "indice_recortes.csv", index=False)
    return indice
