"""Índice de imágenes de los datasets crudos: id original, pHash, tamaño y cajas.

El cálculo del pHash es lo caro: se cachea en un CSV y solo se recalcula lo nuevo
(la clave es ruta + tamaño en bytes).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.datos.util import (
    es_copia_roboflow, id_original, leer_etiquetas_yolo, pares_imagen_etiqueta, split_origen,
)

COLUMNAS = ["alias", "proyecto", "ruta", "split_origen", "id_original", "copia_aumentada",
            "bytes", "ancho", "alto", "phash", "n_cajas", "clases_ids"]


def _medir(ruta: str) -> tuple[int, int, str]:
    """(ancho, alto, phash hex). Se ejecuta en otro proceso."""
    import imagehash
    from PIL import Image

    try:
        with Image.open(ruta) as im:
            ancho, alto = im.size
            return ancho, alto, str(imagehash.phash(im.convert("L")))
    except Exception:  # noqa: BLE001 — imagen corrupta
        return 0, 0, ""


def indexar_dataset(dir_raw: Path, alias: str, proyecto: str, cache: pd.DataFrame | None = None,
                    procesos: int | None = None) -> pd.DataFrame:
    dir_ds = dir_raw / alias
    filas = []
    for img, txt in pares_imagen_etiqueta(dir_ds):
        cajas = leer_etiquetas_yolo(txt)
        filas.append({
            "alias": alias, "proyecto": proyecto,
            "ruta": img.relative_to(dir_raw).as_posix(),
            "split_origen": split_origen(img, dir_ds),
            "id_original": id_original(img.name),
            "copia_aumentada": es_copia_roboflow(img.name),
            "bytes": img.stat().st_size,
            "n_cajas": len(cajas),
            "clases_ids": ";".join(str(c[0]) for c in cajas),
        })
    df = pd.DataFrame(filas, columns=[c for c in COLUMNAS if c not in ("ancho", "alto", "phash")])
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS)

    previas = {}
    if cache is not None and not cache.empty:
        for r in cache[cache["alias"] == alias].itertuples():
            previas[(r.ruta, int(r.bytes))] = (int(r.ancho), int(r.alto), str(r.phash) if isinstance(r.phash, str) else "")
    faltan = [r for r in df.itertuples() if (r.ruta, int(r.bytes)) not in previas]
    if faltan:
        rutas = [str(dir_raw / r.ruta) for r in faltan]
        procesos = procesos or max(1, (os.cpu_count() or 2) - 1)
        if procesos > 1 and len(rutas) > 200:
            with ProcessPoolExecutor(procesos) as ex:
                medidas = list(ex.map(_medir, rutas, chunksize=64))
        else:
            medidas = [_medir(r) for r in rutas]
        for r, m in zip(faltan, medidas):
            previas[(r.ruta, int(r.bytes))] = m
    medidas = [previas[(r.ruta, int(r.bytes))] for r in df.itertuples()]
    df["ancho"] = [m[0] for m in medidas]
    df["alto"] = [m[1] for m in medidas]
    df["phash"] = [m[2] for m in medidas]
    return df[COLUMNAS]


def cargar_indice(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_csv(ruta, dtype={"phash": str, "clases_ids": str, "id_original": str},
                       keep_default_na=False)
