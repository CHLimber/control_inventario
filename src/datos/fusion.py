"""Construcción de los datasets YOLO fusionados (detector de 1 clase y multiclase).

Estructura de salida (la que espera Ultralytics):
  <dir>/images/{train,val,test}/<alias>__<archivo>    (enlace duro al crudo, o copia)
  <dir>/labels/{train,val,test}/<alias>__<archivo>.txt
  <dir>/data.yaml
  <dir>/_manifiesto.csv   (una fila por imagen: origen, grupo, split, cajas)
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.datos.remapeo import DESCARTAR, SOLO_DETECTOR, destino, es_generica
from src.datos.split import seleccionar
from src.datos.util import SPLITS, escribir_etiquetas_yolo, leer_etiquetas_yolo, ruta_etiqueta

MODOS = ("detector_1clase", "multiclase")
MIN_LADO_NORM = 1e-3  # cajas degeneradas (menos de ~0,1 % de la imagen) se ignoran


def cajas_remapeadas(ruta_txt: Path, nombres: list[str], remapeo: dict, alias: str,
                     modo: str, clases: list[str]) -> list[tuple[int, float, float, float, float]]:
    """Cajas de una imagen con el id de clase de salida según el modo."""
    salida = []
    for cid, xc, yc, w, h in leer_etiquetas_yolo(ruta_txt):
        if w < MIN_LADO_NORM or h < MIN_LADO_NORM:
            continue
        nombre = nombres[cid] if 0 <= cid < len(nombres) else str(cid)
        dest = destino(remapeo, alias, nombre)
        if dest == DESCARTAR:
            continue
        if modo == "detector_1clase":
            salida.append((0, xc, yc, w, h))
        elif dest != SOLO_DETECTOR and dest in clases:
            salida.append((clases.index(dest), xc, yc, w, h))
    return salida


def tiene_cajas_genericas(ruta_txt: Path, nombres: list[str], remapeo: dict, alias: str) -> bool:
    """¿Hay cajas de una clase genérica sin marca (que podría ser una de las nuestras)?"""
    for cid, *_ in leer_etiquetas_yolo(ruta_txt):
        nombre = nombres[cid] if 0 <= cid < len(nombres) else str(cid)
        if destino(remapeo, alias, nombre) == SOLO_DETECTOR and es_generica(nombre):
            return True
    return False


def _enlazar(origen: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(origen, dest)
    except OSError:
        shutil.copy2(origen, dest)


def _elegir_negativos(df_neg: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or df_neg.empty:
        return df_neg.iloc[0:0]
    orden = df_neg["ruta"].map(lambda r: hashlib.sha1(r.encode("utf-8")).hexdigest())
    return df_neg.assign(_o=orden).sort_values("_o").head(n).drop(columns="_o")


def construir(df: pd.DataFrame, dir_raw: Path, dir_salida: Path, modo: str, remapeo: dict,
              nombres_por_alias: dict[str, list[str]], clases: list[str], alias_permitidos: set[str],
              copias_train: int = 1, max_negativos: float = 0.1,
              prioridad_alias: list[str] | None = None) -> pd.DataFrame:
    """Arma un dataset fusionado. `df` es el índice con columnas grupo y split.

    - detector_1clase: toda caja que no se descarta pasa a 'producto' (id 0).
    - multiclase: solo cajas de las 9 clases; otras bebidas con marca (cerveza, Red Bull...)
      quedan como fondo. Se excluyen imágenes con cajas de clases genéricas sin marca
      (p. ej. 'Cans'), porque podrían ser una de nuestras marcas sin etiquetar.
    - Imágenes sin ninguna caja útil (negativos): como máximo `max_negativos` × positivos por split.
    """
    assert modo in MODOS
    if dir_salida.exists():
        shutil.rmtree(dir_salida)
    filas = []
    for r in df[df["alias"].isin(alias_permitidos)].itertuples():
        txt = ruta_etiqueta(dir_raw / r.ruta)
        nombres = nombres_por_alias.get(r.alias, [])
        cajas = cajas_remapeadas(txt, nombres, remapeo, r.alias, modo, clases)
        if modo == "multiclase" and tiene_cajas_genericas(txt, nombres, remapeo, r.alias):
            continue
        filas.append({**r._asdict(), "cajas": cajas, "n_cajas_salida": len(cajas)})
    if not filas:
        return pd.DataFrame()
    d = pd.DataFrame(filas).drop(columns=["Index"], errors="ignore")
    elegidas = []
    for split in SPLITS:
        ds = d[d["split"] == split]
        pos = seleccionar(ds[ds["n_cajas_salida"] > 0], copias_train, prioridad_alias)
        # Un negativo no puede ser de un grupo que ya tiene un positivo en este split.
        neg = ds[(ds["n_cajas_salida"] == 0) & ~ds["grupo"].isin(pos["grupo"])]
        neg = seleccionar(neg, copias_train, prioridad_alias)
        neg = _elegir_negativos(neg, int(len(pos) * max_negativos))
        elegidas.append(pos)
        elegidas.append(neg)
    sel = pd.concat(elegidas)

    destinos = []
    for r in sel.itertuples():
        origen = dir_raw / r.ruta
        nombre = f"{r.alias}__{Path(r.ruta).name}"
        img_dest = dir_salida / "images" / r.split / nombre
        _enlazar(origen, img_dest)
        escribir_etiquetas_yolo((dir_salida / "labels" / r.split / nombre).with_suffix(".txt"), r.cajas)
        destinos.append(img_dest.relative_to(dir_salida).as_posix())
    sel = sel.assign(destino=destinos)

    nombres_salida = ["producto"] if modo == "detector_1clase" else list(clases)
    data = {
        "path": dir_salida.resolve().as_posix(),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": len(nombres_salida),
        "names": {i: n for i, n in enumerate(nombres_salida)},
    }
    (dir_salida / "data.yaml").write_text(
        "# Generado por scripts/datos_fusionar.py. 'path' es absoluto: regenerar en otra máquina.\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    manifiesto = sel.drop(columns=["cajas"]).assign(
        clases_salida=[";".join(str(c[0]) for c in cs) for cs in sel["cajas"]])
    manifiesto.to_csv(dir_salida / "_manifiesto.csv", index=False)
    return manifiesto


def conteos(manifiesto: pd.DataFrame, nombres_salida: list[str]) -> pd.DataFrame:
    """Tabla clase x split con cajas e imágenes (con al menos una caja de esa clase)."""
    filas = []
    for split in SPLITS:
        m = manifiesto[manifiesto["split"] == split]
        for i, nombre in enumerate(nombres_salida):
            ids = m["clases_salida"].astype(str).map(lambda s: [int(x) for x in s.split(";") if x != ""])
            cajas = int(sum(c.count(i) for c in ids))
            imagenes = int(sum(1 for c in ids if i in c))
            filas.append({"split": split, "clase": nombre, "imagenes": imagenes, "cajas": cajas})
    return pd.DataFrame(filas)
