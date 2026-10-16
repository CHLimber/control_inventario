"""Rutas del pipeline de datos (todas derivadas de src.config.DIR_DATOS)."""
from __future__ import annotations

from src.config import DIR_CONFIG, DIR_DATOS, RAIZ
from src.datos.fuentes import FUENTES
from src.datos.util import nombres_clases_yolo

DIR_RAW = DIR_DATOS / "raw"
DIR_FUSION = DIR_DATOS / "fusion"
DIR_RECORTES = DIR_DATOS / "recortes"
RUTA_INDICE = DIR_FUSION / "_indice_imagenes.csv"      # índice + grupo + split de TODAS las imágenes crudas
RUTA_REMAPEO = DIR_CONFIG / "remapeo_clases.yaml"
DIR_DOCS = RAIZ / "docs"
DIR_DOCS_DATOS = DIR_DOCS / "datos"


def fuentes_descargadas() -> list[dict]:
    return [f for f in FUENTES if (DIR_RAW / f["alias"] / "data.yaml").exists()]


def alias_para(salida: str) -> set[str]:
    """Datasets descargados que entran a una salida: 'detector', 'multiclase' o 'recortes'."""
    return {f["alias"] for f in fuentes_descargadas() if salida in f.get("usar_en", ())}


def nombres_por_alias() -> dict[str, list[str]]:
    return {f["alias"]: nombres_clases_yolo(DIR_RAW / f["alias"]) for f in fuentes_descargadas()}


def usos() -> dict[str, str]:
    return {f["alias"]: f["uso"] for f in FUENTES}


def prioridad_alias() -> list[str]:
    """Orden de preferencia cuando un grupo tiene imágenes de varios datasets (marcas primero)."""
    return [f["alias"] for f in FUENTES if f["tipo"] == "marcas"] + [f["alias"] for f in FUENTES if f["tipo"] != "marcas"]
