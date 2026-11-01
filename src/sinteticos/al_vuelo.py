"""Un estante sintético por pedido (para el laboratorio web, la demo y mediciones rápidas).

Usa el mismo ``componer`` que los lotes de entrenamiento, pero de a una imagen y con opciones
simples (niveles y orden). La biblioteca de envases y los fondos se cargan una sola vez.

    from src.sinteticos.al_vuelo import generar_estante
    esc = generar_estante(semilla=3, niveles=4, orden="ordenado")
    esc.imagen  # RGB uint8;  esc.cajas: [(cid, x1, y1, x2, y2)];  conteo_escena(esc) -> {clase: n}

Las semillas viven en un espacio propio (``[ESPACIO_SEMILLAS, semilla]``), distinto del de los lotes
(``[semilla_lote, i]``): la escena es nueva aunque la semilla coincida con la de un lote. Ojo: los
ENVASES sí salen de la biblioteca, recortada de fotos de train, así que los modelos ya vieron esas
etiquetas. Sirve para demostrar y comparar modelos entre sí, no como evaluación del dominio real.
"""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import DIR_DATOS, DIR_MAQUETA, cargar_clases
from src.sinteticos import fondos as fnd
from src.sinteticos import objetos as obj
from src.sinteticos.composicion import Escena, Parametros, componer

ESPACIO_SEMILLAS = 7777
ORDENES = ("ordenado", "normal", "desordenado")
DIR_OBJETOS = DIR_DATOS / "sinteticos" / "_objetos"
DIR_FUSION = DIR_DATOS / "fusion"

_recursos: dict = {}
_candado = threading.Lock()


class SinBiblioteca(RuntimeError):
    """No está la biblioteca de envases recortados (scripts/sinteticos_objetos.py)."""


def _cargar_recursos() -> tuple[dict[str, list[str]], list[str]]:
    with _candado:
        if not _recursos:
            if not DIR_OBJETOS.is_dir():
                raise SinBiblioteca(f"No existe {DIR_OBJETOS}: correr antes scripts/sinteticos_objetos.py")
            dir_train = DIR_FUSION / "multiclase" / "images" / "train"
            extra = DIR_FUSION / "_biblioteca" / "images" / "train"
            bib = obj.cargar_biblioteca(DIR_OBJETOS, [dir_train, extra])
            if not bib:
                raise SinBiblioteca(f"La biblioteca de {DIR_OBJETOS} está vacía")
            _recursos["bib"] = {k: [str(p) for p in v] for k, v in bib.items()}
            _recursos["fondos"] = [str(p) for p in fnd.listar_fondos_reales(dir_train, DIR_MAQUETA / "fondos")]
        return _recursos["bib"], _recursos["fondos"]


def disponible() -> bool:
    try:
        _cargar_recursos()
        return True
    except SinBiblioteca:
        return False


def parametros(niveles: int = 0, orden: str | None = None, acercamiento: bool = False,
               lleno: bool = False, base: Parametros | None = None) -> Parametros:
    """Parametros de una foto del estante entero (sin estantes vacíos) con niveles y orden fijos.

    niveles: 3, 4 o 5 (0 = al azar).  orden: 'ordenado' | 'normal' | 'desordenado' (None = al azar).
    lleno: ningún nivel vacío (para la demo; por defecto cada nivel puede quedar vacío como en los lotes).
    """
    p = replace(base or Parametros(), n_niveles=niveles if niveles in (3, 4, 5) else 0,
                p_estante_vacio=0.0, p_acercamiento=0.3 if acercamiento else 0.0)
    if lleno:
        p = replace(p, p_nivel_vacio=0.0)
    if orden is not None:
        if orden not in ORDENES:
            raise ValueError(f"orden debe ser uno de {ORDENES}, no {orden!r}")
        p = replace(p, p_ordenado=1.0 if orden == "ordenado" else 0.0,
                    p_desordenado=1.0 if orden == "desordenado" else 0.0)
    return p


def generar_estante(semilla: int, niveles: int = 0, orden: str | None = None,
                    acercamiento: bool = False, lleno: bool = False) -> Escena:
    bib, fondos = _cargar_recursos()
    rng = np.random.default_rng([ESPACIO_SEMILLAS, int(semilla)])
    return componer(rng, bib, cargar_clases()["clases"], fondos, parametros(niveles, orden, acercamiento, lleno))


def conteo_escena(escena: Escena) -> dict[str, int]:
    """Conteo exacto por clase (las 8 de config/clases.json, en ese orden)."""
    clases = [c["clase"] for c in cargar_clases()["clases"]]
    conteo = dict.fromkeys(clases, 0)
    for cid, *_ in escena.cajas:
        conteo[clases[int(cid)]] += 1
    return conteo


def guardar(escena: Escena, ruta: Path, calidad: int = 92) -> Path:
    from src.sinteticos.composicion import guardar_jpg

    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar_jpg(ruta, escena.imagen, calidad)
    return ruta
