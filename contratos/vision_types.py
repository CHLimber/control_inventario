"""CONTRATO de visión. Lo usan el pipeline (real y mock), la API y la evaluación.

Cambiar este archivo solo con aprobación del orquestador.

Interfaces públicas:
    pipeline:     contar(imagenes: list[Path]) -> ResultadoConteo
    detector:     detectar(imagen) -> list[Caja]          (variante A: clase="producto")
    clasificador: clasificar(recortes) -> list[tuple[str, float]]
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Caja:
    """Una detección en una imagen.

    Coordenadas en PÍXELES de la imagen original (no normalizadas):
    (x, y) es la esquina superior izquierda; w y h, el ancho y el alto.
    """
    x: float
    y: float
    w: float
    h: float
    conf_det: float                 # confianza del detector (0-1)
    clase: str                      # clase de config/clases.json, "desconocido" o "producto"
    conf_cls: float | None = None   # confianza del clasificador; None en la variante B
    indice_imagen: int = 0          # a cuál de las imágenes de entrada pertenece

    def a_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoConteo:
    """Salida de contar().

    conteo: SIEMPRE trae las 9 clases + "desconocido" (0 si no aparecen).
            Con varias imágenes es la mediana por clase (ver combinar_conteos).
    cajas: todas las cajas de todas las imágenes (con su indice_imagen).
    imagen_anotada_path: imagen con las cajas dibujadas. Se guarda junto a la
            primera imagen de entrada como "<nombre>_anotada.jpg".
    """
    conteo: dict[str, int]
    cajas: list[Caja]
    imagen_anotada_path: Path | None
    variante: str = "mock"                        # mock | A | B
    conteos_por_imagen: list[dict[str, int]] = field(default_factory=list)
    tiempo_ms: float = 0.0


class Pipeline(Protocol):
    def contar(self, imagenes: list[Path]) -> ResultadoConteo: ...


class Detector(Protocol):
    def detectar(self, imagen: np.ndarray | Path) -> list[Caja]: ...


class Clasificador(Protocol):
    def clasificar(self, recortes: Sequence[np.ndarray]) -> list[tuple[str, float]]: ...


def conteo_vacio(clases: Sequence[str]) -> dict[str, int]:
    return {c: 0 for c in clases}


def contar_cajas(cajas: Sequence[Caja], clases: Sequence[str]) -> dict[str, int]:
    """Cuenta cajas por clase. Una clase fuera de la lista cuenta como 'desconocido'."""
    conteo = conteo_vacio(clases)
    for caja in cajas:
        clave = caja.clase if caja.clase in conteo else "desconocido"
        conteo[clave] = conteo.get(clave, 0) + 1
    return conteo


def combinar_conteos(conteos: Sequence[dict[str, int]]) -> dict[str, int]:
    """Mediana por clase de varios conteos (una por foto).

    Con un número par de fotos la mediana puede dar x.5: se redondea hacia arriba
    (2 y 3 → 3), para no inventar faltantes por una foto mala.
    """
    if not conteos:
        raise ValueError("Se necesita al menos un conteo")
    claves = list(dict.fromkeys(k for c in conteos for k in c))
    return {k: int(math.floor(statistics.median(c.get(k, 0) for c in conteos) + 0.5))
            for k in claves}
