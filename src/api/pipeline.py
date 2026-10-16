"""Punto de enganche del pipeline de visión (variantes mock / A / B).

La API solo necesita una función `contar(imagenes: list[Path]) -> ResultadoConteo`.
`resolver_pipeline` acepta:
    - None: elige según src.config.VARIANTE_VISION;
    - un módulo u objeto con `.contar`;
    - una función (p. ej. functools.partial(pipeline_mock.contar, semilla=7) en los tests).

Para enchufar las variantes reales alcanza con crear el módulo de la tabla
MODULOS_VARIANTE (o un src.vision.pipeline con contar(imagenes, variante=...)).
"""
from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from contratos.vision_types import ResultadoConteo

FuncionContar = Callable[[list[Path]], ResultadoConteo]

log = logging.getLogger(__name__)

MODULOS_VARIANTE: dict[str, list[str]] = {
    "mock": ["src.vision.pipeline_mock"],
    "A": ["src.vision.pipeline_a", "src.vision.pipeline"],
    "B": ["src.vision.pipeline_b", "src.vision.pipeline"],
}


def _acepta_variante(funcion: Callable) -> bool:
    try:
        return "variante" in inspect.signature(funcion).parameters
    except (TypeError, ValueError):
        return False


def cargar_variante(variante: str) -> FuncionContar:
    """Importa el pipeline de la variante; si todavía no existe, avisa y usa el mock.

    El cierre guarda `resultado.variante`, así que un fallback al mock queda registrado.
    """
    for nombre in MODULOS_VARIANTE.get(variante, []):
        try:
            modulo = importlib.import_module(nombre)
        except ImportError as error:
            log.debug("No se pudo importar %s: %s", nombre, error)
            continue
        contar = getattr(modulo, "contar", None)
        if contar is None:
            continue
        if variante != "mock" and _acepta_variante(contar):
            return partial(contar, variante=variante)
        return contar
    log.warning("Pipeline de la variante %r no disponible; se usa el mock.", variante)
    from src.vision import pipeline_mock
    return pipeline_mock.contar


def resolver_pipeline(pipeline: Any = None, variante: str | None = None) -> FuncionContar:
    if pipeline is None:
        if variante is None:
            from src.config import VARIANTE_VISION
            variante = VARIANTE_VISION
        return cargar_variante(variante)
    contar = getattr(pipeline, "contar", None)
    if callable(contar):
        return contar
    if callable(pipeline):
        return pipeline
    raise TypeError("pipeline debe ser una función contar() o un objeto/módulo con .contar")
