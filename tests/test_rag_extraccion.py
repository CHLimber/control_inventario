"""Comprensión de la pregunta: fechas, productos e intención (sin BD ni LLM)."""
from dataclasses import dataclass
from datetime import date

import pytest

from src.config import cargar_clases
from src.rag.extraccion import (clasificar, extraer_fechas, extraer_id_cierre,
                                extraer_productos, normalizar)

HOY = date(2026, 10, 15)   # jueves


@dataclass
class P:
    id: int
    nombre: str
    marca: str
    formato: str
    clase_modelo: str


PRODUCTOS = [P(c["id"] + 1, c["nombre"], c["marca"], c["formato"], c["clase"])
             for c in cargar_clases()["clases"]]


def test_normalizar():
    assert normalizar("¿Cuánta COCA-COLA había el 12-10?") == "cuanta coca cola habia el 12/10"


@pytest.mark.parametrize("texto, desde, hasta", [
    ("¿qué faltó hoy?", "2026-10-15", "2026-10-15"),
    ("y ayer?", "2026-10-14", "2026-10-14"),
    ("anteayer", "2026-10-13", "2026-10-13"),
    ("faltantes de esta semana", "2026-10-12", "2026-10-15"),
    ("¿y la semana pasada?", "2026-10-05", "2026-10-11"),
    ("en lo que va de este mes", "2026-10-01", "2026-10-15"),
    ("el mes pasado", "2026-09-01", "2026-09-30"),
    ("el 12 de octubre", "2026-10-12", "2026-10-12"),
    ("el 12 de Octubre de 2025", "2025-10-12", "2025-10-12"),
    ("el 12/10", "2026-10-12", "2026-10-12"),
    ("el 3/9/2026", "2026-09-03", "2026-09-03"),
    ("del 1 al 10 de octubre", "2026-10-01", "2026-10-10"),
    ("entre el 28 de septiembre y el 3 de octubre", "2026-09-28", "2026-10-03"),
    ("del 1/10 al 8/10", "2026-10-01", "2026-10-08"),
    ("los últimos 3 días", "2026-10-13", "2026-10-15"),
    ("los últimos diez días", "2026-10-06", "2026-10-15"),
    ("¿qué faltó en las últimas dos semanas?", "2026-10-02", "2026-10-15"),
    ("las últimas 3 semanas", "2026-09-25", "2026-10-15"),
    ("el lunes", "2026-10-12", "2026-10-12"),
    ("el 20 de diciembre", "2025-12-20", "2025-12-20"),   # futuro -> año pasado
])
def test_extraer_fechas(texto, desde, hasta):
    r = extraer_fechas(texto, HOY)
    assert r is not None, texto
    assert (r.desde.isoformat(), r.hasta.isoformat()) == (desde, hasta)


def test_sin_fecha():
    assert extraer_fechas("¿cuánta fanta lata hay?", HOY) is None
    assert extraer_fechas("¿qué pasó en el cierre 3?", HOY) is None


@pytest.mark.parametrize("texto, esperados", [
    ("¿cuánta Fanta lata hay?", {"fanta_lata"}),
    ("stock de coca", {"coca_botella", "coca_lata"}),
    ("coca cola en botella", {"coca_botella"}),
    ("COCA-COLA lata", {"coca_lata"}),
    ("latas de coca", {"coca_lata"}),
    ("¿cuántas latas quedan?", {"coca_lata", "fanta_lata", "sprite_lata", "pepsi_lata"}),
    ("coca lata y fanta botella", {"coca_lata", "fanta_botella"}),
    ("sprite y pepsi", {"sprite_botella", "sprite_lata", "pepsi_botella", "pepsi_lata"}),
    ("faltó pepsi_botella", {"pepsi_botella"}),          # clase del modelo
    ("¿qué faltó hoy?", set()),
])
def test_extraer_productos(texto, esperados):
    por_id = {p.id: p.clase_modelo for p in PRODUCTOS}
    r = extraer_productos(texto, PRODUCTOS)
    assert {por_id[i] for i in r.ids} == esperados


@pytest.mark.parametrize("texto, esperado", [
    ("cierre 3", 3), ("¿qué pasó en el cierre #3?", 3), ("cierre nº 12", 12),
    ("el cierre del 12/10", None), ("cierre 12 de octubre", None), ("¿qué faltó hoy?", None),
])
def test_extraer_id_cierre(texto, esperado):
    assert extraer_id_cierre(texto) == esperado


@pytest.mark.parametrize("texto, intencion", [
    ("¿Qué faltó hoy?", "faltantes_rango"),
    ("¿hubo sobrantes la semana pasada?", "faltantes_rango"),
    ("diferencias del 1 al 10 de octubre", "faltantes_rango"),
    ("¿Cuánta Fanta lata hay?", "stock_producto"),
    ("stock de coca", "stock_producto"),
    ("¿cuántas latas quedan?", "stock_producto"),
    ("¿Qué pasó en el cierre 3?", "detalle_cierre"),
    ("mostrame el último cierre", "detalle_cierre"),
    ("¿qué faltó en el último cierre?", "detalle_cierre"),
    ("¿cuánta coca lata faltó el 8 de octubre?", "diferencia_puntual"),
    ("diferencia de pepsi botella en el cierre 2", "diferencia_puntual"),
    ("hola, ¿cómo estás?", "desconocida"),
])
def test_clasificar(texto, intencion):
    hay = bool(extraer_productos(texto, PRODUCTOS))
    assert clasificar(texto, hay, extraer_id_cierre(texto)) == intencion
