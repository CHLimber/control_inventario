"""Remapeo de clases originales de cada dataset a nuestras clases (config/remapeo_clases.yaml).

Destinos posibles para cada clase original:
  - una de las 9 clases de config/clases.json  -> entra al detector, al multiclase y a los recortes
  - "solo_detector" -> es una bebida/envase útil como "producto", pero sin marca confiable
  - "descartar"     -> no es un producto que nos interese (snacks, vasos, fondo...): la caja se elimina
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

SOLO_DETECTOR = "solo_detector"
DESCARTAR = "descartar"

# Marca -> palabras clave (sobre el nombre normalizado: minúsculas, sin separadores).
_MARCAS = {
    "coca": ("cocacola", "coca", "coke"),
    "fanta": ("fanta",),
    "sprite": ("sprite",),
    "pepsi": ("pepsi",),
    "7up": ("7up", "sevenup"),
}
# Variantes de sabor/dieta con un empaque muy distinto al clásico: no se confía en la marca.
_VARIANTES = ("diet", "zero", "light", "cherry", "vanilla", "spiced", "grape", "berry", "tropical",
              "lemon", "lime", "orange", "peach", "mix")
# Cosas que no son bebidas (o no son productos): se descartan.
_NO_PRODUCTO = ("background", "nonbottle", "cup", "person", "hand", "cheetos", "doritos", "lays", "oreo",
                "chipsahoy", "skittles", "nerds", "crunch", "bueno", "payday", "sourpunch",
                "whatchamacallit", "funyuns", "lennylarry", "snack", "chips")
_LATA = ("can", "lata", "tin", "tincan")
_BOTELLA = ("bottle", "bottel", "botella", "pet", "20oz", "40oz", "1l", "2l", "500ml", "600ml", "15l")


_GENERICAS = {"bottle", "bottles", "can", "cans", "bottledetection", "botella", "lata", "producto",
              "product", "object", "objeto", "drink", "beverage", "soda", "sodacan", "sodabottle"}


def es_generica(nombre: str) -> bool:
    """Clase sin marca (p. ej. 'Cans', 'Bottle'): puede esconder una de nuestras marcas sin etiquetar."""
    return normalizar(nombre) in _GENERICAS


def normalizar(nombre: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", nombre.lower())


def _palabras(nombre: str) -> set[str]:
    con_espacios = re.sub(r"([a-z])([A-Z])", r"\1 \2", nombre)
    con_espacios = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", con_espacios)
    return {p for p in re.split(r"[^a-z0-9]+", con_espacios.lower()) if p}


def detectar_formato(nombre: str, formato_defecto: str | None = None) -> str | None:
    norm, pals = normalizar(nombre), _palabras(nombre)
    if any(p in pals for p in _LATA) or re.search(r"can\d|can$", norm):
        return "lata"
    if any(p in pals for p in _BOTELLA) or any(b in norm for b in ("bottle", "bottel", "botella", "20oz", "40oz")):
        return "botella"
    return formato_defecto


def sugerir(nombre: str, clases_validas: list[str], formato_defecto: str | None = None) -> str:
    """Propuesta automática de destino para una clase original (luego se revisa a mano)."""
    norm = normalizar(nombre)
    if not norm or any(k in norm for k in _NO_PRODUCTO):
        return DESCARTAR
    marcas = {m for m, claves in _MARCAS.items() if any(k in norm for k in claves)}
    if len(marcas) != 1:
        return SOLO_DETECTOR  # sin marca o ambigua (p. ej. "fantacoca-cola")
    if any(v in norm for v in _VARIANTES):
        return SOLO_DETECTOR
    formato = detectar_formato(nombre, formato_defecto)
    clase = f"{marcas.pop()}_{formato}" if formato else None
    return clase if clase in clases_validas else SOLO_DETECTOR


def cargar_remapeo(ruta: Path) -> dict[str, dict[str, str]]:
    if not ruta.exists():
        return {}
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    return {str(a): {str(k): str(v) for k, v in (m or {}).items()} for a, m in datos.items()}


def validar_remapeo(remapeo: dict[str, dict[str, str]], clases_validas: list[str]) -> list[str]:
    """Devuelve la lista de errores (destinos que no existen)."""
    permitidos = set(clases_validas) | {SOLO_DETECTOR, DESCARTAR}
    return [f"{a}: {k!r} -> {v!r} no es un destino válido"
            for a, m in remapeo.items() for k, v in m.items() if v not in permitidos]


def destino(remapeo: dict[str, dict[str, str]], alias: str, nombre_original: str) -> str:
    """Destino de una clase original. Lo que no figura en el YAML se descarta (conservador)."""
    return remapeo.get(alias, {}).get(nombre_original, DESCARTAR)


def escribir_remapeo(ruta: Path, remapeo: dict[str, dict[str, str]], encabezado: str) -> None:
    lineas = [encabezado.rstrip(), ""]
    for alias in remapeo:
        lineas.append(f"{alias}:")
        for nombre, dest in remapeo[alias].items():
            lineas.append(f"  {yaml.safe_dump(nombre, allow_unicode=True).strip().removesuffix('...').strip()}: {dest}")
        lineas.append("")
    ruta.write_text("\n".join(lineas), encoding="utf-8")
