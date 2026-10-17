"""Remapeo de clases originales de cada dataset a nuestras clases (config/remapeo_clases.yaml).

Destinos posibles para cada clase original:
  - una de las 8 clases de config/clases.json  -> entra al detector, al multiclase y a los recortes
  - "solo_detector" -> es una bebida/envase útil como "producto", pero sin marca confiable
  - "descartar"     -> no es un producto que nos interese (snacks, vasos, fondo...): la caja se elimina
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

SOLO_DETECTOR = "solo_detector"
DESCARTAR = "descartar"
# "auto:<marca>": el dataset trae la marca pero no el formato (p. ej. "Pepsi" para botellas y latas).
# El formato lo decide la forma de la caja (alto / ancho en píxeles):
#   lata    1,4 a 2,15   (una lata de 355 ml mide unas 1,85 veces su ancho)
#   botella 2,7 a 5,0    (una botella PET frontal ronda 3 a 3,5)
# Entre 2,15 y 2,7 (botella tapada, caja dudosa) la caja queda como solo_detector.
# "auto:<marca>:botella" (o ":lata") acepta UN solo formato: lo que no encaje queda como solo_detector.
# Hace falta cuando el dataset casi no tiene latas: una botella tapada o cortada mide como una lata
# (verificado a ojo en cold_drinks_2: todo lo que salía como "lata" eran botellas).
AUTO = "auto:"
RATIO_LATA = (1.4, 2.15)
RATIO_BOTELLA = (2.7, 5.0)

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
_LATA = ("can", "cans", "lata", "tin", "tincan", "lon")          # "lon" = lata en vietnamita
_BOTELLA = ("bottle", "bottel", "botella", "chai", "pet", "20oz", "40oz", "1l", "2l", "500ml", "600ml", "15l")


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


def es_auto(dest: str) -> bool:
    return dest.startswith(AUTO)


def resolver_auto(dest: str, ancho_px: float, alto_px: float, clases_validas: list[str]) -> str:
    """Clase de una caja con destino ``auto:<marca>[:formato]``: la marca del destino y el formato por la
    forma de la caja. Si la forma es dudosa, no es el formato pedido o la clase no existe, devuelve SOLO_DETECTOR."""
    marca, _, solo = dest[len(AUTO):].partition(":")
    ratio = alto_px / max(ancho_px, 1e-6)
    if RATIO_LATA[0] <= ratio <= RATIO_LATA[1]:
        formato = "lata"
    elif RATIO_BOTELLA[0] <= ratio <= RATIO_BOTELLA[1]:
        formato = "botella"
    else:
        return SOLO_DETECTOR
    if solo and formato != solo:
        return SOLO_DETECTOR
    clase = f"{marca}_{formato}"
    return clase if clase in clases_validas else SOLO_DETECTOR


def _auto_valido(dest: str, marcas: set[str]) -> bool:
    marca, _, solo = dest[len(AUTO):].partition(":")
    return marca in marcas and solo in ("", "botella", "lata")


def validar_remapeo(remapeo: dict[str, dict[str, str]], clases_validas: list[str]) -> list[str]:
    """Devuelve la lista de errores (destinos que no existen)."""
    permitidos = set(clases_validas) | {SOLO_DETECTOR, DESCARTAR}
    marcas = {c.split("_")[0] for c in clases_validas}
    return [f"{a}: {k!r} -> {v!r} no es un destino válido"
            for a, m in remapeo.items() for k, v in m.items()
            if v not in permitidos and not (es_auto(v) and _auto_valido(v, marcas))]


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
