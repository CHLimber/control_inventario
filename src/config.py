"""Configuración compartida: rutas del proyecto, clases y variables de entorno.

Es la ÚNICA fuente de rutas. Todo el código usa estas constantes (pathlib,
relativas a la raíz del repo), nunca rutas escritas a mano.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DIR_CONFIG = RAIZ / "config"
# INVENTARIO_DATOS permite que los agentes en worktrees escriban los datos pesados en la carpeta principal.
DIR_DATOS = Path(os.getenv("INVENTARIO_DATOS") or RAIZ / "datos")
DIR_ESCANEOS = DIR_DATOS / "escaneos"      # fotos subidas por el escáner
DIR_MAQUETA = DIR_DATOS / "maqueta"        # fotos de la maqueta + escenas.csv
# INVENTARIO_MODELOS / INVENTARIO_LOGS: idem, para pesos y logs de TensorBoard.
DIR_MODELOS = Path(os.getenv("INVENTARIO_MODELOS") or RAIZ / "modelos")
DIR_WEB = RAIZ / "web"
DIR_LOGS = Path(os.getenv("INVENTARIO_LOGS") or RAIZ / "logs")

ARCHIVO_CLASES = DIR_CONFIG / "clases.json"
CLASE_DESCONOCIDO = "desconocido"


def _env(nombre: str, defecto: str) -> str:
    valor = os.getenv(nombre, "").strip()
    return valor or defecto


# Variables de entorno (ver .env.example)
VARIANTE_VISION = _env("VARIANTE_VISION", "mock")   # mock | A | B
UMBRAL_DET = float(_env("UMBRAL_DET", "0.25"))
UMBRAL_CONF = float(_env("UMBRAL_CONF", "0.5"))
MODELO_CLAUDE = _env("MODELO_CLAUDE", "claude-haiku-5-5")
PUERTO = int(_env("PUERTO", "8000"))


def url_bd() -> str:
    """URL de SQLAlchemy. Una URL sqlite relativa se resuelve contra la raíz del repo."""
    url = _env("URL_BD", "sqlite:///datos/inventario.db")
    prefijo = "sqlite:///"
    if url.startswith(prefijo) and url != "sqlite:///:memory:":
        ruta = Path(url[len(prefijo):])
        if not ruta.is_absolute():
            ruta = RAIZ / ruta
        ruta.parent.mkdir(parents=True, exist_ok=True)
        url = prefijo + ruta.as_posix()
    return url


@lru_cache(maxsize=1)
def cargar_clases() -> dict:
    """Contenido completo de config/clases.json."""
    return json.loads(ARCHIVO_CLASES.read_text(encoding="utf-8"))


def nombres_clases() -> list[str]:
    """Las 8 clases del modelo, en el orden de sus salidas (sin 'desconocido')."""
    return [c["clase"] for c in cargar_clases()["clases"]]


def clases_con_desconocido() -> list[str]:
    """Claves que siempre trae ResultadoConteo.conteo."""
    return nombres_clases() + [CLASE_DESCONOCIDO]
