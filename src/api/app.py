"""Fábrica de la aplicación FastAPI.

    crear_app(url_bd=None, pipeline=None, dir_escaneos=None, variante=None)

Los tests inyectan una BD temporal, el pipeline (p. ej. el mock con semilla) y la
carpeta de fotos. El servidor real usa src.api.main:app (todo desde src.config).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.api.pipeline import resolver_pipeline
from src.api.rutas_cierres import PREFIJO_URL_ESCANEOS
from src.api.rutas_cierres import router as router_cierres
from src.api.rutas_inventario import router as router_inventario
from src.api.rutas_laboratorio import router as router_laboratorio
from src.conciliacion import (AjusteInvalido, CierreNoEncontrado, CierreYaConfirmado,
                              ErrorConciliacion)
from src.db import crear_fabrica_sesiones, crear_motor, crear_tablas

DESCRIPCION = """
API del **control de inventario por visión** (arqueo de mercadería).

* CRUD de productos y movimientos (stock_inicial, entrada, venta, ajuste) y stock esperado.
* `POST /api/escaneos`: el celular sube 1-5 fotos del estante, el servidor cuenta con visión
  y crea un cierre pendiente con la conciliación contra la BD.
* `POST /api/cierres/{id}/confirmar`: el cajero corrige y confirma; se generan los ajustes.
* `POST /api/asistente`: preguntas en lenguaje natural sobre las diferencias (RAG).

Web: `/escaner/` (celular), `/panel/` (PC) y `/laboratorio/` (probar los modelos). Fotos: `/escaneos/...`.
"""

_CODIGOS_ERROR: dict[type[ErrorConciliacion], int] = {
    CierreNoEncontrado: 404,
    CierreYaConfirmado: 409,
    AjusteInvalido: 422,
}


class EstaticosOpcionales(StaticFiles):
    """StaticFiles que no falla si la carpeta todavía no existe o está vacía (responde 404)."""

    def __init__(self, directory: Path, html: bool = False) -> None:
        super().__init__(directory=directory, html=html, check_dir=False)

    async def check_config(self) -> None:
        return None

    def lookup_path(self, path: str) -> tuple[str, os.stat_result | None]:
        if not Path(self.directory).is_dir():
            return "", None
        return super().lookup_path(path)


def crear_app(url_bd: str | None = None, pipeline: Any = None,
              dir_escaneos: Path | str | None = None, variante: str | None = None,
              dir_web: Path | str | None = None) -> FastAPI:
    """Crea la app con su propio motor de BD (crea las tablas si faltan).

    url_bd:       URL de SQLAlchemy; por defecto src.config.url_bd() (URL_BD).
    pipeline:     función contar(imagenes) u objeto/módulo con .contar; por defecto la
                  variante de VARIANTE_VISION (hoy solo existe el mock).
    dir_escaneos: dónde se guardan las fotos; por defecto src.config.DIR_ESCANEOS.
    """
    variante = variante or config.VARIANTE_VISION
    dir_escaneos = Path(dir_escaneos or config.DIR_ESCANEOS)
    dir_escaneos.mkdir(parents=True, exist_ok=True)
    dir_web = Path(dir_web or config.DIR_WEB)

    motor = crear_motor(url_bd or config.url_bd())
    crear_tablas(motor)

    app = FastAPI(title="Control de inventario por visión", version="1.0.0",
                  description=DESCRIPCION, docs_url="/docs", redoc_url="/redoc")
    app.state.motor = motor
    app.state.fabrica_sesiones = crear_fabrica_sesiones(motor)
    app.state.contar = resolver_pipeline(pipeline, variante)
    app.state.variante = variante
    app.state.dir_escaneos = dir_escaneos

    # CORS abierto: solo para pruebas en la red local.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.exception_handler(ErrorConciliacion)
    async def _error_conciliacion(_request: Request, error: ErrorConciliacion) -> JSONResponse:
        codigo = next((c for t, c in _CODIGOS_ERROR.items() if isinstance(error, t)), 400)
        return JSONResponse(status_code=codigo, content={"detail": str(error)})

    app.include_router(router_inventario)
    app.include_router(router_cierres)
    app.include_router(router_laboratorio)

    @app.get("/", include_in_schema=False)
    def _raiz() -> RedirectResponse:
        return RedirectResponse("/panel/")

    app.mount(PREFIJO_URL_ESCANEOS, EstaticosOpcionales(dir_escaneos), name="escaneos")
    app.mount("/escaner", EstaticosOpcionales(dir_web / "escaner", html=True), name="escaner")
    app.mount("/panel", EstaticosOpcionales(dir_web / "panel", html=True), name="panel")
    app.mount("/laboratorio", EstaticosOpcionales(dir_web / "laboratorio", html=True), name="laboratorio")
    app.mount("/comun", EstaticosOpcionales(dir_web / "comun"), name="comun")
    return app
