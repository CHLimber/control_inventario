"""Punto de entrada para uvicorn:

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Usa la configuración de src.config (.env): URL_BD, VARIANTE_VISION, INVENTARIO_DATOS.
Para tests, importar crear_app desde src.api.app (no crea la BD real).
"""
from src.api.app import crear_app

app = crear_app()
