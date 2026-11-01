"""Punto de entrada para uvicorn:

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Usa la configuración de src.config (.env): URL_BD, VARIANTE_VISION, INVENTARIO_DATOS.
Para tests, importar crear_app desde src.api.app (no crea la BD real).
Con la variante A o B, los modelos se cargan en segundo plano mientras arranca el servidor.
"""
from src.api.app import crear_app, precargar_en_segundo_plano

app = crear_app()
if app.state.variante in ("A", "B"):
    precargar_en_segundo_plano(app.state.variante)
