"""CONTRATO de la API REST (Pydantic v2). La implementa el agente BD-API y la consume el agente Web.

Cambiar este archivo solo con aprobación del orquestador.

Convenciones:
- Prefijo /api. JSON en UTF-8. Fechas ISO 8601 ("2026-10-15" o "2026-10-15T21:30:00").
- Errores: código HTTP 4xx/5xx con cuerpo ErrorOut {"detail": "..."} (formato de FastAPI).
- Las imágenes se sirven como estáticos en /escaneos/... (carpeta datos/escaneos/).
- La web se sirve en /escaner/ y /panel/ (carpetas web/escaner y web/panel).
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TipoMovimiento = Literal["stock_inicial", "entrada", "venta", "ajuste"]
EstadoCierre = Literal["pendiente", "confirmado"]
EstadoFila = Literal["ok", "faltante", "sobrante"]

# Tabla de endpoints (documentación ejecutable; la usan los tests de contrato).
ENDPOINTS: list[tuple[str, str, str]] = [
    # método, ruta, descripción  →  entrada / salida
    ("GET",    "/api/clases",                 "clases.json tal cual → dict"),
    ("GET",    "/api/productos",              "?incluir_inactivos=false → list[ProductoOut]"),
    ("POST",   "/api/productos",              "ProductoCrear → ProductoOut (201)"),
    ("GET",    "/api/productos/{id}",         "→ ProductoOut"),
    ("PUT",    "/api/productos/{id}",         "ProductoActualizar → ProductoOut"),
    ("DELETE", "/api/productos/{id}",         "baja lógica (activo=false) → 204"),
    ("GET",    "/api/movimientos",            "?desde&hasta&id_producto&tipo → list[MovimientoOut]"),
    ("POST",   "/api/movimientos",            "MovimientoCrear → MovimientoOut (201)"),
    ("GET",    "/api/stock",                  "?fecha=YYYY-MM-DD (fin de ese día; por defecto ahora) → StockOut"),
    ("POST",   "/api/escaneos",               "multipart: imagenes (1-5 archivos), usuario? → CierreOut (201, pendiente)"),
    ("GET",    "/api/cierres",                "?desde&hasta&estado → list[CierreResumen]"),
    ("GET",    "/api/cierres/{id}",           "→ CierreOut"),
    ("POST",   "/api/cierres/{id}/confirmar", "ConfirmarCierreIn → CierreOut (confirmado)"),
    ("GET",    "/api/diferencias",            "?id_producto&desde&hasta → list[DiferenciaHistorial]"),
    ("POST",   "/api/asistente",              "PreguntaIn → RespuestaAsistente"),
]

MAX_IMAGENES_ESCANEO = 5


class _Modelo(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ErrorOut(_Modelo):
    detail: str


# ---------------------------------------------------------------- Productos
class ProductoCrear(_Modelo):
    nombre: str = Field(min_length=1, max_length=100)
    marca: str = Field(min_length=1, max_length=50)
    formato: Literal["botella", "lata"]
    precio: Decimal = Field(ge=0, decimal_places=2)
    clase_modelo: str = Field(description="Clave de config/clases.json; única")
    activo: bool = True


class ProductoActualizar(_Modelo):
    """Todos opcionales: solo se cambian los campos enviados."""
    nombre: str | None = None
    marca: str | None = None
    formato: Literal["botella", "lata"] | None = None
    precio: Decimal | None = Field(default=None, ge=0)
    clase_modelo: str | None = None
    activo: bool | None = None


class ProductoOut(ProductoCrear):
    id: int


# ---------------------------------------------------------------- Movimientos
class MovimientoCrear(_Modelo):
    id_producto: int
    tipo: TipoMovimiento
    cantidad: int = Field(description="> 0, salvo 'ajuste' que lleva signo (≠ 0)")
    fecha: datetime | None = Field(default=None, description="Por defecto, ahora")
    usuario: str = "cajero"


class MovimientoOut(_Modelo):
    id: int
    id_producto: int
    tipo: TipoMovimiento
    cantidad: int
    fecha: datetime
    usuario: str
    id_cierre: int | None = None


# ---------------------------------------------------------------- Stock
class StockItem(_Modelo):
    id_producto: int
    clase_modelo: str
    nombre: str
    stock_inicial: int
    entradas: int
    ventas: int
    ajustes: int
    stock: int = Field(description="stock_inicial + entradas − ventas ± ajustes")


class StockOut(_Modelo):
    fecha: datetime
    items: list[StockItem]


# ---------------------------------------------------------------- Escaneo / cierre
class CajaOut(_Modelo):
    x: float
    y: float
    w: float
    h: float
    clase: str
    conf_det: float
    conf_cls: float | None = None
    indice_imagen: int = 0


class FilaConciliacion(_Modelo):
    id_producto: int
    clase_modelo: str
    nombre: str
    esperado: int
    detectado: int
    ajustado: int | None = None
    diferencia: int = Field(description="(ajustado ?? detectado) − esperado; < 0 faltante")
    estado: EstadoFila
    confianza_media: float | None = None


class Totales(_Modelo):
    esperado: int
    detectado: int
    diferencia: int
    productos_con_diferencia: int


class CierreResumen(_Modelo):
    id: int
    fecha: datetime
    usuario: str
    estado: EstadoCierre
    variante_modelo: str
    totales: Totales


class CierreOut(CierreResumen):
    observacion: str | None = None
    imagen_anotada_url: str | None = Field(default=None, description="p. ej. /escaneos/cierre_12/foto1_anotada.jpg")
    imagenes_urls: list[str] = []
    filas: list[FilaConciliacion]
    cajas: list[CajaOut] = []
    no_reconocidos: int = Field(0, description="Cantidad de cajas 'desconocido'")
    tiempo_ms: float = 0.0


class AjusteFila(_Modelo):
    id_producto: int
    ajustado: int = Field(ge=0)


class ConfirmarCierreIn(_Modelo):
    """Conteos corregidos por el cajero. Los productos no enviados quedan con ajustado = detectado.

    Al confirmar se genera un movimiento 'ajuste' (ajustado − esperado) por cada producto
    con diferencia ≠ 0, así el stock esperado del día siguiente parte del conteo real.
    """
    ajustes: list[AjusteFila] = []
    observacion: str | None = None
    usuario: str = "cajero"


class DiferenciaHistorial(_Modelo):
    id_cierre: int
    fecha: datetime
    id_producto: int
    nombre: str
    esperado: int
    detectado: int
    ajustado: int | None
    diferencia: int


# ---------------------------------------------------------------- Asistente (RAG)
class PreguntaIn(_Modelo):
    pregunta: str = Field(min_length=1, max_length=500)


class RespuestaAsistente(_Modelo):
    respuesta: str
    intencion: str = Field(description="faltantes_rango | detalle_cierre | stock_producto | diferencia_puntual | desconocida")
    parametros: dict[str, Any] = Field(default_factory=dict, description="Fechas/productos extraídos")
    datos: list[dict[str, Any]] = Field(default_factory=list, description="Registros usados como contexto")
    sin_datos: bool = False
    modelo: str | None = Field(default=None, description="Modelo de Claude usado; None si se usó plantilla")
