"""Orquestación del asistente: entender -> recuperar (SQL) -> generar."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from contratos.api_schemas import RespuestaAsistente

from . import consultas, generacion
from .extraccion import (clasificar, extraer_fechas, extraer_id_cierre,
                         extraer_productos)

DIAS_POR_DEFECTO = 7    # "¿qué faltó?" sin fecha = últimos 7 días


def _recuperar(intencion: str, sesion: Session, rango, productos, id_cierre: int | None,
               hoy: date, parametros: dict[str, Any]) -> dict | None:
    """Ejecuta la consulta de la intención. Devuelve el contexto, o None si no hay datos."""
    if intencion == "stock_producto":
        fecha = rango.hasta if rango else hoy
        parametros["fecha_stock"] = min(fecha, hoy).isoformat()
        ctx = consultas.consultar_stock(sesion, productos.ids, fecha, hoy)
        return ctx if ctx["registros"] else None

    if intencion == "faltantes_rango":
        desde, hasta = ((rango.desde, rango.hasta) if rango
                        else (hoy - timedelta(days=DIAS_POR_DEFECTO - 1), hoy))
        parametros.setdefault("fechas", {"desde": desde.isoformat(), "hasta": hasta.isoformat(),
                                         "expresion": f"ultimos {DIAS_POR_DEFECTO} dias"})
        ctx = consultas.consultar_faltantes(sesion, desde, hasta, productos.ids)
        return ctx if ctx.pop("hay_cierres") else None

    if intencion == "diferencia_puntual":
        ctx = consultas.consultar_diferencia_producto(
            sesion, productos.ids, rango.desde if rango else None,
            rango.hasta if rango else None, id_cierre)
        return ctx if ctx["registros"] else None

    if intencion == "detalle_cierre":
        cierre = consultas.buscar_cierre(sesion, id_cierre, rango.desde if rango else None,
                                         rango.hasta if rango else None)
        if cierre is None:
            return None
        parametros["id_cierre"] = cierre.id
        ctx = consultas.consultar_cierre(sesion, cierre)
        return ctx if ctx["registros"] else None
    return None


def responder(pregunta: str, sesion: Session, cliente=None,
              hoy: date | None = None) -> RespuestaAsistente:
    """Responde una pregunta en español sobre el inventario.

    - `sesion`: sesión SQLAlchemy abierta (no se hace commit ni se escribe nada).
    - `cliente`: cliente de Anthropic (o un doble de prueba). Si es None se crea uno con
      ANTHROPIC_API_KEY; sin clave, o si la API falla, responde con plantilla (modelo=None).
    - `hoy`: fecha de referencia para "hoy", "ayer", etc. (por defecto, date.today()).
    """
    hoy = hoy or date.today()
    productos = extraer_productos(pregunta, consultas.productos_activos(sesion))
    rango = extraer_fechas(pregunta, hoy)
    id_cierre = extraer_id_cierre(pregunta)
    intencion = clasificar(pregunta, bool(productos), id_cierre)

    parametros: dict[str, Any] = {}
    if rango:
        parametros["fechas"] = rango.como_dict()
    if productos:
        parametros["productos"] = productos.nombres
        parametros["menciones"] = productos.menciones
    if id_cierre is not None:
        parametros["id_cierre"] = id_cierre

    if intencion == "desconocida":
        return RespuestaAsistente(respuesta=generacion.NO_ENTENDI, intencion=intencion,
                                  parametros=parametros, sin_datos=True)

    contexto = _recuperar(intencion, sesion, rango, productos, id_cierre, hoy, parametros)
    if contexto is None:
        return RespuestaAsistente(respuesta=generacion.SIN_DATOS, intencion=intencion,
                                  parametros=parametros, sin_datos=True)

    contexto = {"intencion": intencion, "fecha_de_hoy": hoy.isoformat(),
                "parametros": parametros, **contexto}
    generado = generacion.generar_con_llm(pregunta, contexto, cliente)
    if generado is None:
        texto, modelo = generacion.plantilla(intencion, contexto), None
    else:
        texto, modelo = generado
    return RespuestaAsistente(respuesta=texto, intencion=intencion, parametros=parametros,
                              datos=contexto["registros"], sin_datos=False, modelo=modelo)
