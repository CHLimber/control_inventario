"""Recuperación estructurada: consultas SQLAlchemy que arman el contexto del asistente.

Cada función devuelve un dict JSON-serializable:
    {"resumen": {...}, "registros": [ {...}, ... ]}   # registros: como máximo MAX_FILAS
Todas las cifras (stock, totales, faltantes) se calculan acá, en Python; el LLM solo redacta.
"""
from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from contratos.modelos_bd import Cierre, CierreDetalle, Deteccion, Movimiento, Producto

MAX_FILAS = 50


def _inicio(d: date) -> datetime:
    return datetime.combine(d, time.min)


def _fin(d: date) -> datetime:
    return datetime.combine(d, time.max)


def _iso(v):
    if isinstance(v, datetime):
        return v.isoformat(timespec="minutes")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def estado_fila(diferencia: int) -> str:
    return "ok" if diferencia == 0 else ("faltante" if diferencia < 0 else "sobrante")


def productos_activos(sesion: Session) -> list[Producto]:
    return list(sesion.scalars(select(Producto).where(Producto.activo.is_(True))
                               .order_by(Producto.id)))


# ------------------------------------------------------------------ stock
def stock_producto(sesion: Session, producto: Producto, hasta: datetime) -> dict:
    """Stock esperado a la fecha `hasta` según docs/esquema_bd.md.

    base = último stock_inicial con fecha ≤ hasta (0 si no hay); luego se suman
    entradas, se restan ventas y se suman ajustes (con signo) en (base.fecha, hasta].
    """
    base = sesion.scalars(
        select(Movimiento)
        .where(Movimiento.id_producto == producto.id, Movimiento.tipo == "stock_inicial",
               Movimiento.fecha <= hasta)
        .order_by(Movimiento.fecha.desc(), Movimiento.id.desc())
        .limit(1)).first()
    filtro = [Movimiento.id_producto == producto.id, Movimiento.fecha <= hasta,
              Movimiento.tipo.in_(("entrada", "venta", "ajuste"))]
    if base is not None:
        filtro.append(Movimiento.fecha > base.fecha)
    sumas = dict(sesion.execute(
        select(Movimiento.tipo, func.coalesce(func.sum(Movimiento.cantidad), 0))
        .where(*filtro).group_by(Movimiento.tipo)).all())
    inicial = base.cantidad if base is not None else 0
    entradas, ventas, ajustes = (int(sumas.get(k, 0)) for k in ("entrada", "venta", "ajuste"))
    return {
        "producto": producto.nombre,
        "clase_modelo": producto.clase_modelo,
        "stock_inicial": inicial,
        "fecha_stock_inicial": _iso(base.fecha) if base is not None else None,
        "entradas": entradas,
        "ventas": ventas,
        "ajustes": ajustes,
        "stock": inicial + entradas - ventas + ajustes,
    }


def consultar_stock(sesion: Session, ids: list[int], fecha: date, hoy: date) -> dict:
    productos = [p for p in productos_activos(sesion) if not ids or p.id in ids]
    # El stock "de hoy" es el actual: fin del día de hoy.
    hasta = _fin(min(fecha, hoy))
    registros = [stock_producto(sesion, p, hasta) for p in productos][:MAX_FILAS]
    hay_movimientos = sesion.scalar(
        select(func.count(Movimiento.id)).where(Movimiento.fecha <= hasta)) or 0
    return {
        "resumen": {
            "fecha_corte": _iso(hasta),
            "productos": len(registros),
            "stock_total_unidades": sum(r["stock"] for r in registros),
            "hay_movimientos": bool(hay_movimientos),
        },
        "registros": registros if hay_movimientos else [],
    }


# ------------------------------------------------------------------ diferencias
def _fila_detalle(det: CierreDetalle, cierre: Cierre, producto: Producto) -> dict:
    contado = det.ajustado if det.ajustado is not None else det.detectado
    return {
        "id_cierre": cierre.id,
        "fecha": _iso(cierre.fecha),
        "estado_cierre": cierre.estado,
        "producto": producto.nombre,
        "esperado": det.esperado,
        "detectado": det.detectado,
        "ajustado": det.ajustado,
        "contado": contado,
        "diferencia": det.diferencia,
        "estado": estado_fila(det.diferencia),
    }


def _cierres_en(sesion: Session, desde: date, hasta: date) -> list[Cierre]:
    return list(sesion.scalars(
        select(Cierre).where(Cierre.fecha >= _inicio(desde), Cierre.fecha <= _fin(hasta))
        .order_by(Cierre.fecha, Cierre.id)))


def consultar_faltantes(sesion: Session, desde: date, hasta: date, ids: list[int]) -> dict:
    """Faltantes y sobrantes (diferencia ≠ 0) de los cierres en [desde, hasta]."""
    cierres = _cierres_en(sesion, desde, hasta)
    filas = sesion.execute(
        select(CierreDetalle, Cierre, Producto)
        .join(Cierre, CierreDetalle.id_cierre == Cierre.id)
        .join(Producto, CierreDetalle.id_producto == Producto.id)
        .where(Cierre.fecha >= _inicio(desde), Cierre.fecha <= _fin(hasta),
               CierreDetalle.diferencia != 0,
               *( [CierreDetalle.id_producto.in_(ids)] if ids else [] ))
        .order_by(Cierre.fecha, Cierre.id, Producto.id)).all()
    registros = [_fila_detalle(d, c, p) for d, c, p in filas]

    por_producto: dict[str, dict] = {}
    for r in registros:
        acc = por_producto.setdefault(r["producto"], {"producto": r["producto"],
                                                      "faltante_total": 0, "sobrante_total": 0,
                                                      "neto": 0})
        if r["diferencia"] < 0:
            acc["faltante_total"] += -r["diferencia"]
        else:
            acc["sobrante_total"] += r["diferencia"]
        acc["neto"] += r["diferencia"]
    return {
        "resumen": {
            "desde": desde.isoformat(),
            "hasta": hasta.isoformat(),
            "cierres_revisados": [{"id_cierre": c.id, "fecha": _iso(c.fecha), "estado": c.estado}
                                  for c in cierres][:MAX_FILAS],
            "unidades_faltantes": sum(a["faltante_total"] for a in por_producto.values()),
            "unidades_sobrantes": sum(a["sobrante_total"] for a in por_producto.values()),
            "por_producto": sorted(por_producto.values(), key=lambda a: a["neto"]),
            "filas_truncadas": len(registros) > MAX_FILAS,
        },
        "registros": registros[:MAX_FILAS],
        "hay_cierres": bool(cierres),
    }


def consultar_diferencia_producto(sesion: Session, ids: list[int], desde: date | None,
                                  hasta: date | None, id_cierre: int | None) -> dict:
    """Esperado vs contado de productos concretos, cierre por cierre (incluye los 'ok')."""
    q = (select(CierreDetalle, Cierre, Producto)
         .join(Cierre, CierreDetalle.id_cierre == Cierre.id)
         .join(Producto, CierreDetalle.id_producto == Producto.id)
         .where(CierreDetalle.id_producto.in_(ids)))
    if id_cierre is not None:
        q = q.where(Cierre.id == id_cierre)
    if desde is not None:
        q = q.where(Cierre.fecha >= _inicio(desde))
    if hasta is not None:
        q = q.where(Cierre.fecha <= _fin(hasta))
    # Los más recientes primero para que el recorte a MAX_FILAS deje los últimos.
    filas = sesion.execute(q.order_by(Cierre.fecha.desc(), Cierre.id.desc(), Producto.id)
                           .limit(MAX_FILAS)).all()
    registros = [_fila_detalle(d, c, p) for d, c, p in reversed(filas)]
    return {
        "resumen": {
            "cierres": len({r["id_cierre"] for r in registros}),
            "diferencia_neta": sum(r["diferencia"] for r in registros),
            "unidades_faltantes": sum(-r["diferencia"] for r in registros if r["diferencia"] < 0),
            "unidades_sobrantes": sum(r["diferencia"] for r in registros if r["diferencia"] > 0),
        },
        "registros": registros,
    }


# ------------------------------------------------------------------ cierre
def buscar_cierre(sesion: Session, id_cierre: int | None, desde: date | None,
                  hasta: date | None) -> Cierre | None:
    """Por id; si no, el último del rango; si no hay rango, el último de todos."""
    if id_cierre is not None:
        return sesion.get(Cierre, id_cierre)
    q = select(Cierre)
    if desde is not None:
        q = q.where(Cierre.fecha >= _inicio(desde))
    if hasta is not None:
        q = q.where(Cierre.fecha <= _fin(hasta))
    return sesion.scalars(q.order_by(Cierre.fecha.desc(), Cierre.id.desc()).limit(1)).first()


def consultar_cierre(sesion: Session, cierre: Cierre) -> dict:
    filas = sesion.execute(
        select(CierreDetalle, Producto)
        .join(Producto, CierreDetalle.id_producto == Producto.id)
        .where(CierreDetalle.id_cierre == cierre.id).order_by(Producto.id)).all()
    registros = []
    for d, p in filas:
        fila = _fila_detalle(d, cierre, p)
        for k in ("id_cierre", "fecha", "estado_cierre"):
            fila.pop(k)
        fila["confianza_media"] = round(d.confianza_media, 3) if d.confianza_media is not None else None
        registros.append(fila)
    no_reconocidos = sesion.scalar(
        select(func.count(Deteccion.id)).where(Deteccion.id_cierre == cierre.id,
                                               Deteccion.id_producto.is_(None))) or 0
    return {
        "resumen": {
            "id_cierre": cierre.id,
            "fecha": _iso(cierre.fecha),
            "estado": cierre.estado,
            "usuario": cierre.usuario,
            "variante_modelo": cierre.variante_modelo,
            "observacion": cierre.observacion,
            "total_esperado": sum(r["esperado"] for r in registros),
            "total_contado": sum(r["contado"] for r in registros),
            "diferencia_total": sum(r["diferencia"] for r in registros),
            "productos_con_diferencia": sum(1 for r in registros if r["diferencia"] != 0),
            "unidades_faltantes": sum(-r["diferencia"] for r in registros if r["diferencia"] < 0),
            "unidades_sobrantes": sum(r["diferencia"] for r in registros if r["diferencia"] > 0),
            "no_reconocidos": no_reconocidos,
        },
        "registros": registros[:MAX_FILAS],
    }
