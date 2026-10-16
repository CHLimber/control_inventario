"""Stock esperado a una fecha (regla de docs/esquema_bd.md).

    base   = último movimiento stock_inicial con fecha ≤ t   (0 si no hay)
    stock  = base.cantidad + Σ entradas − Σ ventas + Σ ajustes   (fecha en (base.fecha, t])

`resumir_movimientos` es lógica pura (sin BD); `stock_a_fecha` la aplica a la sesión.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from contratos.modelos_bd import Movimiento, Producto


class _MovimientoLike(Protocol):
    tipo: str
    cantidad: int
    fecha: datetime


@dataclass(frozen=True)
class ResumenStock:
    stock_inicial: int = 0
    entradas: int = 0
    ventas: int = 0
    ajustes: int = 0          # con signo

    @property
    def stock(self) -> int:
        return self.stock_inicial + self.entradas - self.ventas + self.ajustes


def fin_del_dia(dia: date) -> datetime:
    return datetime.combine(dia, time.max)


def resumir_movimientos(movimientos: Iterable[_MovimientoLike],
                        hasta: datetime | None = None) -> ResumenStock:
    """Aplica la regla del stock esperado a los movimientos de UN producto.

    Si hay dos stock_inicial con la misma fecha gana el de mayor `id` (el último cargado).
    Los movimientos con la misma fecha que el stock_inicial base no cuentan: el intervalo
    es (base.fecha, t].
    """
    hasta = hasta or datetime.now()
    movs = [m for m in movimientos if m.fecha <= hasta]
    iniciales = [m for m in movs if m.tipo == "stock_inicial"]
    if iniciales:
        base = max(iniciales, key=lambda m: (m.fecha, getattr(m, "id", 0) or 0))
        movs = [m for m in movs if m.fecha > base.fecha]
        stock_inicial = base.cantidad
    else:
        stock_inicial = 0
    sumas: dict[str, int] = defaultdict(int)
    for m in movs:
        sumas[m.tipo] += m.cantidad
    return ResumenStock(stock_inicial=stock_inicial, entradas=sumas["entrada"],
                        ventas=sumas["venta"], ajustes=sumas["ajuste"])


def stock_a_fecha(sesion: Session, hasta: datetime | None = None, solo_activos: bool = True,
                  ids_producto: Iterable[int] | None = None) -> list[tuple[Producto, ResumenStock]]:
    """Stock esperado de cada producto a la fecha `hasta` (por defecto, ahora), ordenado por id."""
    hasta = hasta or datetime.now()
    consulta = select(Producto).order_by(Producto.id)
    if solo_activos:
        consulta = consulta.where(Producto.activo.is_(True))
    if ids_producto is not None:
        consulta = consulta.where(Producto.id.in_(list(ids_producto)))
    productos = list(sesion.scalars(consulta))
    if not productos:
        return []
    por_producto: dict[int, list[Movimiento]] = defaultdict(list)
    movs = sesion.scalars(select(Movimiento).where(
        Movimiento.fecha <= hasta, Movimiento.id_producto.in_([p.id for p in productos])))
    for m in movs:
        por_producto[m.id_producto].append(m)
    return [(p, resumir_movimientos(por_producto[p.id], hasta)) for p in productos]


def stock_de_producto(sesion: Session, id_producto: int, hasta: datetime | None = None) -> int:
    filas = stock_a_fecha(sesion, hasta, solo_activos=False, ids_producto=[id_producto])
    return filas[0][1].stock if filas else 0
