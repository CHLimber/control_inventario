"""Datos de ejemplo: los 9 productos de config/clases.json y un día ficticio de movimientos.

Uso desde código:  sembrar(sesion, dia=date(2026, 10, 14))
Uso por consola:   python scripts/seed.py [--fecha 2026-10-14] [--reiniciar]
"""
from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from contratos.modelos_bd import Movimiento, Producto
from src.config import cargar_clases

# Precios de referencia en Bs. Botella = 2 L; lata = 355 ml.
PRECIOS_BS: dict[str, Decimal] = {
    "coca_botella": Decimal("15.00"),
    "coca_lata": Decimal("6.00"),
    "fanta_botella": Decimal("13.00"),
    "fanta_lata": Decimal("5.50"),
    "sprite_botella": Decimal("13.00"),
    "sprite_lata": Decimal("5.50"),
    "pepsi_botella": Decimal("12.00"),
    "pepsi_lata": Decimal("5.00"),
    "7up_lata": Decimal("5.00"),
}

# Día ficticio por producto: (stock_inicial 07:00, entrada 10:00, ventas a lo largo del día).
MOVIMIENTOS_DIA: dict[str, tuple[int, int, list[int]]] = {
    "coca_botella": (6, 4, [2, 1, 2]),     # 6 + 4 − 5 = 5
    "coca_lata": (8, 6, [3, 2, 1]),        # 8 + 6 − 6 = 8
    "fanta_botella": (4, 0, [1, 1]),       # 2
    "fanta_lata": (6, 0, [2]),             # 4
    "sprite_botella": (4, 2, [1]),         # 5
    "sprite_lata": (5, 0, [1, 1]),         # 3
    "pepsi_botella": (3, 0, [1]),          # 2
    "pepsi_lata": (5, 4, [2, 2]),          # 5
    "7up_lata": (2, 0, [2]),               # 0: termina el día en cero
}
_HORA_STOCK_INICIAL = time(7, 0)
_HORA_ENTRADA = time(10, 0)
_HORAS_VENTA = [time(11, 30), time(15, 0), time(19, 45)]


def sembrar_productos(sesion: Session) -> dict[str, Producto]:
    """Crea los productos de clases.json que falten (idempotente). Devuelve {clase: Producto}."""
    existentes = {p.clase_modelo: p for p in sesion.scalars(select(Producto))}
    for c in cargar_clases()["clases"]:
        if c["clase"] in existentes:
            continue
        p = Producto(nombre=c["nombre"], marca=c["marca"], formato=c["formato"],
                     precio=PRECIOS_BS.get(c["clase"], Decimal("5.00")),
                     clase_modelo=c["clase"], activo=True)
        sesion.add(p)
        existentes[c["clase"]] = p
    sesion.flush()
    return existentes


def sembrar_movimientos(sesion: Session, productos: dict[str, Producto], dia: date,
                        usuario: str = "cajero") -> int:
    """Agrega el día ficticio de movimientos. Devuelve cuántos movimientos creó."""
    n = 0
    for clase, (inicial, entrada, ventas) in MOVIMIENTOS_DIA.items():
        p = productos.get(clase)
        if p is None:
            continue
        filas = [("stock_inicial", inicial, _HORA_STOCK_INICIAL)]
        if entrada:
            filas.append(("entrada", entrada, _HORA_ENTRADA))
        filas += [("venta", v, _HORAS_VENTA[i % len(_HORAS_VENTA)]) for i, v in enumerate(ventas)]
        for tipo, cantidad, hora in filas:
            sesion.add(Movimiento(id_producto=p.id, tipo=tipo, cantidad=cantidad,
                                  fecha=datetime.combine(dia, hora), usuario=usuario))
            n += 1
    sesion.flush()
    return n


def stock_final_esperado() -> dict[str, int]:
    """Stock al final del día ficticio, por clase (útil en tests y en la demo)."""
    return {c: i + e - sum(v) for c, (i, e, v) in MOVIMIENTOS_DIA.items()}


def sembrar(sesion: Session, dia: date | None = None, con_movimientos: bool = True) -> dict:
    """Productos + (opcional) movimientos del día `dia` (por defecto hoy). Hace commit."""
    dia = dia or date.today()
    productos = sembrar_productos(sesion)
    n = sembrar_movimientos(sesion, productos, dia) if con_movimientos else 0
    sesion.commit()
    return {"productos": len(productos), "movimientos": n, "dia": dia.isoformat()}
