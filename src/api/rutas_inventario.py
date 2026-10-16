"""Rutas del CRUD: clases, productos, movimientos y stock."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from contratos.api_schemas import (MovimientoCrear, MovimientoOut, ProductoActualizar,
                                   ProductoCrear, ProductoOut, StockItem, StockOut,
                                   TipoMovimiento)
from contratos.modelos_bd import Movimiento, Producto
from src.api.dependencias import a_hora_local, obtener_sesion, parsear_fecha
from src.conciliacion import stock_a_fecha
from src.config import cargar_clases, nombres_clases

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- clases
@router.get("/clases", tags=["clases"], summary="config/clases.json tal cual")
def listar_clases() -> dict:
    return cargar_clases()


# ---------------------------------------------------------------- productos
def _producto_o_404(sesion: Session, id_producto: int) -> Producto:
    producto = sesion.get(Producto, id_producto)
    if producto is None:
        raise HTTPException(404, f"No existe el producto {id_producto}")
    return producto


def _validar_clase(sesion: Session, clase: str, excluir_id: int | None = None) -> None:
    if clase not in nombres_clases():
        raise HTTPException(422, f"clase_modelo {clase!r} no está en config/clases.json "
                                 f"(válidas: {', '.join(nombres_clases())})")
    otro = sesion.scalar(select(Producto).where(Producto.clase_modelo == clase))
    if otro is not None and otro.id != excluir_id:
        raise HTTPException(409, f"Ya existe un producto con clase_modelo {clase!r} (id {otro.id})")


@router.get("/productos", response_model=list[ProductoOut], tags=["productos"])
def listar_productos(incluir_inactivos: bool = False, sesion: Session = Depends(obtener_sesion)):
    consulta = select(Producto).order_by(Producto.id)
    if not incluir_inactivos:
        consulta = consulta.where(Producto.activo.is_(True))
    return list(sesion.scalars(consulta))


@router.post("/productos", response_model=ProductoOut, status_code=201, tags=["productos"])
def crear_producto(datos: ProductoCrear, sesion: Session = Depends(obtener_sesion)):
    _validar_clase(sesion, datos.clase_modelo)
    producto = Producto(**datos.model_dump())
    sesion.add(producto)
    sesion.commit()
    return producto


@router.get("/productos/{id}", response_model=ProductoOut, tags=["productos"])
def obtener_producto(id: int, sesion: Session = Depends(obtener_sesion)):
    return _producto_o_404(sesion, id)


@router.put("/productos/{id}", response_model=ProductoOut, tags=["productos"])
def actualizar_producto(id: int, datos: ProductoActualizar,
                        sesion: Session = Depends(obtener_sesion)):
    producto = _producto_o_404(sesion, id)
    cambios = datos.model_dump(exclude_unset=True)
    for campo in ("nombre", "marca", "formato", "precio", "clase_modelo", "activo"):
        if campo in cambios and cambios[campo] is None:
            raise HTTPException(422, f"El campo {campo!r} no puede ser null")
    if "nombre" in cambios and not cambios["nombre"].strip():
        raise HTTPException(422, "El nombre no puede estar vacío")
    if "clase_modelo" in cambios and cambios["clase_modelo"] != producto.clase_modelo:
        _validar_clase(sesion, cambios["clase_modelo"], excluir_id=producto.id)
    for campo, valor in cambios.items():
        setattr(producto, campo, valor)
    sesion.commit()
    return producto


@router.delete("/productos/{id}", status_code=204, tags=["productos"],
               summary="Baja lógica (activo=false)")
def borrar_producto(id: int, sesion: Session = Depends(obtener_sesion)):
    producto = _producto_o_404(sesion, id)
    producto.activo = False
    sesion.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------- movimientos
@router.get("/movimientos", response_model=list[MovimientoOut], tags=["movimientos"])
def listar_movimientos(desde: str | None = Query(None, description="YYYY-MM-DD o ISO"),
                       hasta: str | None = Query(None, description="YYYY-MM-DD (incluye todo ese día) o ISO"),
                       id_producto: int | None = None, tipo: TipoMovimiento | None = None,
                       sesion: Session = Depends(obtener_sesion)):
    consulta = select(Movimiento).order_by(Movimiento.fecha, Movimiento.id)
    if (d := parsear_fecha(desde, parametro="desde")) is not None:
        consulta = consulta.where(Movimiento.fecha >= d)
    if (h := parsear_fecha(hasta, fin_de_dia=True, parametro="hasta")) is not None:
        consulta = consulta.where(Movimiento.fecha <= h)
    if id_producto is not None:
        consulta = consulta.where(Movimiento.id_producto == id_producto)
    if tipo is not None:
        consulta = consulta.where(Movimiento.tipo == tipo)
    return list(sesion.scalars(consulta))


@router.post("/movimientos", response_model=MovimientoOut, status_code=201, tags=["movimientos"])
def crear_movimiento(datos: MovimientoCrear, sesion: Session = Depends(obtener_sesion)):
    producto = _producto_o_404(sesion, datos.id_producto)
    if not producto.activo:
        raise HTTPException(422, f"El producto {producto.id} está dado de baja")
    if datos.tipo == "ajuste":
        if datos.cantidad == 0:
            raise HTTPException(422, "Un ajuste debe tener cantidad ≠ 0 (+ sobrante, − faltante)")
    elif datos.tipo == "stock_inicial":
        if datos.cantidad < 0:
            raise HTTPException(422, "El stock inicial no puede ser negativo")
    elif datos.cantidad <= 0:
        raise HTTPException(422, f"La cantidad de una {datos.tipo} debe ser > 0")
    movimiento = Movimiento(
        id_producto=producto.id, tipo=datos.tipo, cantidad=datos.cantidad,
        fecha=a_hora_local(datos.fecha) if datos.fecha else datetime.now(),
        usuario=datos.usuario or "cajero")
    sesion.add(movimiento)
    sesion.commit()
    return movimiento


# ---------------------------------------------------------------- stock
@router.get("/stock", response_model=StockOut, tags=["stock"])
def consultar_stock(fecha: str | None = Query(
        None, description="YYYY-MM-DD → fin de ese día; ISO → ese instante; vacío → ahora"),
        sesion: Session = Depends(obtener_sesion)):
    hasta = parsear_fecha(fecha, fin_de_dia=True) or datetime.now()
    items = [StockItem(id_producto=p.id, clase_modelo=p.clase_modelo, nombre=p.nombre,
                       stock_inicial=r.stock_inicial, entradas=r.entradas, ventas=r.ventas,
                       ajustes=r.ajustes, stock=r.stock)
             for p, r in stock_a_fecha(sesion, hasta)]
    return StockOut(fecha=hasta, items=items)
