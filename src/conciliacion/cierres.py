"""Conciliación del cierre: conteo de visión vs. stock esperado (docs/esquema_bd.md).

Flujo (sin FastAPI; la API solo maneja archivos y HTTP):
    1. iniciar_cierre()    crea el cierre 'pendiente' (para tener id y carpeta de fotos).
    2. completar_cierre()  guarda el detalle por producto activo y las detecciones.
    3. confirmar_cierre()  guarda los ajustados y crea un movimiento 'ajuste' por cada
                           diferencia ≠ 0. Un cierre confirmado no se reconfirma.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.orm import Session

from contratos.modelos_bd import Cierre, CierreDetalle, Deteccion, Movimiento, Producto
from contratos.vision_types import ResultadoConteo
from src.conciliacion.stock import stock_a_fecha


# ---------------------------------------------------------------- errores
class ErrorConciliacion(Exception):
    """Base de los errores de negocio (la API los traduce a 4xx)."""


class CierreNoEncontrado(ErrorConciliacion):
    pass


class CierreYaConfirmado(ErrorConciliacion):
    pass


class AjusteInvalido(ErrorConciliacion):
    pass


# ---------------------------------------------------------------- reglas puras
def calcular_diferencia(esperado: int, detectado: int, ajustado: int | None = None) -> int:
    """(ajustado ?? detectado) − esperado. Negativa = faltante; positiva = sobrante."""
    return (detectado if ajustado is None else ajustado) - esperado


def estado_fila(diferencia: int) -> str:
    if diferencia < 0:
        return "faltante"
    if diferencia > 0:
        return "sobrante"
    return "ok"


def calcular_totales(detalles: Iterable[CierreDetalle]) -> dict[str, int]:
    detalles = list(detalles)
    return {
        "esperado": sum(d.esperado for d in detalles),
        # Conteo final: lo ajustado por el cajero si ya confirmó; así esperado + diferencia = detectado.
        "detectado": sum(d.detectado if d.ajustado is None else d.ajustado for d in detalles),
        "diferencia": sum(d.diferencia for d in detalles),
        "productos_con_diferencia": sum(1 for d in detalles if d.diferencia != 0),
    }


# ---------------------------------------------------------------- consultas
def obtener_cierre(sesion: Session, id_cierre: int) -> Cierre:
    cierre = sesion.get(Cierre, id_cierre)
    if cierre is None:
        raise CierreNoEncontrado(f"No existe el cierre {id_cierre}")
    return cierre


def detalles_ordenados(cierre: Cierre) -> list[CierreDetalle]:
    return sorted(cierre.detalles, key=lambda d: d.id_producto)


# ---------------------------------------------------------------- flujo del cierre
def iniciar_cierre(sesion: Session, *, usuario: str = "cajero", variante: str = "mock",
                   fecha: datetime | None = None) -> Cierre:
    """Crea el cierre pendiente y hace flush para obtener su id (sin commit)."""
    cierre = Cierre(fecha=fecha or datetime.now(), usuario=usuario, estado="pendiente",
                    variante_modelo=variante)
    sesion.add(cierre)
    sesion.flush()
    return cierre


def _ruta_relativa(ruta: Path | None, base: Path | None) -> str | None:
    if ruta is None:
        return None
    ruta = Path(ruta)
    if base is not None:
        try:
            return ruta.resolve().relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            pass
    return ruta.as_posix()


def completar_cierre(sesion: Session, cierre: Cierre, resultado: ResultadoConteo,
                     dir_escaneos: Path | None = None) -> Cierre:
    """Guarda el detalle (una fila por producto activo) y las detecciones del conteo.

    esperado = stock a la fecha del cierre; detectado = resultado.conteo[clase_modelo].
    Las cajas de clases que no son productos de la BD se guardan con id_producto NULL.
    """
    cierre.variante_modelo = resultado.variante or cierre.variante_modelo
    cierre.imagen_path = _ruta_relativa(resultado.imagen_anotada_path, dir_escaneos)

    todos = {p.clase_modelo: p for p in sesion.scalars(select(Producto))}
    confianzas: dict[str, list[float]] = defaultdict(list)
    for caja in resultado.cajas:
        producto = todos.get(caja.clase)
        confianza = caja.conf_cls if caja.conf_cls is not None else caja.conf_det
        confianzas[caja.clase].append(float(confianza))
        cierre.detecciones.append(Deteccion(
            id_producto=producto.id if producto else None, indice_imagen=caja.indice_imagen,
            x=float(caja.x), y=float(caja.y), w=float(caja.w), h=float(caja.h),
            conf_detector=float(caja.conf_det),
            conf_clasificador=None if caja.conf_cls is None else float(caja.conf_cls)))

    for producto, resumen in stock_a_fecha(sesion, cierre.fecha):
        detectado = int(resultado.conteo.get(producto.clase_modelo, 0))
        confs = confianzas.get(producto.clase_modelo)
        cierre.detalles.append(CierreDetalle(
            id_producto=producto.id, esperado=resumen.stock, detectado=detectado, ajustado=None,
            diferencia=calcular_diferencia(resumen.stock, detectado),
            confianza_media=round(fmean(confs), 4) if confs else None))
    sesion.flush()
    return cierre


def confirmar_cierre(sesion: Session, id_cierre: int, ajustes: Mapping[int, int] | None = None,
                     *, observacion: str | None = None, usuario: str = "cajero") -> Cierre:
    """Confirma el cierre con los conteos corregidos por el cajero (sin commit).

    `ajustes` = {id_producto: ajustado}; los productos no enviados quedan con
    ajustado = detectado. Por cada diferencia ≠ 0 se crea un movimiento 'ajuste'
    con fecha del cierre, así el stock posterior parte del conteo real.
    """
    cierre = obtener_cierre(sesion, id_cierre)
    if cierre.estado == "confirmado":
        raise CierreYaConfirmado(f"El cierre {id_cierre} ya está confirmado")
    ajustes = dict(ajustes or {})
    detalles = {d.id_producto: d for d in cierre.detalles}
    ajenos = sorted(set(ajustes) - set(detalles))
    if ajenos:
        raise AjusteInvalido(f"Productos que no están en el cierre {id_cierre}: {ajenos}")
    for id_producto, valor in ajustes.items():
        if valor is None or int(valor) < 0:
            raise AjusteInvalido(f"Ajustado inválido para el producto {id_producto}: {valor}")

    for id_producto, d in sorted(detalles.items()):
        d.ajustado = int(ajustes.get(id_producto, d.detectado))
        d.diferencia = calcular_diferencia(d.esperado, d.detectado, d.ajustado)
        if d.diferencia != 0:
            sesion.add(Movimiento(id_producto=id_producto, tipo="ajuste", cantidad=d.diferencia,
                                  fecha=cierre.fecha, usuario=usuario, id_cierre=cierre.id))
    cierre.estado = "confirmado"
    if observacion is not None:
        cierre.observacion = observacion
    sesion.flush()
    return cierre
