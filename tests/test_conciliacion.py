"""Stock esperado y conciliación de cierres (lógica pura, sin FastAPI)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from contratos.modelos_bd import Movimiento, Producto
from contratos.vision_types import Caja, ResultadoConteo
from src.conciliacion import (AjusteInvalido, CierreNoEncontrado, CierreYaConfirmado,
                              calcular_diferencia, calcular_totales, completar_cierre,
                              confirmar_cierre, estado_fila, iniciar_cierre,
                              resumir_movimientos, stock_a_fecha, stock_de_producto)
from src.config import clases_con_desconocido
from src.db import crear_fabrica_sesiones, crear_motor, crear_tablas
from src.db.seed import sembrar, stock_final_esperado

DIA = date(2026, 10, 14)


@dataclass
class M:
    tipo: str
    cantidad: int
    fecha: datetime
    id: int = 0


def h(hora: int, minuto: int = 0, dia: int = 14) -> datetime:
    return datetime(2026, 10, dia, hora, minuto)


@pytest.fixture()
def sesion(tmp_path: Path):
    motor = crear_motor(f"sqlite:///{(tmp_path / 'prueba.db').as_posix()}")
    crear_tablas(motor)
    with crear_fabrica_sesiones(motor)() as s:
        sembrar(s, DIA)
        yield s
    motor.dispose()


def _producto(sesion, clase: str) -> Producto:
    return sesion.scalar(select(Producto).where(Producto.clase_modelo == clase))


def _resultado(conteo: dict[str, int], cajas: list[Caja] | None = None) -> ResultadoConteo:
    completo = {c: 0 for c in clases_con_desconocido()} | conteo
    return ResultadoConteo(conteo=completo, cajas=cajas or [], imagen_anotada_path=None,
                           variante="mock")


# ---------------------------------------------------------------- stock (puro)
def test_stock_suma_entradas_resta_ventas():
    movs = [M("stock_inicial", 10, h(7)), M("entrada", 5, h(9)), M("venta", 3, h(12)),
            M("venta", 2, h(18))]
    r = resumir_movimientos(movs, h(23))
    assert (r.stock_inicial, r.entradas, r.ventas, r.ajustes, r.stock) == (10, 5, 5, 0, 10)


def test_stock_respeta_la_fecha_de_corte():
    movs = [M("stock_inicial", 10, h(7)), M("venta", 3, h(12)), M("venta", 2, h(18))]
    assert resumir_movimientos(movs, h(13)).stock == 7
    assert resumir_movimientos(movs, h(6)).stock == 0       # antes de todo


def test_stock_inicial_nuevo_reinicia():
    movs = [M("stock_inicial", 10, h(7, dia=13)), M("venta", 4, h(12, dia=13)),
            M("ajuste", -1, h(22, dia=13)),
            M("stock_inicial", 20, h(7)), M("venta", 2, h(12))]
    r = resumir_movimientos(movs, h(23))
    assert (r.stock_inicial, r.ventas, r.ajustes, r.stock) == (20, 2, 0, 18)
    # A una fecha anterior al reinicio vale el stock_inicial viejo.
    assert resumir_movimientos(movs, h(23, dia=13)).stock == 5


def test_stock_inicial_ignora_movimientos_de_su_mismo_instante():
    movs = [M("venta", 3, h(7), id=1), M("stock_inicial", 10, h(7), id=2)]
    assert resumir_movimientos(movs, h(8)).stock == 10


def test_stock_inicial_empate_gana_el_ultimo_id():
    movs = [M("stock_inicial", 10, h(7), id=1), M("stock_inicial", 4, h(7), id=2)]
    assert resumir_movimientos(movs, h(8)).stock == 4


def test_ajustes_con_signo():
    movs = [M("stock_inicial", 10, h(7)), M("ajuste", -3, h(20)), M("ajuste", 1, h(21))]
    r = resumir_movimientos(movs, h(23))
    assert r.ajustes == -2 and r.stock == 8


def test_sin_movimientos_stock_cero():
    assert resumir_movimientos([], h(23)).stock == 0


# ---------------------------------------------------------------- stock (BD)
def test_stock_a_fecha_con_el_seed(sesion):
    stock = {p.clase_modelo: r.stock for p, r in stock_a_fecha(sesion, datetime(2026, 10, 14, 23))}
    assert stock == stock_final_esperado()
    assert stock["7up_lata"] == 0


def test_stock_a_fecha_excluye_inactivos(sesion):
    _producto(sesion, "7up_lata").activo = False
    sesion.flush()
    clases = {p.clase_modelo for p, _ in stock_a_fecha(sesion, datetime(2026, 10, 14, 23))}
    assert "7up_lata" not in clases and len(clases) == 8


# ---------------------------------------------------------------- reglas de conciliación
@pytest.mark.parametrize("esperado,detectado,ajustado,dif,estado", [
    (5, 3, None, -2, "faltante"),
    (5, 7, None, 2, "sobrante"),
    (5, 5, None, 0, "ok"),
    (5, 3, 5, 0, "ok"),          # el cajero corrige al esperado
    (0, 0, None, 0, "ok"),       # producto en cero
    (0, 1, None, 1, "sobrante"),
])
def test_diferencia_y_estado(esperado, detectado, ajustado, dif, estado):
    assert calcular_diferencia(esperado, detectado, ajustado) == dif
    assert estado_fila(dif) == estado


# ---------------------------------------------------------------- cierre completo
def _crear_cierre(sesion, conteo: dict[str, int], cajas=None):
    cierre = iniciar_cierre(sesion, fecha=datetime(2026, 10, 14, 21, 0))
    completar_cierre(sesion, cierre, _resultado(conteo, cajas))
    sesion.commit()
    return cierre


def test_conciliacion_faltante_sobrante_y_cero(sesion):
    esperado = stock_final_esperado()        # coca_lata 8, fanta_lata 4, 7up_lata 0
    conteo = dict(esperado) | {"coca_lata": 6, "fanta_lata": 5}
    cajas = [Caja(0, 0, 10, 10, 0.9, "coca_lata", 0.8), Caja(0, 0, 10, 10, 0.9, "coca_lata", 0.6),
             Caja(0, 0, 10, 10, 0.7, "desconocido", 0.3)]
    cierre = _crear_cierre(sesion, conteo, cajas)

    filas = {d.producto.clase_modelo: d for d in cierre.detalles}
    assert len(filas) == 9
    assert (filas["coca_lata"].esperado, filas["coca_lata"].detectado,
            filas["coca_lata"].diferencia) == (8, 6, -2)
    assert estado_fila(filas["coca_lata"].diferencia) == "faltante"
    assert filas["fanta_lata"].diferencia == 1 and estado_fila(1) == "sobrante"
    assert (filas["7up_lata"].esperado, filas["7up_lata"].diferencia) == (0, 0)
    assert filas["coca_lata"].confianza_media == pytest.approx(0.7)
    assert filas["sprite_lata"].confianza_media is None
    assert len(cierre.detecciones) == 3
    assert sum(1 for d in cierre.detecciones if d.id_producto is None) == 1
    totales = calcular_totales(cierre.detalles)
    assert totales["diferencia"] == -1 and totales["productos_con_diferencia"] == 2
    assert cierre.estado == "pendiente"


def test_confirmar_crea_ajustes_y_actualiza_el_stock(sesion):
    conteo = stock_final_esperado() | {"coca_lata": 6, "fanta_lata": 5}
    cierre = _crear_cierre(sesion, conteo)
    coca, pepsi = _producto(sesion, "coca_lata"), _producto(sesion, "pepsi_lata")

    # El cajero recontó: coca_lata en realidad hay 7; pepsi_lata (esperado 5) hay 3.
    confirmar_cierre(sesion, cierre.id, {coca.id: 7, pepsi.id: 3}, observacion="recontado")
    sesion.commit()

    filas = {d.producto.clase_modelo: d for d in cierre.detalles}
    assert (filas["coca_lata"].ajustado, filas["coca_lata"].diferencia) == (7, -1)
    assert (filas["pepsi_lata"].ajustado, filas["pepsi_lata"].diferencia) == (3, -2)
    assert (filas["fanta_lata"].ajustado, filas["fanta_lata"].diferencia) == (5, 1)  # = detectado
    assert filas["sprite_lata"].diferencia == 0
    assert cierre.estado == "confirmado" and cierre.observacion == "recontado"

    ajustes = list(sesion.scalars(select(Movimiento).where(Movimiento.tipo == "ajuste")))
    por_clase = {a.producto.clase_modelo: a for a in ajustes}
    assert {c: a.cantidad for c, a in por_clase.items()} == {
        "coca_lata": -1, "pepsi_lata": -2, "fanta_lata": 1}
    assert all(a.id_cierre == cierre.id for a in ajustes)

    # El stock esperado posterior parte del conteo real.
    despues = datetime(2026, 10, 14, 23, 59)
    assert stock_de_producto(sesion, coca.id, despues) == 7
    assert stock_de_producto(sesion, pepsi.id, despues) == 3


def test_no_se_reconfirma(sesion):
    cierre = _crear_cierre(sesion, stock_final_esperado())
    confirmar_cierre(sesion, cierre.id)
    sesion.commit()
    assert not list(sesion.scalars(select(Movimiento).where(Movimiento.tipo == "ajuste")))
    with pytest.raises(CierreYaConfirmado):
        confirmar_cierre(sesion, cierre.id)


def test_errores_de_confirmacion(sesion):
    with pytest.raises(CierreNoEncontrado):
        confirmar_cierre(sesion, 999)
    cierre = _crear_cierre(sesion, stock_final_esperado())
    with pytest.raises(AjusteInvalido):
        confirmar_cierre(sesion, cierre.id, {12345: 1})
    with pytest.raises(AjusteInvalido):
        confirmar_cierre(sesion, cierre.id, {_producto(sesion, "coca_lata").id: -1})
