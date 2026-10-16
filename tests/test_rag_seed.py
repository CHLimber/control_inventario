"""Seed propio del agente RAG (BD SQLite en memoria). No depende de src/db.

Escenario (HOY = jueves 15/10/2026):
- 01/10 08:00  stock_inicial de 20 unidades de cada uno de los 9 productos.
- 05/10 10:00  entrada de 12 Coca-Cola lata.
- todos los días 01..15/10 a las 15:00  venta de 1 unidad de cada producto.
- cierre 1 (08/10 21:00, confirmado): Coca-Cola lata −2, Fanta lata +1.
- cierre 2 (12/10 21:00, confirmado): Pepsi botella −2 (detectó 7, el cajero corrigió a 6),
  Sprite lata +1.
- cierre 3 (15/10 21:00, confirmado): Coca-Cola lata −1, 7Up lata −3, Fanta botella +2;
  2 cajas no reconocidas y una observación.
El stock esperado se lleva acá en un dict, independiente de la implementación de src/rag.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from contratos.modelos_bd import Base, Cierre, CierreDetalle, Deteccion, Movimiento, Producto
from src.config import cargar_clases

HOY = date(2026, 10, 15)

# clase -> (detectado - esperado, ajustado - esperado o None)
CIERRES = {
    1: (datetime(2026, 10, 8, 21), {"coca_lata": (-2, None), "fanta_lata": (1, None)}, None),
    2: (datetime(2026, 10, 12, 21), {"pepsi_botella": (-1, -2), "sprite_lata": (1, None)}, None),
    3: (datetime(2026, 10, 15, 21), {"coca_lata": (-1, None), "7up_lata": (-3, None),
                                     "fanta_botella": (2, None)}, "Luz baja en el estante"),
}


def crear_sesion_seed(con_datos: bool = True) -> tuple[Session, dict]:
    """Devuelve (sesion, info). info['stock_final'][clase] = stock esperado al fin de HOY;
    info['esperado'][id_cierre][clase] = esperado de ese cierre; info['ids'][clase] = id."""
    motor = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(motor)
    s = Session(motor)
    info: dict = {"ids": {}, "esperado": {}, "stock_final": {}}

    productos = {}
    for c in cargar_clases()["clases"]:
        p = Producto(nombre=c["nombre"], marca=c["marca"], formato=c["formato"],
                     precio=Decimal("5.00") if c["formato"] == "lata" else Decimal("10.00"),
                     clase_modelo=c["clase"], activo=True)
        s.add(p)
        productos[c["clase"]] = p
    s.flush()
    info["ids"] = {k: p.id for k, p in productos.items()}
    if not con_datos:
        s.commit()
        return s, info

    stock = {k: 0 for k in productos}
    eventos: list[tuple[datetime, str, str | int, int]] = []   # (fecha, tipo, clase|cierre, cant)
    for k in productos:
        eventos.append((datetime(2026, 10, 1, 8), "stock_inicial", k, 20))
    eventos.append((datetime(2026, 10, 5, 10), "entrada", "coca_lata", 12))
    d = date(2026, 10, 1)
    while d <= HOY:
        for k in productos:
            eventos.append((datetime(d.year, d.month, d.day, 15), "venta", k, 1))
        d += timedelta(days=1)
    for id_c, (fecha, _, _) in CIERRES.items():
        eventos.append((fecha, "cierre", id_c, 0))
    eventos.sort(key=lambda e: (e[0], e[1] == "cierre"))

    for fecha, tipo, clave, cant in eventos:
        if tipo == "cierre":
            _, difs, obs = CIERRES[clave]
            c = Cierre(id=clave, fecha=fecha, usuario="cajero", estado="confirmado",
                       variante_modelo="mock", observacion=obs,
                       imagen_path=f"cierre_{clave}/foto1_anotada.jpg")
            s.add(c)
            info["esperado"][clave] = dict(stock)
            for k, p in productos.items():
                det_off, aj_off = difs.get(k, (0, None))
                esperado = stock[k]
                detectado = esperado + det_off
                ajustado = esperado + aj_off if aj_off is not None else None
                diferencia = (ajustado if ajustado is not None else detectado) - esperado
                s.add(CierreDetalle(id_cierre=clave, id_producto=p.id, esperado=esperado,
                                    detectado=detectado, ajustado=ajustado,
                                    diferencia=diferencia, confianza_media=0.9))
                if diferencia:
                    s.add(Movimiento(id_producto=p.id, tipo="ajuste", cantidad=diferencia,
                                     fecha=fecha, usuario="cajero", id_cierre=clave))
                    stock[k] += diferencia
                for _ in range(min(detectado, 2)):
                    s.add(Deteccion(id_cierre=clave, id_producto=p.id, x=0, y=0, w=10, h=10,
                                    conf_detector=0.9, conf_clasificador=0.9))
            if clave == 3:
                for _ in range(2):
                    s.add(Deteccion(id_cierre=clave, id_producto=None, x=0, y=0, w=10, h=10,
                                    conf_detector=0.6, conf_clasificador=0.3))
            continue
        p = productos[clave]
        s.add(Movimiento(id_producto=p.id, tipo=tipo, cantidad=cant, fecha=fecha))
        if tipo == "stock_inicial":
            stock[clave] = cant
        elif tipo == "entrada":
            stock[clave] += cant
        elif tipo == "venta":
            stock[clave] -= cant
    s.commit()
    info["stock_final"] = dict(stock)
    return s, info


def test_seed_es_coherente():
    s, info = crear_sesion_seed()
    assert info["stock_final"]["coca_lata"] == 20 + 12 - 15 - 2 - 1
    assert info["stock_final"]["7up_lata"] == 20 - 15 - 3
    assert info["esperado"][2]["pepsi_botella"] == 20 - 12
    s.close()
