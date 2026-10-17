"""API REST: CRUD, errores 4xx y flujo completo venta → escaneo → confirmación (con el mock)."""
from __future__ import annotations

import io
import shutil
import sys
import types
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from contratos.api_schemas import ENDPOINTS, CierreOut, RespuestaAsistente
from src import config
from src.api.app import crear_app
from src.db import crear_fabrica_sesiones, crear_motor, crear_tablas
from src.db.seed import MOVIMIENTOS_DIA, sembrar, stock_final_esperado
from src.vision import pipeline_mock

SEMILLA = 7
AYER = date.today() - timedelta(days=1)


def _jpg(ancho: int = 640, alto: int = 480) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (ancho, alto), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def entorno(tmp_path: Path, monkeypatch):
    """App con BD temporal sembrada (día de ayer), fotos en tmp_path y mock con semilla."""
    datos = tmp_path / "datos"
    monkeypatch.setenv("INVENTARIO_DATOS", str(datos))
    monkeypatch.setattr(config, "DIR_ESCANEOS", datos / "escaneos")
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    motor = crear_motor(url)
    crear_tablas(motor)
    with crear_fabrica_sesiones(motor)() as s:
        sembrar(s, AYER)
    motor.dispose()
    app = crear_app(url_bd=url, pipeline=partial(pipeline_mock.contar, semilla=SEMILLA),
                    dir_web=tmp_path / "web")
    with TestClient(app) as cliente:
        cliente.tmp_path = tmp_path
        cliente.dir_escaneos = datos / "escaneos"
        yield cliente
    app.state.motor.dispose()


def _ids(cliente) -> dict[str, int]:
    return {p["clase_modelo"]: p["id"] for p in cliente.get("/api/productos").json()}


def _escanear(cliente, n: int = 1, contenido: bytes | None = None):
    archivos = [("imagenes", (f"f{i}.jpg", contenido or _jpg(), "image/jpeg")) for i in range(n)]
    return cliente.post("/api/escaneos", files=archivos, data={"usuario": "ana"})


# ---------------------------------------------------------------- rutas
def test_todos_los_endpoints_del_contrato_existen(entorno):
    esquema = entorno.get("/openapi.json").json()["paths"]
    rutas = {(m.upper(), ruta) for ruta, metodos in esquema.items() for m in metodos}
    assert rutas == {(m, r) for m, r, _ in ENDPOINTS}
    assert entorno.get("/docs").status_code == 200


def test_clases(entorno):
    r = entorno.get("/api/clases")
    assert r.status_code == 200 and len(r.json()["clases"]) == 8


# ---------------------------------------------------------------- productos
def test_crud_productos(entorno):
    lista = entorno.get("/api/productos").json()
    assert len(lista) == 8 and lista[0]["precio"]

    # Liberar una clase para poder crear otro producto con ella.
    ids = _ids(entorno)
    assert entorno.delete(f"/api/productos/{ids['sprite_botella']}").status_code == 204
    assert len(entorno.get("/api/productos").json()) == 7
    inactivos = entorno.get("/api/productos", params={"incluir_inactivos": True}).json()
    assert len(inactivos) == 8 and not next(p for p in inactivos if p["clase_modelo"] == "sprite_botella")["activo"]

    r = entorno.put(f"/api/productos/{ids['sprite_botella']}", json={"precio": "4.50", "activo": True})
    assert r.status_code == 200 and r.json()["precio"] == "4.50" and r.json()["activo"]

    r = entorno.get(f"/api/productos/{ids['coca_lata']}")
    assert r.status_code == 200 and r.json()["nombre"] == "Coca-Cola lata"


def test_crear_producto_y_errores(entorno):
    nuevo = {"nombre": "Coca-Cola lata 2", "marca": "Coca-Cola", "formato": "lata",
             "precio": "6.00", "clase_modelo": "coca_lata"}
    assert entorno.post("/api/productos", json=nuevo).status_code == 409      # clase repetida
    assert entorno.post("/api/productos", json=nuevo | {"clase_modelo": "inca_kola"}).status_code == 422
    assert entorno.post("/api/productos", json=nuevo | {"formato": "caja"}).status_code == 422
    assert entorno.post("/api/productos", json=nuevo | {"precio": "-1"}).status_code == 422
    assert entorno.get("/api/productos/999").status_code == 404
    assert entorno.put("/api/productos/999", json={"nombre": "x"}).status_code == 404
    assert entorno.delete("/api/productos/999").status_code == 404
    ids = _ids(entorno)
    assert entorno.put(f"/api/productos/{ids['coca_lata']}",
                       json={"clase_modelo": "fanta_lata"}).status_code == 409
    assert entorno.put(f"/api/productos/{ids['coca_lata']}", json={"nombre": None}).status_code == 422

    # Reenviar su propia clase no es conflicto.
    r = entorno.put(f"/api/productos/{ids['sprite_botella']}", json={"clase_modelo": "sprite_botella",
                                                               "nombre": "Sprite 2 L"})
    assert r.status_code == 200 and r.json()["nombre"] == "Sprite 2 L"


# ---------------------------------------------------------------- movimientos y stock
def test_movimientos_y_stock(entorno):
    ids = _ids(entorno)
    stock = {i["clase_modelo"]: i["stock"] for i in entorno.get("/api/stock").json()["items"]}
    assert stock == stock_final_esperado()

    r = entorno.post("/api/movimientos", json={"id_producto": ids["coca_lata"], "tipo": "venta",
                                               "cantidad": 2})
    assert r.status_code == 201 and r.json()["usuario"] == "cajero" and r.json()["id_cierre"] is None
    r = entorno.post("/api/movimientos", json={"id_producto": ids["coca_lata"], "tipo": "ajuste",
                                               "cantidad": -1})
    assert r.status_code == 201

    items = {i["clase_modelo"]: i for i in entorno.get("/api/stock").json()["items"]}
    coca = items["coca_lata"]
    assert (coca["stock_inicial"], coca["entradas"], coca["ventas"], coca["ajustes"],
            coca["stock"]) == (8, 6, 8, -1, 5)

    # El stock de ayer (fin del día) no ve los movimientos de hoy.
    ayer = entorno.get("/api/stock", params={"fecha": AYER.isoformat()}).json()
    assert {i["clase_modelo"]: i["stock"] for i in ayer["items"]}["coca_lata"] == 8
    antes = entorno.get("/api/stock", params={"fecha": (AYER - timedelta(days=1)).isoformat()})
    assert all(i["stock"] == 0 for i in antes.json()["items"])

    # Filtros
    hoy = entorno.get("/api/movimientos", params={"desde": date.today().isoformat()}).json()
    assert len(hoy) == 2
    ventas = entorno.get("/api/movimientos", params={"tipo": "venta", "id_producto": ids["coca_lata"]})
    assert len(ventas.json()) == 4
    solo_ayer = entorno.get("/api/movimientos", params={"desde": AYER.isoformat(),
                                                        "hasta": AYER.isoformat()}).json()
    assert len(solo_ayer) == sum(1 + (e > 0) + len(v) for _, e, v in MOVIMIENTOS_DIA.values())


@pytest.mark.parametrize("cuerpo", [
    {"tipo": "venta", "cantidad": 0},
    {"tipo": "entrada", "cantidad": -3},
    {"tipo": "ajuste", "cantidad": 0},
    {"tipo": "stock_inicial", "cantidad": -1},
    {"tipo": "robo", "cantidad": 1},
])
def test_movimiento_invalido_422(entorno, cuerpo):
    ids = _ids(entorno)
    assert entorno.post("/api/movimientos", json={"id_producto": ids["coca_lata"]} | cuerpo).status_code == 422


def test_movimiento_otros_errores(entorno):
    assert entorno.post("/api/movimientos", json={"id_producto": 999, "tipo": "venta",
                                                  "cantidad": 1}).status_code == 404
    assert entorno.get("/api/movimientos", params={"desde": "15/10/2026"}).status_code == 422
    assert entorno.get("/api/stock", params={"fecha": "ayer"}).status_code == 422
    ids = _ids(entorno)
    entorno.delete(f"/api/productos/{ids['pepsi_lata']}")
    assert entorno.post("/api/movimientos", json={"id_producto": ids["pepsi_lata"], "tipo": "venta",
                                                  "cantidad": 1}).status_code == 422


# ---------------------------------------------------------------- escaneos: errores
def test_escaneo_sin_imagenes_422(entorno):
    assert entorno.post("/api/escaneos", data={"usuario": "ana"}).status_code == 422


def test_escaneo_mas_de_5_imagenes_422(entorno):
    assert _escanear(entorno, n=6).status_code == 422


def test_escaneo_archivo_que_no_es_imagen_422(entorno):
    assert _escanear(entorno, contenido=b"esto no es una foto").status_code == 422
    assert entorno.get("/api/cierres").json() == []


def test_cierre_inexistente_404(entorno):
    assert entorno.get("/api/cierres/999").status_code == 404
    assert entorno.post("/api/cierres/999/confirmar", json={}).status_code == 404


def test_pipeline_que_falla_no_deja_cierre(entorno):
    def contar_roto(_imagenes):
        raise RuntimeError("modelo no cargado")

    entorno.app.state.contar = contar_roto
    r = _escanear(entorno)
    assert r.status_code == 500 and "modelo no cargado" in r.json()["detail"]
    assert entorno.get("/api/cierres").json() == []
    assert not any(entorno.dir_escaneos.glob("cierre_*"))


# ---------------------------------------------------------------- flujo completo
def test_flujo_venta_escaneo_confirmacion(entorno):
    ids = _ids(entorno)

    # 1. Venta de hoy: coca_lata pasa de 8 a 5.
    entorno.post("/api/movimientos", json={"id_producto": ids["coca_lata"], "tipo": "venta",
                                           "cantidad": 3})
    esperado = stock_final_esperado() | {"coca_lata": 5}

    # Lo que va a contar el mock con la misma semilla y una foto igual.
    referencia = entorno.tmp_path / "referencia" / "foto.jpg"
    referencia.parent.mkdir()
    referencia.write_bytes(_jpg())
    conteo = pipeline_mock.contar([referencia], semilla=SEMILLA).conteo

    # 2. Escaneo.
    r = _escanear(entorno)
    assert r.status_code == 201, r.text
    cierre = CierreOut.model_validate(r.json())
    assert cierre.estado == "pendiente" and cierre.usuario == "ana" and cierre.variante_modelo == "mock"
    assert len(cierre.filas) == 8
    for fila in cierre.filas:
        assert fila.esperado == esperado[fila.clase_modelo]
        assert fila.detectado == conteo[fila.clase_modelo]
        assert fila.diferencia == fila.detectado - fila.esperado
        assert fila.estado == ("ok" if fila.diferencia == 0 else
                               "faltante" if fila.diferencia < 0 else "sobrante")
    assert cierre.no_reconocidos == conteo["desconocido"]
    assert len(cierre.cajas) == sum(conteo.values())
    assert cierre.totales.esperado == sum(esperado.values())
    assert cierre.totales.detectado == sum(conteo[c] for c in esperado)
    assert cierre.tiempo_ms > 0

    # Fotos guardadas fuera del repo y servidas en /escaneos.
    carpeta = entorno.dir_escaneos / f"cierre_{cierre.id}"
    assert (carpeta / "foto1.jpg").exists()
    assert cierre.imagenes_urls == [f"/escaneos/cierre_{cierre.id}/foto1.jpg"]
    assert cierre.imagen_anotada_url == f"/escaneos/cierre_{cierre.id}/foto1_anotada.jpg"
    assert entorno.get(cierre.imagen_anotada_url).status_code == 200
    assert entorno.dir_escaneos.is_relative_to(entorno.tmp_path)

    # GET del cierre y listado.
    assert entorno.get(f"/api/cierres/{cierre.id}").json()["filas"] == r.json()["filas"]
    listado = entorno.get("/api/cierres", params={"estado": "pendiente"}).json()
    assert [c["id"] for c in listado] == [cierre.id]

    # 3. Confirmación: el cajero corrige coca_lata a 4 (faltante de 1) y deja el resto.
    r = entorno.post(f"/api/cierres/{cierre.id}/confirmar",
                     json={"ajustes": [{"id_producto": ids["coca_lata"], "ajustado": 4}],
                           "observacion": "recontado", "usuario": "jefe"})
    assert r.status_code == 200, r.text
    confirmado = CierreOut.model_validate(r.json())
    assert confirmado.estado == "confirmado" and confirmado.observacion == "recontado"
    filas = {f.clase_modelo: f for f in confirmado.filas}
    assert (filas["coca_lata"].ajustado, filas["coca_lata"].diferencia,
            filas["coca_lata"].estado) == (4, -1, "faltante")
    for clase, fila in filas.items():
        if clase != "coca_lata":
            assert fila.ajustado == fila.detectado == conteo[clase]

    # Movimientos de ajuste: uno por cada diferencia ≠ 0, con id_cierre.
    ajustes = entorno.get("/api/movimientos", params={"tipo": "ajuste"}).json()
    esperados = {f.id_producto: f.diferencia for f in confirmado.filas if f.diferencia != 0}
    assert {a["id_producto"]: a["cantidad"] for a in ajustes} == esperados
    assert all(a["id_cierre"] == cierre.id and a["usuario"] == "jefe" for a in ajustes)

    # El stock ahora coincide con lo contado.
    stock = {i["clase_modelo"]: i["stock"] for i in entorno.get("/api/stock").json()["items"]}
    assert stock == {f.clase_modelo: f.ajustado for f in confirmado.filas}

    # No se reconfirma.
    assert entorno.post(f"/api/cierres/{cierre.id}/confirmar", json={}).status_code == 409

    # Historial de diferencias.
    hist = entorno.get("/api/diferencias", params={"id_producto": ids["coca_lata"]}).json()
    assert len(hist) == 1 and hist[0]["diferencia"] == -1 and hist[0]["ajustado"] == 4
    assert len(entorno.get("/api/diferencias").json()) == 8
    assert entorno.get("/api/diferencias", params={"hasta": AYER.isoformat()}).json() == []


def test_escaneo_con_varias_fotos_usa_la_mediana(entorno):
    r = _escanear(entorno, n=3)
    assert r.status_code == 201
    cuerpo = r.json()
    assert len(cuerpo["imagenes_urls"]) == 3
    assert {c["indice_imagen"] for c in cuerpo["cajas"]} == {0, 1, 2}


def test_confirmar_con_producto_ajeno_422(entorno):
    cierre = _escanear(entorno).json()
    r = entorno.post(f"/api/cierres/{cierre['id']}/confirmar",
                     json={"ajustes": [{"id_producto": 999, "ajustado": 1}]})
    assert r.status_code == 422
    r = entorno.post(f"/api/cierres/{cierre['id']}/confirmar",
                     json={"ajustes": [{"id_producto": 1, "ajustado": -1}]})
    assert r.status_code == 422
    assert entorno.get(f"/api/cierres/{cierre['id']}").json()["estado"] == "pendiente"


# ---------------------------------------------------------------- asistente
def test_asistente_sin_rag_503(entorno, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.rag", types.ModuleType("src.rag"))
    r = entorno.post("/api/asistente", json={"pregunta": "¿qué faltó ayer?"})
    assert r.status_code == 503 and "src.rag" in r.json()["detail"]


def test_asistente_con_rag(entorno, monkeypatch):
    falso = types.ModuleType("src.rag")
    falso.responder = lambda pregunta, sesion: RespuestaAsistente(
        respuesta=f"eco: {pregunta}", intencion="desconocida", sin_datos=True)
    monkeypatch.setitem(sys.modules, "src.rag", falso)
    r = entorno.post("/api/asistente", json={"pregunta": "hola"})
    assert r.status_code == 200 and r.json()["respuesta"] == "eco: hola"
    assert entorno.post("/api/asistente", json={"pregunta": ""}).status_code == 422


# ---------------------------------------------------------------- estáticos
def test_estaticos_y_raiz(entorno):
    r = entorno.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/panel/"
    # Sin carpeta web no falla: 404.
    assert entorno.get("/panel/").status_code == 404
    assert entorno.get("/escaner/").status_code == 404
    panel = entorno.tmp_path / "web" / "panel"
    panel.mkdir(parents=True)
    assert entorno.get("/panel/").status_code == 404           # carpeta vacía
    (panel / "index.html").write_text("<h1>Panel</h1>", encoding="utf-8")
    r = entorno.get("/panel/")
    assert r.status_code == 200 and "Panel" in r.text


def test_cors_abierto(entorno):
    r = entorno.options("/api/productos", headers={"Origin": "http://192.168.1.50:5500",
                                                    "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] in ("*", "http://192.168.1.50:5500")


def test_app_por_defecto_usa_config(tmp_path: Path, monkeypatch):
    """crear_app() sin argumentos toma URL_BD y DIR_ESCANEOS de src.config (aquí, temporales)."""
    monkeypatch.setenv("URL_BD", f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    monkeypatch.setattr(config, "DIR_ESCANEOS", tmp_path / "esc")
    app = crear_app()
    with TestClient(app) as c:
        assert c.get("/api/productos").json() == []
    app.state.motor.dispose()
    assert (tmp_path / "x.db").exists() and (tmp_path / "esc").is_dir()
    shutil.rmtree(tmp_path / "esc")
