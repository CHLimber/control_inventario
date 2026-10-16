"""Consultas contra el seed propio, plantillas sin clave y cliente de Claude mockeado.

Ningún test de este archivo llama a la API real: ANTHROPIC_API_KEY se vacía siempre.
"""
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from contratos.api_schemas import RespuestaAsistente
from contratos.modelos_bd import Cierre, Movimiento, Producto
from src import config
from src.rag import consultas, responder
from src.rag.generacion import MAX_TOKENS, SYSTEM_PROMPT
from tests.test_rag_seed import CIERRES, HOY, crear_sesion_seed


@pytest.fixture(autouse=True)
def sin_clave(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def seed():
    s, info = crear_sesion_seed()
    yield s, info
    s.close()


def _cliente_mock(texto="Respuesta redactada por Claude."):
    cliente = MagicMock()
    cliente.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=texto)], stop_reason="end_turn")
    return cliente


# ------------------------------------------------------------------ consultas
def test_stock_reglas_del_esquema(seed):
    s, info = seed
    coca_lata = s.get(Producto, info["ids"]["coca_lata"])
    r = consultas.stock_producto(s, coca_lata, datetime(2026, 10, 15, 23, 59))
    assert r == {"producto": "Coca-Cola lata", "clase_modelo": "coca_lata", "stock_inicial": 20,
                 "fecha_stock_inicial": "2026-10-01T08:00", "entradas": 12, "ventas": 15,
                 "ajustes": -3, "stock": info["stock_final"]["coca_lata"]}
    # A una fecha pasada, sin los movimientos posteriores
    r = consultas.stock_producto(s, coca_lata, datetime(2026, 10, 4, 23, 59))
    assert r["stock"] == 20 - 4 and r["entradas"] == 0


def test_stock_inicial_nuevo_reinicia(seed):
    s, info = seed
    pid = info["ids"]["sprite_botella"]
    s.add(Movimiento(id_producto=pid, tipo="stock_inicial", cantidad=50,
                               fecha=datetime(2026, 10, 15, 22)))
    s.commit()
    p = s.get(Producto, pid)
    assert consultas.stock_producto(s, p, datetime(2026, 10, 15, 23, 59))["stock"] == 50


def test_stock_coincide_con_esperado_de_cada_cierre(seed):
    s, info = seed
    for id_c, (fecha, _, _) in CIERRES.items():
        for clase, pid in info["ids"].items():
            p = s.get(Producto, pid)
            # justo antes del escaneo (los ajustes del cierre llevan la misma fecha)
            antes = fecha - timedelta(seconds=1)
            assert consultas.stock_producto(s, p, antes)["stock"] == info["esperado"][id_c][clase]


def test_consulta_faltantes_hoy(seed):
    s, _ = seed
    ctx = consultas.consultar_faltantes(s, HOY, HOY, [])
    assert ctx["hay_cierres"]
    res = ctx["resumen"]
    assert res["unidades_faltantes"] == 4 and res["unidades_sobrantes"] == 2
    assert [c["id_cierre"] for c in res["cierres_revisados"]] == [3]
    assert {r["producto"]: r["diferencia"] for r in ctx["registros"]} == {
        "Coca-Cola lata": -1, "7Up lata": -3, "Fanta botella": 2}


def test_consulta_cierre_detalle(seed):
    s, info = seed
    ctx = consultas.consultar_cierre(s, s.get(Cierre, 2))
    res = ctx["resumen"]
    assert res["productos_con_diferencia"] == 2 and res["diferencia_total"] == -1
    pepsi = next(r for r in ctx["registros"] if r["producto"] == "Pepsi botella")
    assert (pepsi["esperado"], pepsi["detectado"], pepsi["ajustado"], pepsi["diferencia"]) == (8, 7, 6, -2)
    assert len(ctx["registros"]) == 9


# ------------------------------------------------------------------ responder (plantilla)
def test_que_falto_hoy_con_plantilla(seed):
    s, _ = seed
    r = responder("¿Qué faltó hoy?", s, hoy=HOY)
    assert isinstance(r, RespuestaAsistente)
    assert r.intencion == "faltantes_rango" and not r.sin_datos and r.modelo is None
    assert r.parametros["fechas"]["desde"] == "2026-10-15"
    assert "7Up lata" in r.respuesta and "faltaron 3 unidades" in r.respuesta
    assert "4 unidades faltantes y 2 unidades sobrantes" in r.respuesta
    assert len(r.datos) == 3
    json.dumps(r.model_dump(mode="json"))     # serializable para la API


def test_faltantes_esta_semana(seed):
    s, _ = seed
    r = responder("faltantes de esta semana", s, hoy=HOY)
    ids = {d["id_cierre"] for d in r.datos}
    assert ids == {2, 3}
    assert "Pepsi botella: faltaron 2 unidades" in r.respuesta


def test_faltantes_filtrados_por_producto_en_rango(seed):
    s, _ = seed
    r = responder("¿cuánta coca lata faltó del 1 al 15 de octubre?", s, hoy=HOY)
    assert r.intencion == "diferencia_puntual"
    assert [(d["id_cierre"], d["diferencia"]) for d in r.datos] == [(1, -2), (2, 0), (3, -1)]
    assert "Diferencia neta: -3" in r.respuesta


def test_stock_fanta_lata(seed):
    s, info = seed
    r = responder("¿Cuánta Fanta lata hay?", s, hoy=HOY)
    assert r.intencion == "stock_producto" and r.parametros["productos"] == ["Fanta lata"]
    assert r.datos[0]["stock"] == info["stock_final"]["fanta_lata"] == 20 - 15 + 1
    assert f"Fanta lata: {info['stock_final']['fanta_lata']} unidades" in r.respuesta


def test_stock_en_fecha_pasada_y_todas_las_latas(seed):
    s, _ = seed
    r = responder("¿cuántas latas había el 4 de octubre?", s, hoy=HOY)
    assert r.parametros["fecha_stock"] == "2026-10-04"
    assert len(r.datos) == 5 and all(d["stock"] == 16 for d in r.datos)


def test_detalle_cierre_3(seed):
    s, _ = seed
    r = responder("¿Qué pasó en el cierre 3?", s, hoy=HOY)
    assert r.intencion == "detalle_cierre" and r.parametros["id_cierre"] == 3
    assert "Cierre #3" in r.respuesta and "7Up lata -3 (faltante)" in r.respuesta
    assert "2 objetos no reconocidos" in r.respuesta and "Luz baja" in r.respuesta


def test_ultimo_cierre_y_cierre_por_fecha(seed):
    s, _ = seed
    assert responder("mostrame el último cierre", s, hoy=HOY).parametros["id_cierre"] == 3
    assert responder("¿cómo salió el cierre del 12/10?", s, hoy=HOY).parametros["id_cierre"] == 2


# ------------------------------------------------------------------ sin datos
@pytest.mark.parametrize("pregunta, intencion", [
    ("¿qué faltó ayer?", "faltantes_rango"),          # no hubo cierre el 14/10
    ("¿qué pasó en el cierre 99?", "detalle_cierre"),
    ("¿cuánta pepsi lata faltó el 1 de octubre?", "diferencia_puntual"),
    ("¿cuánta coca había el 12 de octubre de 2025?", "stock_producto"),   # antes de todo movimiento
])
def test_sin_datos_no_llama_al_llm(seed, pregunta, intencion):
    s, _ = seed
    cliente = _cliente_mock()
    r = responder(pregunta, s, cliente=cliente, hoy=HOY)
    assert r.intencion == intencion and r.sin_datos and r.datos == [] and r.modelo is None
    assert "No hay datos" in r.respuesta
    cliente.messages.create.assert_not_called()


def test_bd_vacia():
    s, _ = crear_sesion_seed(con_datos=False)
    for p in ("¿cuánta fanta lata hay?", "¿qué faltó hoy?", "último cierre"):
        assert responder(p, s, hoy=HOY).sin_datos
    s.close()


def test_pregunta_desconocida(seed):
    s, _ = seed
    cliente = _cliente_mock()
    r = responder("hola, ¿cómo estás?", s, cliente=cliente, hoy=HOY)
    assert r.intencion == "desconocida" and r.sin_datos and "No entendí" in r.respuesta
    cliente.messages.create.assert_not_called()


# ------------------------------------------------------------------ LLM mockeado
def test_cliente_recibe_contexto_y_modelo(seed, monkeypatch):
    s, _ = seed
    monkeypatch.setattr(config, "MODELO_CLAUDE", "claude-haiku-5-5")
    cliente = _cliente_mock("Hoy faltaron 4 unidades.")
    r = responder("¿Qué faltó hoy?", s, cliente=cliente, hoy=HOY)

    assert r.respuesta == "Hoy faltaron 4 unidades." and r.modelo == "claude-haiku-5-5"
    cliente.messages.create.assert_called_once()
    kw = cliente.messages.create.call_args.kwargs
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["max_tokens"] == MAX_TOKENS <= 400
    assert kw["system"] == SYSTEM_PROMPT
    contenido = kw["messages"][0]["content"]
    assert kw["messages"][0]["role"] == "user" and "¿Qué faltó hoy?" in contenido
    ctx = json.loads(contenido.split("Contexto (JSON, única fuente de verdad):\n", 1)[1])
    assert ctx["intencion"] == "faltantes_rango"
    assert ctx["resumen"]["unidades_faltantes"] == 4
    assert ctx["registros"] == r.datos


def test_otro_modelo_por_config(seed, monkeypatch):
    s, _ = seed
    monkeypatch.setattr(config, "MODELO_CLAUDE", "claude-sonnet-5-5")
    cliente = _cliente_mock()
    r = responder("¿qué pasó en el cierre 1?", s, cliente=cliente, hoy=HOY)
    kw = cliente.messages.create.call_args.kwargs
    assert kw["model"] == r.modelo == "claude-sonnet-5-5"
    assert "thinking" not in kw     # 'disabled' solo se manda a Haiku


def test_si_la_api_falla_usa_plantilla(seed):
    s, _ = seed
    cliente = MagicMock()
    cliente.messages.create.side_effect = RuntimeError("sin red")
    r = responder("¿Qué faltó hoy?", s, cliente=cliente, hoy=HOY)
    assert r.modelo is None and not r.sin_datos and "7Up lata" in r.respuesta


def test_rechazo_o_texto_vacio_usa_plantilla(seed):
    s, _ = seed
    cliente = MagicMock()
    cliente.messages.create.return_value = SimpleNamespace(content=[], stop_reason="refusal")
    r = responder("¿qué pasó en el cierre 2?", s, cliente=cliente, hoy=HOY)
    assert r.modelo is None and "Cierre #2" in r.respuesta


def test_sin_clave_no_crea_cliente(seed, monkeypatch):
    s, _ = seed
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", MagicMock(side_effect=AssertionError("no")))
    r = responder("¿cuánta fanta lata hay?", s, hoy=HOY)
    assert r.modelo is None and not r.sin_datos


def test_hoy_por_defecto_es_date_today(seed):
    s, _ = seed
    r = responder("¿cuánta fanta lata hay?", s)
    assert r.parametros["fecha_stock"] == date.today().isoformat()
