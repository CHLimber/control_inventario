"""Verifica que los contratos de la Fase 0 importan y son coherentes entre sí."""
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine, inspect

from contratos import api_schemas as api
from contratos.modelos_bd import Base
from contratos.vision_types import Caja, ResultadoConteo, combinar_conteos, contar_cajas
from src.config import clases_con_desconocido, nombres_clases
from src.vision import pipeline_mock


def test_clases_son_9_mas_desconocido():
    clases = nombres_clases()
    assert len(clases) == 8 and len(set(clases)) == 8
    assert clases_con_desconocido()[-1] == "desconocido"


def test_esquema_bd_crea_las_5_tablas():
    motor = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(motor)
    assert set(inspect(motor).get_table_names()) == {
        "producto", "movimiento", "cierre", "cierre_detalle", "deteccion"}


def test_mediana_redondea_hacia_arriba():
    assert combinar_conteos([{"a": 2}, {"a": 3}]) == {"a": 3}
    assert combinar_conteos([{"a": 1}, {"a": 5}, {"a": 2}]) == {"a": 2}


def test_contar_cajas_manda_clases_raras_a_desconocido():
    cajas = [Caja(0, 0, 1, 1, 0.9, "coca_lata", 0.9), Caja(0, 0, 1, 1, 0.9, "xyz", 0.1)]
    conteo = contar_cajas(cajas, clases_con_desconocido())
    assert conteo["coca_lata"] == 1 and conteo["desconocido"] == 1


@pytest.mark.parametrize("n_fotos", [1, 3])
def test_mock_devuelve_resultado_valido(tmp_path: Path, n_fotos: int):
    fotos = []
    for i in range(n_fotos):
        ruta = tmp_path / f"foto{i}.jpg"
        Image.new("RGB", (640, 480), "gray").save(ruta)
        fotos.append(ruta)
    r = pipeline_mock.contar(fotos, semilla=1)
    assert isinstance(r, ResultadoConteo)
    assert list(r.conteo) == clases_con_desconocido()
    assert all(isinstance(v, int) and v >= 0 for v in r.conteo.values())
    assert r.imagen_anotada_path.exists()
    assert {c.indice_imagen for c in r.cajas} == set(range(n_fotos))
    if n_fotos == 1:
        assert r.conteo == contar_cajas(r.cajas, clases_con_desconocido())


def test_esquemas_api_construyen_un_cierre():
    fila = api.FilaConciliacion(id_producto=1, clase_modelo="coca_lata", nombre="Coca-Cola lata",
                                esperado=5, detectado=4, diferencia=-1, estado="faltante")
    totales = api.Totales(esperado=5, detectado=4, diferencia=-1, productos_con_diferencia=1)
    cierre = api.CierreOut(id=1, fecha="2026-10-15T21:00:00", usuario="cajero", estado="pendiente",
                           variante_modelo="mock", totales=totales, filas=[fila])
    assert cierre.model_dump(mode="json")["filas"][0]["diferencia"] == -1
    assert len(api.ENDPOINTS) == len({(m, r) for m, r, _ in api.ENDPOINTS})
