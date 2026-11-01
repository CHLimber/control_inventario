"""Pipeline real (src/vision/pipeline.py) con detector y clasificador falsos (sin TF ni YOLO)."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from contratos.vision_types import Caja
from src.api.pipeline import cargar_variante
from src.vision import pipeline


class DetectorFalso:
    def __init__(self, cajas):
        self.cajas = cajas

    def detectar(self, imagen, indice_imagen=0):
        assert isinstance(imagen, np.ndarray) and imagen.shape[2] == 3
        return [Caja(c.x, c.y, c.w, c.h, c.conf_det, c.clase, None, indice_imagen) for c in self.cajas]


class ClasificadorFalso:
    def __init__(self):
        self.recortes = []

    def clasificar(self, recortes):
        self.recortes.extend(recortes)
        return [("coca_lata", 0.91), ("desconocido", 0.32)][:len(recortes)]


@pytest.fixture()
def foto(tmp_path: Path) -> Path:
    ruta = tmp_path / "foto1.jpg"
    Image.new("RGB", (320, 240), "gray").save(ruta)
    return ruta


def test_variantes_reales_no_caen_al_mock():
    for variante in ("A", "B"):
        contar = cargar_variante(variante)
        assert isinstance(contar, partial) and contar.func is pipeline.contar
        assert contar.keywords == {"variante": variante}


def test_variante_a_clasifica_cada_recorte(foto, monkeypatch):
    cajas = [Caja(10, 20, 60, 100, 0.9, "producto"), Caja(100, 20, 40, 60, 0.8, "producto")]
    clf = ClasificadorFalso()
    monkeypatch.setattr(pipeline, "detector", lambda v: DetectorFalso(cajas))
    monkeypatch.setattr(pipeline, "clasificador", lambda: clf)

    r = pipeline.contar([foto, foto], variante="A")
    assert r.variante == "A"
    assert r.conteo["coca_lata"] == 1 and r.conteo["desconocido"] == 1
    assert [c.clase for c in r.cajas[:2]] == ["coca_lata", "desconocido"]
    assert r.cajas[0].conf_cls == pytest.approx(0.91)
    assert [c.indice_imagen for c in r.cajas] == [0, 0, 1, 1]
    assert clf.recortes[0].shape[:2] == (110, 66)        # 5 % de margen: 60x100 → 66x110
    assert r.imagen_anotada_path == foto.with_name("foto1_anotada.jpg") and r.imagen_anotada_path.exists()


def test_variante_b_usa_la_clase_del_detector(foto, monkeypatch):
    cajas = [Caja(10, 20, 60, 100, 0.9, "pepsi_lata"), Caja(100, 20, 40, 60, 0.8, "pepsi_lata")]
    monkeypatch.setattr(pipeline, "detector", lambda v: DetectorFalso(cajas))
    monkeypatch.setattr(pipeline, "clasificador", lambda: pytest.fail("la variante B no clasifica"))
    r = pipeline.contar([foto], variante="B")
    assert r.conteo["pepsi_lata"] == 2 and sum(r.conteo.values()) == 2
    assert all(c.conf_cls is None for c in r.cajas)


def test_errores(foto):
    with pytest.raises(ValueError):
        pipeline.contar([], variante="A")
    with pytest.raises(ValueError):
        pipeline.detector("C")
