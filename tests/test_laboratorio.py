"""Laboratorio de modelos: catálogo y rutas, con detector y clasificador falsos (sin TF ni YOLO)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from contratos.vision_types import Caja
from src.api import rutas_laboratorio as lab
from src.api.app import crear_app

CLASES_9 = ["coca_botella", "coca_lata", "fanta_botella", "fanta_lata", "sprite_botella",
            "sprite_lata", "pepsi_botella", "pepsi_lata", "7up_lata"]


def _jpg(ancho: int = 320, alto: int = 240) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (ancho, alto), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


class DetectorFalso:
    def __init__(self, cajas: list[Caja]):
        self.cajas = cajas

    def detectar(self, imagen):
        assert isinstance(imagen, np.ndarray) and imagen.shape[2] == 3
        return list(self.cajas)


@pytest.fixture()
def cliente(tmp_path: Path, monkeypatch):
    raiz = tmp_path
    modelos = raiz / "modelos"
    (modelos / "detector").mkdir(parents=True)
    (modelos / "detector" / "detector_1clase.pt").write_bytes(b"x")
    (modelos / "yolo26n.pt").write_bytes(b"x")
    (raiz / "runs" / "detector" / "multiclase_prueba" / "weights").mkdir(parents=True)
    (raiz / "runs" / "detector" / "multiclase_prueba" / "weights" / "best.pt").write_bytes(b"x")
    viejo = modelos / "clasificador" / "noche"
    viejo.mkdir(parents=True)
    (viejo / "clasificador.keras").write_bytes(b"x")
    (viejo / "clases.json").write_text(json.dumps(CLASES_9), encoding="utf-8")
    (modelos / "clasificador" / "huerfano.keras").write_bytes(b"x")  # sin clases.json: se ignora

    monkeypatch.setattr(lab, "RAIZ", raiz)
    monkeypatch.setattr(lab, "DIR_MODELOS", modelos)
    monkeypatch.setattr(lab, "DIR_RUNS_DETECTOR", raiz / "runs" / "detector")

    cajas = [Caja(10, 20, 60, 100, 0.9, "producto"), Caja(100, 20, 40, 60, 0.3, "bottle"),
             Caja(200, 20, 40, 60, 0.8, "person")]
    monkeypatch.setattr(lab, "_detector", lambda id_modelo: DetectorFalso(cajas))
    recortes_vistos: list = []

    def probabilidades(id_modelo, recortes):
        recortes_vistos.extend(recortes)
        probs = np.zeros((len(recortes), len(CLASES_9)), dtype=np.float32)
        probs[:, 1] = 0.7
        probs[:, 8] = 0.3
        return probs

    monkeypatch.setattr(lab, "_probabilidades", probabilidades)
    app = crear_app(url_bd=f"sqlite:///{(tmp_path / 'lab.db').as_posix()}", dir_escaneos=tmp_path / "esc")
    c = TestClient(app)
    c.recortes_vistos = recortes_vistos
    return c


def test_catalogo_de_modelos(cliente):
    datos = cliente.get("/api/laboratorio/modelos").json()
    tipos = {d["id"]: d["tipo"] for d in datos["detectores"]}
    assert tipos == {"modelos/detector/detector_1clase.pt": "1clase",
                     "runs/detector/multiclase_prueba/weights/best.pt": "multiclase",
                     "modelos/yolo26n.pt": "coco"}
    [cls] = datos["clasificadores"]
    assert cls["id"] == "modelos/clasificador/noche/clasificador.keras"
    assert cls["nombre"] == "noche/clasificador"
    assert cls["clases"] == CLASES_9 and cls["coincide_config"] is False
    assert len(datos["clases_config"]) == 8


def test_analizar_devuelve_cajas_y_probabilidades(cliente):
    r = cliente.post("/api/laboratorio/analizar", files={"imagen": ("e.jpg", _jpg())},
                     data={"detector": "modelos/detector/detector_1clase.pt",
                           "clasificador": "modelos/clasificador/noche/clasificador.keras"})
    assert r.status_code == 200, r.text
    datos = r.json()
    assert (datos["ancho"], datos["alto"]) == (320, 240)
    assert len(datos["cajas"]) == 3
    assert datos["cajas"][0]["probs"][1] == pytest.approx(0.7)
    # recorte con el 5 % de margen: caja de 60x100 → 66x110
    assert cliente.recortes_vistos[0].shape[:2] == (110, 66)


def test_coco_filtra_clases_y_sin_clasificador(cliente):
    r = cliente.post("/api/laboratorio/analizar", files={"imagen": ("e.jpg", _jpg())},
                     data={"detector": "modelos/yolo26n.pt"})
    datos = r.json()
    assert [c["clase_det"] for c in datos["cajas"]] == ["bottle"]
    assert datos["cajas"][0]["probs"] is None and datos["clasificador"] is None


def test_clasificar_un_recorte(cliente):
    r = cliente.post("/api/laboratorio/clasificar", files={"imagen": ("r.jpg", _jpg(60, 120))},
                     data={"clasificador": "modelos/clasificador/noche/clasificador.keras"})
    assert r.status_code == 200
    assert len(r.json()["probs"]) == 9


def test_errores(cliente):
    assert cliente.post("/api/laboratorio/analizar", files={"imagen": ("e.jpg", _jpg())},
                        data={"detector": "../secreto.pt"}).status_code == 404
    assert cliente.post("/api/laboratorio/analizar", files={"imagen": ("e.jpg", b"no es imagen")},
                        data={"detector": "modelos/yolo26n.pt"}).status_code == 422
    assert cliente.get("/laboratorio/").status_code == 200
