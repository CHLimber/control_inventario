"""Tests del detector YOLO (src/vision/detector.py).

No necesitan el detector entrenado: usan yolo26n.pt (COCO) como reemplazo, solo para
verificar la FORMA de las Cajas (contrato). Se saltan si no hay pesos o falta ultralytics.
"""
from pathlib import Path

import numpy as np
import pytest

from contratos.vision_types import Caja
from src.config import DIR_MODELOS
from src.vision.detector import PESOS_1CLASE, Detector

ultralytics = pytest.importorskip("ultralytics")
from ultralytics.utils import ASSETS  # noqa: E402

PESOS_COCO = DIR_MODELOS / "yolo26n.pt"
BUS = Path(ASSETS) / "bus.jpg"

requiere_coco = pytest.mark.skipif(not PESOS_COCO.exists() or not BUS.exists(),
                                   reason=f"faltan {PESOS_COCO} o {BUS}")


@pytest.fixture(scope="module")
def detector():
    return Detector(pesos=PESOS_COCO, conf=0.25)


def _verificar_forma(cajas, ancho, alto, indice=0):
    assert isinstance(cajas, list) and len(cajas) > 0
    for c in cajas:
        assert isinstance(c, Caja)
        assert 0 <= c.conf_det <= 1
        assert c.w > 0 and c.h > 0
        # píxeles de la imagen original (no normalizados), esquina superior izquierda
        assert -1 <= c.x and c.x + c.w <= ancho + 1
        assert -1 <= c.y and c.y + c.h <= alto + 1
        assert isinstance(c.clase, str) and c.clase
        assert c.conf_cls is None
        assert c.indice_imagen == indice


@requiere_coco
def test_detectar_desde_ruta(detector):
    from PIL import Image

    ancho, alto = Image.open(BUS).size
    cajas = detector.detectar(BUS)
    _verificar_forma(cajas, ancho, alto)
    assert max(c.w for c in cajas) > 2  # en píxeles, no en [0, 1]
    assert "person" in {c.clase for c in cajas}  # COCO multiclase: devuelve el nombre de la clase


@requiere_coco
def test_detectar_desde_rgb_igual_que_desde_ruta(detector):
    from PIL import Image

    rgb = np.array(Image.open(BUS).convert("RGB"))
    cajas_arr = detector.detectar(rgb, indice_imagen=2)
    _verificar_forma(cajas_arr, rgb.shape[1], rgb.shape[0], indice=2)
    cajas_ruta = detector.detectar(BUS)
    assert len(cajas_arr) == len(cajas_ruta)  # RGB->BGR bien convertido


@requiere_coco
def test_modelo_cacheado(detector):
    detector.detectar(BUS)
    m = detector._modelo
    detector.detectar(BUS)
    assert detector._modelo is m


def test_carga_diferida_y_error_claro(tmp_path):
    d = Detector(pesos=tmp_path / "no_existe.pt")
    assert d._modelo is None  # crear la instancia no carga nada
    with pytest.raises(FileNotFoundError, match="detector_entrenar"):
        d.detectar(np.zeros((64, 64, 3), np.uint8))


def test_pesos_por_defecto_variante_a():
    assert Detector().pesos == PESOS_1CLASE
