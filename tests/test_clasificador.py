"""Tests del clasificador (variante A): preprocesamiento, entrenamiento mini e inferencia.

Todo corre sobre un dataset SINTÉTICO en tmp_path (scripts/clasificador_sintetico.py) con un
modelo "cnn" entrenado 1 época: se prueba la mecánica, no la calidad.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

import clasificador_entrenar as ent  # noqa: E402
import clasificador_sintetico as sint  # noqa: E402

from src.config import CLASE_DESCONOCIDO, nombres_clases  # noqa: E402
from src.vision import clasificador as clf_mod  # noqa: E402
from src.vision.clasificador import Clasificador, ErrorClasesModelo, verificar_clases  # noqa: E402
from src.vision.clasificador_modelos import COLOR_RELLENO, cargar_imagen, letterbox  # noqa: E402


# ----------------------------------------------------------------------------- letterbox
def test_letterbox_botella_alta_no_se_deforma():
    botella = np.zeros((200, 50, 3), dtype=np.uint8)
    botella[..., 0] = 255                                   # roja
    out = letterbox(botella, 128)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    # el lado mayor ocupa todo el alto; el ancho queda en 128*50/200 = 32 px centrados
    fila = out[64]
    rojos = np.where((fila[:, 0] > 200) & (fila[:, 1] < 50))[0]
    assert 30 <= len(rojos) <= 34
    assert abs((rojos.min() + rojos.max()) / 2 - 63.5) <= 1.5
    assert (out[:, :40] == COLOR_RELLENO).all() and (out[:, -40:] == COLOR_RELLENO).all()


def test_letterbox_imagen_ancha_y_formatos_raros():
    ancha = np.full((30, 90, 3), 10, dtype=np.uint8)
    out = letterbox(ancha, 96)
    assert out.shape == (96, 96, 3)
    assert (out[:20] == COLOR_RELLENO).all() and (out[48, :, 0] == 10).all()
    assert letterbox(np.full((40, 20), 200, np.uint8), 64).shape == (64, 64, 3)           # gris
    assert letterbox(np.zeros((40, 20, 4), np.uint8), 64).shape == (64, 64, 3)            # RGBA
    assert letterbox(np.ones((40, 20, 3), np.float32), 64)[32, 32, 0] == 255             # float 0-1
    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 10, 3), np.uint8), 64)


def test_mismo_preprocesamiento_en_entrenamiento_e_inferencia(tmp_path):
    datos = sint.generar(tmp_path / "recortes", por_clase=2)
    rutas, y = ent.listar_split(datos, "train")
    ds = ent.crear_dataset(rutas[:3], y[:3], 128)
    x_entrenamiento = next(iter(ds))[0].numpy()
    x_inferencia = np.stack([letterbox(cargar_imagen(r), 128) for r in rutas[:3]]).astype(np.float32)
    np.testing.assert_array_equal(x_entrenamiento, x_inferencia)


# ----------------------------------------------------------------------------- utilidades
def test_pesos_clase_compensan_desbalance():
    y = np.array([0] * 30 + [1] * 10)
    w = ent.pesos_clase(y, 3)
    assert w[1] == pytest.approx(3 * w[0]) and w[2] == 1.0


def test_confusiones_clave_y_umbral_sugerido():
    clases = nombres_clases()
    m = np.zeros((9, 9), dtype=int)
    i = {c: k for k, c in enumerate(clases)}
    m[i["coca_botella"], i["coca_lata"]] = 3
    m[i["sprite_lata"], i["7up_lata"]] = 2
    m[i["coca_botella"], i["coca_botella"]] = 7
    c = ent.confusiones_clave(m, clases)
    assert c["total_botella_lata"] == 3 and c["total_sprite_7up"] == 2
    barrido = [{"umbral": t, "cobertura": 1 - t / 2, "accuracy_aceptadas": 0.8 + t / 5} for t in ent.UMBRALES]
    assert ent.sugerir_umbral(barrido) == 0.75        # primer umbral ≥ 0.5 con accuracy ≥ 0.95


def test_aumento_sin_volteos():
    from src.vision.clasificador_modelos import capas_aumento

    tipos = [type(c).__name__ for c in capas_aumento().layers]
    assert "RandomFlip" not in tipos
    assert {"RandomRotation", "RandomZoom", "RandomBrightness", "RandomContrast", "RandomTranslation"} <= set(tipos)


def test_mobilenet_finetune_no_descongela_batchnorm():
    from keras import layers

    from src.vision import clasificador_modelos as cm

    modelo = cm.construir_mobilenet(9, tamano=96, pesos=None)
    base = modelo.get_layer(cm.NOMBRE_BASE)
    assert not base.trainable and modelo.output_shape == (None, 9)
    cm.habilitar_finetune(modelo, 20, 1e-5)
    entrenables = [c for c in base.layers if c.trainable]
    assert entrenables and all(not isinstance(c, layers.BatchNormalization) for c in entrenables)
    assert all(not c.trainable for c in base.layers[:-20])


# ----------------------------------------------------------------------------- modelo mini
@pytest.fixture(scope="module")
def modelo_mini(tmp_path_factory):
    """Entrena 'cnn' 1 época con 4 recortes sintéticos por clase y devuelve la carpeta de salida."""
    base = tmp_path_factory.mktemp("clasificador")
    datos = sint.generar(base / "recortes", por_clase=6)
    salida = base / "modelos"
    ent.main(["--modelo", "cnn", "--epocas", "1", "--max-por-clase", "4", "--tamano", "64",
              "--datos", str(datos), "--salida", str(salida), "--logs", str(base / "logs")])
    clf_mod.limpiar_cache()
    return salida


def test_entrenamiento_guarda_artefactos(modelo_mini):
    for nombre in ("clasificador.keras", "clases.json", "metricas.json", "comparacion.csv", "cnn_metricas.json",
                   "cnn_historial.csv", "matriz_cnn_val.png"):
        assert (modelo_mini / nombre).exists(), nombre
    met = json.loads((modelo_mini / "metricas.json").read_text(encoding="utf-8"))
    assert met["tamano_entrada"] == 64 and met["modelo"] == "cnn" and 0.5 <= met["umbral_sugerido"] <= 0.95
    assert met["clases"] == nombres_clases()
    assert any((modelo_mini.parent / "logs").rglob("*.v2"))           # eventos de TensorBoard


def test_clasificar_forma_de_la_salida(modelo_mini):
    rng = np.random.default_rng(0)
    recortes = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((120, 40), (60, 60), (30, 90))]
    clf = Clasificador(modelo_mini / "clasificador.keras", umbral=0.0)
    res = clf.clasificar(recortes)
    assert len(res) == 3
    for clase, conf in res:
        assert clase in nombres_clases() and isinstance(conf, float) and 0.0 <= conf <= 1.0
    assert clf.clasificar([]) == []
    assert clf.tamano == 64
    probs = clf.probabilidades(recortes)
    assert probs.shape == (3, 9) and np.allclose(probs.sum(axis=1), 1, atol=1e-4)


def test_umbral_manda_a_desconocido(modelo_mini):
    recorte = [np.full((80, 40, 3), 127, np.uint8)]
    alto = Clasificador(modelo_mini / "clasificador.keras", umbral=1.01).clasificar(recorte)
    assert alto[0][0] == CLASE_DESCONOCIDO and 0 < alto[0][1] <= 1
    bajo = Clasificador(modelo_mini / "clasificador.keras", umbral=0.0).clasificar(recorte)
    assert bajo[0][0] in nombres_clases() and bajo[0][1] == pytest.approx(alto[0][1])


def test_orden_de_clases_distinto_da_error(modelo_mini, tmp_path):
    otra = tmp_path / "otro"
    otra.mkdir()
    shutil.copy2(modelo_mini / "clasificador.keras", otra / "clasificador.keras")
    clases = nombres_clases()
    (otra / "clases.json").write_text(json.dumps([clases[1], clases[0], *clases[2:]]), encoding="utf-8")
    with pytest.raises(ErrorClasesModelo, match="orden de clases"):
        Clasificador(otra / "clasificador.keras").clasificar([np.zeros((10, 10, 3), np.uint8)])
    (otra / "clases.json").unlink()
    clf_mod.limpiar_cache()
    with pytest.raises(ErrorClasesModelo, match="Falta"):
        Clasificador(otra / "clasificador.keras").clasificar([np.zeros((10, 10, 3), np.uint8)])
    with pytest.raises(ErrorClasesModelo, match="salidas"):
        verificar_clases(clases, 8)


def test_modelo_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="clasificador_entrenar"):
        Clasificador(tmp_path / "no.keras").clasificar([np.zeros((5, 5, 3), np.uint8)])
