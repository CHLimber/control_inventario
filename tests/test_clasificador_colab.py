"""Tests de scripts/clasificador_empaquetar_colab.py (ida y vuelta con Colab, todo en tmp_path)."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

import clasificador_empaquetar_colab as colab  # noqa: E402
import clasificador_sintetico as sint  # noqa: E402


def test_empaquetar_lleva_codigo_y_recortes(tmp_path):
    datos = sint.generar(tmp_path / "recortes", por_clase=2)
    zip_ = colab.empaquetar(tmp_path / "c.zip", datos)
    nombres = set(zipfile.ZipFile(zip_).namelist())
    for rel in colab.CODIGO:
        assert f"control_inventario/{rel}" in nombres
    assert "control_inventario/datos/recortes/indice_recortes.csv" in nombres
    n_jpg = sum(1 for n in nombres if n.endswith(".jpg"))
    assert n_jpg == len(list(datos.rglob("*.jpg")))


def test_importar_ubica_modelos_y_logs_e_ignora_lo_demas(tmp_path, monkeypatch):
    monkeypatch.setattr(colab.config, "DIR_MODELOS", tmp_path / "modelos")
    monkeypatch.setattr(colab.config, "DIR_LOGS", tmp_path / "logs")
    res = tmp_path / "resultados.zip"
    with zipfile.ZipFile(res, "w") as z:
        z.writestr("modelos/clasificador/clasificador.keras", b"x")
        z.writestr("logs/clasificador/mobilenet_1/train/events.out", b"y")
        z.writestr("modelos/clasificador/../../malo.txt", b"z")
        z.writestr("src/config.py", b"w")
    colab.importar(res)
    assert (tmp_path / "modelos/clasificador/clasificador.keras").read_bytes() == b"x"
    assert (tmp_path / "logs/clasificador/mobilenet_1/train/events.out").exists()
    assert not (tmp_path / "malo.txt").exists() and not (tmp_path / "src").exists()
