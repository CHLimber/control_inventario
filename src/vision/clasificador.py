"""Clasificador de envases (variante A): recorte RGB → (clase, confianza).

Cumple el contrato `clasificar(recortes) -> list[tuple[str, float]]` de
contratos/vision_types.py. Uso:

    from src.vision.clasificador import Clasificador
    clf = Clasificador()                       # modelos/clasificador/clasificador.keras
    clf.clasificar([recorte1, recorte2])       # [("coca_lata", 0.97), ("desconocido", 0.41)]

- Los recortes son arreglos numpy RGB uint8 (H, W, 3) de cualquier tamaño; se les aplica el
  MISMO letterbox que en el entrenamiento (src/vision/clasificador_modelos.py).
- Se clasifica en lote (una sola pasada por la red para todos los recortes de una foto).
- Si la confianza (probabilidad máxima del softmax) es menor que `umbral`, se devuelve
  ("desconocido", conf): mejor admitir la duda que sumar una lata al producto equivocado.
- El modelo se carga recién en la primera llamada (carga diferida) y queda en caché por ruta,
  así varias instancias o varias peticiones de la API no lo vuelven a leer del disco.
- Al cargar se verifica que el orden de clases con el que se entrenó (clases.json junto al
  modelo) sea el de config/clases.json: si no, el índice 3 del softmax significaría otra cosa.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import CLASE_DESCONOCIDO, DIR_MODELOS, UMBRAL_CONF, nombres_clases
from src.vision.clasificador_modelos import preprocesar_lote

DIR_CLASIFICADOR = DIR_MODELOS / "clasificador"
RUTA_POR_DEFECTO = DIR_CLASIFICADOR / "clasificador.keras"
LOTE_INFERENCIA = 64

_cache: dict[tuple[str, float], tuple[object, int]] = {}
_candado = threading.Lock()


class ErrorClasesModelo(RuntimeError):
    """El modelo guardado no corresponde a las clases de config/clases.json."""


def leer_clases_modelo(ruta_modelo: Path) -> list[str]:
    """Clases (en orden de salida) guardadas en clases.json junto al modelo.

    Acepta la copia completa de config/clases.json ({"clases": [{"clase": ...}, ...]}) o una
    lista simple de nombres.
    """
    ruta = Path(ruta_modelo).with_name("clases.json")
    if not ruta.exists():
        raise ErrorClasesModelo(
            f"Falta {ruta}: el modelo debe guardarse junto a una copia de config/clases.json "
            "para poder verificar el orden de sus salidas.")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if isinstance(datos, dict):
        datos = datos["clases"]
    return [c["clase"] if isinstance(c, dict) else str(c) for c in datos]


def verificar_clases(clases_modelo: Sequence[str], n_salidas: int, esperadas: Sequence[str] | None = None) -> None:
    """Error claro si el orden/cantidad de clases del modelo no coincide con config/clases.json."""
    esperadas = list(esperadas if esperadas is not None else nombres_clases())
    if list(clases_modelo) != esperadas:
        raise ErrorClasesModelo(
            "El orden de clases del modelo no coincide con config/clases.json.\n"
            f"  modelo: {list(clases_modelo)}\n  config: {esperadas}\n"
            "Reentrená el clasificador con las clases actuales (no se reordenan a mano).")
    if n_salidas != len(esperadas):
        raise ErrorClasesModelo(
            f"El modelo tiene {n_salidas} salidas pero config/clases.json define {len(esperadas)} clases.")


def _cargar(ruta: Path):
    """Carga (una sola vez por ruta y fecha de modificación) el .keras y su tamaño de entrada."""
    clave = (str(ruta.resolve()), ruta.stat().st_mtime)
    with _candado:
        if clave not in _cache:
            import keras

            modelo = keras.models.load_model(ruta, compile=False)
            n_salidas = int(modelo.outputs[0].shape[-1])
            verificar_clases(leer_clases_modelo(ruta), n_salidas)
            tamano = int(modelo.inputs[0].shape[1])
            _cache[clave] = (modelo, tamano)
        return _cache[clave]


def limpiar_cache() -> None:
    """Olvida los modelos cargados (útil en tests o tras reemplazar el .keras)."""
    with _candado:
        _cache.clear()


class Clasificador:
    """Implementación real del protocolo `Clasificador` de contratos/vision_types.py."""

    def __init__(self, ruta: str | Path | None = None, umbral: float = UMBRAL_CONF):
        self.ruta = Path(ruta) if ruta is not None else RUTA_POR_DEFECTO
        self.umbral = float(umbral)
        self.clases = nombres_clases()

    # carga diferida -----------------------------------------------------------
    def _modelo(self):
        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No existe el modelo del clasificador: {self.ruta}\n"
                "Entrenalo con scripts/clasificador_entrenar.py (o en Colab, ver docs/clasificador.md).")
        return _cargar(self.ruta)

    @property
    def tamano(self) -> int:
        """Lado de la entrada cuadrada del modelo (lo dice el propio .keras)."""
        return self._modelo()[1]

    # API ----------------------------------------------------------------------
    def probabilidades(self, recortes: Sequence[np.ndarray]) -> np.ndarray:
        """Matriz (n, 9) con el softmax de cada recorte (en el orden de config/clases.json)."""
        if len(recortes) == 0:
            return np.zeros((0, len(self.clases)), dtype=np.float32)
        modelo, tamano = self._modelo()
        x = preprocesar_lote(list(recortes), tamano).astype(np.float32)
        partes = [np.asarray(modelo(x[i:i + LOTE_INFERENCIA], training=False))
                  for i in range(0, len(x), LOTE_INFERENCIA)]
        return np.concatenate(partes, axis=0)

    def clasificar(self, recortes: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        """(clase, confianza) por recorte; "desconocido" si la confianza < umbral."""
        probs = self.probabilidades(recortes)
        resultado: list[tuple[str, float]] = []
        for fila in probs:
            i = int(np.argmax(fila))
            conf = float(fila[i])
            resultado.append((self.clases[i] if conf >= self.umbral else CLASE_DESCONOCIDO, conf))
        return resultado


def clasificar(recortes: Sequence[np.ndarray]) -> list[tuple[str, float]]:
    """Atajo funcional con el modelo y el umbral por defecto."""
    return Clasificador().clasificar(recortes)
