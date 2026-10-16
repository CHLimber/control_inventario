"""Detector YOLO entrenado (Fase 2). Cumple la interfaz `detectar(imagen) -> list[Caja]`.

Dos usos con la MISMA clase:
  * Variante A: pesos de 1 clase (`detector_1clase.pt`) -> cada Caja trae clase="producto";
    luego el clasificador CNN decide la marca sobre el recorte.
  * Variante B (línea base): pesos multiclase (`detector_multiclase.pt`) -> cada Caja trae
    el nombre de la clase de config/clases.json y conf_cls=None (no hay clasificador aparte).

Las cajas salen en PÍXELES de la imagen original, con (x, y) = esquina superior izquierda
(contrato de contratos/vision_types.py). El modelo se carga recién en la primera llamada
(carga diferida) y queda guardado en la instancia (cacheado).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from contratos.vision_types import Caja
from src.config import DIR_MODELOS, UMBRAL_DET

DIR_PESOS = DIR_MODELOS / "detector"
PESOS_1CLASE = DIR_PESOS / "detector_1clase.pt"
PESOS_MULTICLASE = DIR_PESOS / "detector_multiclase.pt"
CLASE_PRODUCTO = "producto"


class Detector:
    """Envoltorio de un YOLO de Ultralytics.

    pesos: ruta a un .pt; por defecto DIR_MODELOS/detector/detector_1clase.pt (variante A).
    conf:  umbral de confianza de detección (UMBRAL_DET del .env, 0.25 por defecto).
    imgsz: lado de la imagen que ve la red (la misma con la que se entrenó).
    hilos: si se indica, cantidad de hilos de torch en CPU (Ultralytics fija 8 en la 1.ª llamada).
    """

    def __init__(self, pesos: Path | str | None = None, conf: float = UMBRAL_DET,
                 imgsz: int = 640, hilos: int | None = None, max_det: int = 300):
        self.pesos = Path(pesos) if pesos is not None else PESOS_1CLASE
        self.conf = conf
        self.imgsz = imgsz
        self.hilos = hilos
        self.max_det = max_det
        self._modelo = None  # se carga en la primera llamada a detectar()

    # ------------------------------------------------------------------ carga
    @property
    def modelo(self):
        if self._modelo is None:
            if not self.pesos.exists():
                raise FileNotFoundError(
                    f"No existen los pesos del detector: {self.pesos}\n"
                    "Entrená el detector (scripts/detector_entrenar.py o notebooks/04_detector_colab.ipynb) "
                    f"y copiá best.pt como {PESOS_1CLASE.name} (variante A) o {PESOS_MULTICLASE.name} "
                    f"(variante B) en {DIR_PESOS}.")
            from ultralytics import YOLO  # import diferido: importar ultralytics tarda ~1 s

            self._modelo = YOLO(str(self.pesos))
        return self._modelo

    @property
    def nombres(self) -> dict[int, str]:
        """Nombre de clase por índice de salida. Un modelo de 1 clase devuelve siempre 'producto'."""
        nombres = dict(self.modelo.names)
        if len(nombres) == 1:
            return {0: CLASE_PRODUCTO}
        return nombres

    # -------------------------------------------------------------- inferencia
    def detectar(self, imagen: np.ndarray | Path | str, indice_imagen: int = 0) -> list[Caja]:
        """Detecta productos en UNA imagen.

        imagen: arreglo numpy RGB (alto, ancho, 3) o ruta a un archivo de imagen.
        """
        if isinstance(imagen, np.ndarray):
            if imagen.ndim != 3 or imagen.shape[2] != 3:
                raise ValueError(f"Se espera una imagen RGB (alto, ancho, 3); llegó {imagen.shape}")
            fuente = np.ascontiguousarray(imagen[..., ::-1])  # Ultralytics espera BGR (como OpenCV)
        else:
            fuente = str(imagen)
            if not Path(fuente).exists():
                raise FileNotFoundError(f"No existe la imagen: {fuente}")

        nombres = self.nombres
        res = self.modelo.predict(fuente, conf=self.conf, imgsz=self.imgsz, max_det=self.max_det,
                                  verbose=False)[0]
        if self.hilos:  # después de la 1.ª llamada (select_device ya fijó sus hilos)
            import torch

            torch.set_num_threads(self.hilos)

        cajas: list[Caja] = []
        b = res.boxes
        if b is None or len(b) == 0:
            return cajas
        for (x1, y1, x2, y2), c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                          b.cls.cpu().numpy()):
            cajas.append(Caja(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1),
                              conf_det=float(c), clase=nombres.get(int(k), str(int(k))),
                              conf_cls=None, indice_imagen=indice_imagen))
        return cajas

    def __repr__(self) -> str:
        return f"Detector(pesos={self.pesos}, conf={self.conf}, imgsz={self.imgsz})"
