"""Pipeline real de visión: contar(imagenes, variante) -> ResultadoConteo (contratos/vision_types.py).

    Variante A: detector de 1 clase (modelos/detector/detector_1clase.pt) encuentra cada envase y el
                clasificador CNN (modelos/clasificador/clasificador.keras) decide la clase del recorte.
                Si la confianza del clasificador < UMBRAL_CONF, la caja queda como "desconocido".
    Variante B: detector multiclase (modelos/detector/detector_multiclase.pt): ya trae la clase.

Con varias fotos, el conteo es la mediana por clase (combinar_conteos). La imagen anotada se guarda
junto a la primera foto como "<nombre>_anotada.jpg". Los modelos se cargan una sola vez por proceso
(`precargar` lo hace por adelantado para que el primer escaneo no espere).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from contratos.vision_types import Caja, ResultadoConteo, combinar_conteos, contar_cajas
from src.config import CLASE_DESCONOCIDO, UMBRAL_CONF, UMBRAL_DET, clases_con_desconocido
from src.vision.detector import PESOS_1CLASE, PESOS_MULTICLASE, Detector

PESOS_VARIANTE = {"A": PESOS_1CLASE, "B": PESOS_MULTICLASE}
MARGEN_RECORTE = 0.05   # el mismo margen con el que se generaron los recortes (src/datos/recortes.py)

# Color por marca (RGB); la lata va en un tono más claro que la botella. Igual que en la web.
COLORES = {
    "coca_botella": (215, 25, 28), "coca_lata": (255, 107, 107),
    "fanta_botella": (230, 106, 0), "fanta_lata": (255, 169, 77),
    "sprite_botella": (27, 138, 58), "sprite_lata": (92, 214, 92),
    "pepsi_botella": (28, 79, 184), "pepsi_lata": (92, 157, 255),
    CLASE_DESCONOCIDO: (224, 224, 224),
}

_detectores: dict[str, Detector] = {}
_clasificador = None
_candado = threading.Lock()


# ------------------------------------------------------------------ modelos (en caché)
def detector(variante: str) -> Detector:
    if variante not in PESOS_VARIANTE:
        raise ValueError(f"Variante desconocida {variante!r} (se esperaba A o B)")
    with _candado:
        if variante not in _detectores:
            _detectores[variante] = Detector(PESOS_VARIANTE[variante], conf=UMBRAL_DET)
        return _detectores[variante]


def clasificador():
    global _clasificador
    with _candado:
        if _clasificador is None:
            from src.vision.clasificador import Clasificador

            _clasificador = Clasificador(umbral=UMBRAL_CONF)
        return _clasificador


def precargar(variante: str) -> None:
    """Carga los pesos y hace una pasada en vacío (la primera inferencia de torch/keras es lenta)."""
    vacia = np.zeros((64, 64, 3), dtype=np.uint8)
    detector(variante).detectar(vacia)
    if variante == "A":
        clasificador().clasificar([vacia])


# ------------------------------------------------------------------ utilidades
def leer_rgb(ruta: Path) -> np.ndarray:
    """Foto como RGB uint8, girada según su EXIF (las fotos del celular vienen rotadas)."""
    with Image.open(ruta) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def recortar(imagen: np.ndarray, x: float, y: float, w: float, h: float,
             margen: float = MARGEN_RECORTE) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    mx, my = w * margen, h * margen
    x0, y0 = max(0, int(round(x - mx))), max(0, int(round(y - my)))
    x1, y1 = min(ancho, int(round(x + w + mx))), min(alto, int(round(y + h + my)))
    return imagen[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]


def _fuente(tamano: int):
    for nombre in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(nombre, tamano)
        except OSError:
            continue
    return ImageFont.load_default()


def anotar(imagen: np.ndarray, cajas: list[Caja], salida: Path) -> Path:
    """Dibuja las cajas (color por clase, punteado gris = desconocido) y guarda un JPG."""
    im = Image.fromarray(imagen)
    dibujo = ImageDraw.Draw(im)
    grosor = max(2, im.width // 350)
    fuente = _fuente(max(12, im.width // 60))
    for c in sorted(cajas, key=lambda c: -c.w * c.h):   # las chicas encima
        color = COLORES.get(c.clase, (255, 79, 216))
        dibujo.rectangle([c.x, c.y, c.x + c.w, c.y + c.h], outline=color, width=grosor)
        conf = c.conf_cls if c.conf_cls is not None else c.conf_det
        texto = f"{c.clase} {conf:.2f}"
        x0, y0, x1, y1 = dibujo.textbbox((0, 0), texto, font=fuente)
        ty = c.y - (y1 - y0) - 6 if c.y > (y1 - y0) + 6 else c.y
        dibujo.rectangle([c.x, ty, c.x + (x1 - x0) + 8, ty + (y1 - y0) + 6], fill=color)
        dibujo.text((c.x + 4, ty + 3 - y0), texto, fill=(17, 17, 17), font=fuente)
    im.save(salida, quality=88)
    return salida


# ------------------------------------------------------------------ conteo
def cajas_de_imagen(imagen: np.ndarray, variante: str, indice: int = 0) -> list[Caja]:
    cajas = detector(variante).detectar(imagen, indice_imagen=indice)
    if variante == "A" and cajas:
        lecturas = clasificador().clasificar([recortar(imagen, c.x, c.y, c.w, c.h) for c in cajas])
        for caja, (clase, conf) in zip(cajas, lecturas):
            caja.clase, caja.conf_cls = clase, round(conf, 4)
    return cajas


def contar(imagenes: list[Path], variante: str = "A") -> ResultadoConteo:
    if not imagenes:
        raise ValueError("Se necesita al menos una imagen")
    inicio = time.perf_counter()
    claves = clases_con_desconocido()
    todas: list[Caja] = []
    por_imagen: list[dict[str, int]] = []
    anotada = None
    for i, ruta in enumerate(map(Path, imagenes)):
        imagen = leer_rgb(ruta)
        cajas = cajas_de_imagen(imagen, variante, i)
        todas.extend(cajas)
        por_imagen.append(contar_cajas(cajas, claves))
        if i == 0:
            anotada = anotar(imagen, cajas, ruta.with_name(f"{ruta.stem}_anotada.jpg"))
    return ResultadoConteo(conteo=combinar_conteos(por_imagen), cajas=todas, imagen_anotada_path=anotada,
                           variante=variante, conteos_por_imagen=por_imagen,
                           tiempo_ms=(time.perf_counter() - inicio) * 1000)
