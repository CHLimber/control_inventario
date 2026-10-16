"""Pipeline FALSO que cumple el contrato de contar() (contratos/vision_types.py).

Sirve para que la API y la web avancen antes de tener modelos entrenados.
Genera cajas aleatorias acomodadas en 3 niveles (como la maqueta), cuenta por
clase, combina varias fotos con la mediana y dibuja la imagen anotada.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

from PIL import Image, ImageDraw

from contratos.vision_types import Caja, ResultadoConteo, combinar_conteos, contar_cajas
from src.config import CLASE_DESCONOCIDO, cargar_clases, clases_con_desconocido

VARIANTE = "mock"


def _cajas_falsas(ancho: int, alto: int, indice: int, rng: random.Random) -> list[Caja]:
    clases = cargar_clases()["clases"]
    botellas = [c["clase"] for c in clases if c["formato"] == "botella"]
    latas = [c["clase"] for c in clases if c["formato"] == "lata"]
    # Nivel superior: botellas grandes; medio: botellas; inferior: latas.
    niveles = [(0.05, 0.30, botellas), (0.38, 0.62, botellas), (0.70, 0.92, latas)]
    cajas: list[Caja] = []
    for y0, y1, candidatas in niveles:
        n = rng.randint(3, 8)
        ancho_caja = ancho * 0.9 / max(n, 1)
        for i in range(n):
            if rng.random() < 0.08:
                clase, conf_cls = CLASE_DESCONOCIDO, rng.uniform(0.2, 0.45)
            else:
                clase, conf_cls = rng.choice(candidatas), rng.uniform(0.6, 0.99)
            cajas.append(Caja(
                x=round(ancho * 0.05 + i * ancho_caja, 1), y=round(alto * y0, 1),
                w=round(ancho_caja * 0.85, 1), h=round(alto * (y1 - y0), 1),
                conf_det=round(rng.uniform(0.5, 0.98), 3), clase=clase,
                conf_cls=round(conf_cls, 3), indice_imagen=indice))
    return cajas


def _anotar(imagen: Path, cajas: list[Caja]) -> Path:
    salida = imagen.with_name(f"{imagen.stem}_anotada.jpg")
    with Image.open(imagen) as im:
        im = im.convert("RGB")
        dibujo = ImageDraw.Draw(im)
        grosor = max(2, im.width // 400)
        for c in cajas:
            color = "red" if c.clase == CLASE_DESCONOCIDO else "lime"
            dibujo.rectangle([c.x, c.y, c.x + c.w, c.y + c.h], outline=color, width=grosor)
            dibujo.text((c.x + 3, c.y + 3), f"{c.clase} {c.conf_cls or c.conf_det:.2f}", fill=color)
        im.save(salida, quality=85)
    return salida


def contar(imagenes: list[Path], semilla: int | None = None) -> ResultadoConteo:
    """Conteo falso con la forma real. `semilla` hace el resultado reproducible (tests)."""
    if not imagenes:
        raise ValueError("Se necesita al menos una imagen")
    inicio = time.perf_counter()
    rng = random.Random(semilla)
    claves = clases_con_desconocido()
    todas: list[Caja] = []
    por_imagen: list[dict[str, int]] = []
    for i, ruta in enumerate(map(Path, imagenes)):
        with Image.open(ruta) as im:
            ancho, alto = im.size
        cajas = _cajas_falsas(ancho, alto, i, rng)
        todas.extend(cajas)
        por_imagen.append(contar_cajas(cajas, claves))
    anotada = _anotar(Path(imagenes[0]), [c for c in todas if c.indice_imagen == 0])
    return ResultadoConteo(
        conteo=combinar_conteos(por_imagen), cajas=todas, imagen_anotada_path=anotada,
        variante=VARIANTE, conteos_por_imagen=por_imagen,
        tiempo_ms=(time.perf_counter() - inicio) * 1000)
