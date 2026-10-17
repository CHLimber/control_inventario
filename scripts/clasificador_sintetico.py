"""Mini dataset SINTÉTICO de recortes para probar el clasificador sin los datos reales.

Dibuja envases de juguete (botella alta / lata baja, color y franja por marca) con la misma
estructura que produce el agente Datos:

    <destino>/{train,val,test}/<clase>/*.jpg   +   <destino>/indice_recortes.csv

Sirve para los tests y para medir tiempos por época cuando todavía no hay recortes reales.
NO sirve para medir la calidad del modelo (es un problema mucho más fácil que el real).

Uso:
  python scripts/clasificador_sintetico.py <destino> [--por-clase 20]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import nombres_clases  # noqa: E402

# color del cuerpo y de la franja/etiqueta por marca
COLORES = {
    "coca": ((200, 20, 30), (245, 245, 245)),
    "fanta": ((250, 130, 10), (30, 60, 160)),
    "sprite": ((20, 150, 70), (240, 220, 40)),
    "pepsi": ((20, 50, 160), (220, 30, 40)),
}
SPLITS = {"train": 1.0, "val": 0.4, "test": 0.4}


def dibujar(clase: str, rng: random.Random) -> Image.Image:
    marca, formato = clase.split("_")
    cuerpo, franja = COLORES[marca]
    jitter = lambda c: tuple(max(0, min(255, v + rng.randint(-20, 20))) for v in c)  # noqa: E731
    if formato == "botella":
        ancho, alto = rng.randint(40, 60), rng.randint(130, 170)
    else:
        ancho, alto = rng.randint(55, 75), rng.randint(90, 115)
    margen = rng.randint(4, 12)
    img = Image.new("RGB", (ancho + 2 * margen, alto + 2 * margen), jitter((200, 200, 195)))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = margen, margen, margen + ancho, margen + alto
    if formato == "botella":
        cuello = ancho // 3
        d.rectangle([x0 + cuello, y0, x1 - cuello, y0 + alto // 5], fill=jitter(cuerpo))
        d.rectangle([x0, y0 + alto // 5, x1, y1], fill=jitter(cuerpo))
        d.rectangle([x0 + cuello, y0, x1 - cuello, y0 + 6], fill=(30, 30, 30))      # tapa
        d.rectangle([x0, y0 + alto // 2, x1, y0 + alto // 2 + alto // 6], fill=jitter(franja))
    else:
        d.rectangle([x0, y0, x1, y1], fill=jitter(cuerpo))
        d.rectangle([x0, y0, x1, y0 + 5], fill=(180, 180, 185))                     # tapa metálica
        d.ellipse([x0 + ancho // 4, y0 + alto // 3, x1 - ancho // 4, y0 + 2 * alto // 3], fill=jitter(franja))
    return img


def generar(destino: Path, por_clase: int = 20, semilla: int = 0) -> Path:
    """Crea el dataset y devuelve la carpeta. `por_clase` es para train; val/test llevan menos."""
    rng = random.Random(semilla)
    destino = Path(destino)
    filas = []
    for split, factor in SPLITS.items():
        n = max(2, round(por_clase * factor))
        for clase in nombres_clases():
            carpeta = destino / split / clase
            carpeta.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                ruta = carpeta / f"sint_{clase}_{k:04d}.jpg"
                dibujar(clase, rng).save(ruta, quality=90)
                filas.append({"ruta": ruta.relative_to(destino).as_posix(), "split": split, "clase": clase})
    with (destino / "indice_recortes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ruta", "split", "clase"])
        w.writeheader()
        w.writerows(filas)
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("destino", type=Path)
    ap.add_argument("--por-clase", type=int, default=20)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    print("Dataset sintético en", generar(a.destino, a.por_clase, a.semilla))


if __name__ == "__main__":
    main()
