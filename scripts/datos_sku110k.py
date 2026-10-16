"""(OPCIONAL, PENDIENTE) Extrae un subconjunto de SKU-110K para el detector de 1 clase.

SKU-110K son ~11 700 fotos de góndolas densas con ~1,7 M cajas de productos (una sola clase).
El archivo pesa 13,6 GB (el mismo que usa SKU-110K.yaml de Ultralytics). Este script lo lee
EN STREAMING y guarda solo un subconjunto determinista (por hash del nombre), así en disco
queda únicamente lo elegido. Igual hay que bajar los 13,6 GB, por eso exige --confirmar.

Salida: DIR_DATOS/raw/sku110k_sub/{train,valid,test}/{images,labels} + data.yaml (clase 'producto').
Después: datos_indexar.py y datos_fusionar.py lo incluyen en detector_1clase (entrada 'sku110k_sub'
de src/datos/fuentes.py).
Licencia: SKU-110K se publicó para investigación (ver https://github.com/eg4000/SKU110K_CVPR19);
uso académico, NO comercial.

Uso:  python scripts/datos_sku110k.py --max-imagenes 1500 --confirmar
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.datos.rutas import DIR_RAW  # noqa: E402
from src.datos.util import escribir_etiquetas_yolo  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/SKU110K_fixed.tar.gz"
ALIAS = "sku110k_sub"
SPLIT_ARCHIVO = {"train": "train", "val": "valid", "test": "test"}


def elegida(nombre: str, fraccion: float) -> bool:
    return int(hashlib.sha1(nombre.encode()).hexdigest()[:8], 16) / 16 ** 8 < fraccion


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-imagenes", type=int, default=1500, help="tope total de imágenes a guardar")
    ap.add_argument("--confirmar", action="store_true", help="necesario: descarga 13,6 GB")
    args = ap.parse_args()

    destino = DIR_RAW / ALIAS
    if (destino / "data.yaml").exists():
        print(f"Ya existe {destino}; borralo para regenerarlo.")
        return
    if not args.confirmar:
        sys.exit("SKU-110K pesa 13,6 GB. Volvé a correr con --confirmar si querés descargarlo.")

    fraccion = min(1.0, args.max_imagenes / 11743 * 1.1)
    guardadas: dict[str, Path] = {}
    anotaciones: dict[str, list[list[str]]] = defaultdict(list)
    with requests.get(URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                nombre = Path(m.name).name
                if m.name.endswith(".csv") and "annotations_" in nombre:
                    texto = tar.extractfile(m).read().decode("utf-8")
                    for fila in csv.reader(io.StringIO(texto)):
                        if fila:
                            anotaciones[fila[0]].append(fila)
                elif nombre.lower().endswith(".jpg") and len(guardadas) < args.max_imagenes and elegida(nombre, fraccion):
                    split = SPLIT_ARCHIVO.get(nombre.split("_")[0])
                    if not split:
                        continue
                    ruta = destino / split / "images" / nombre
                    ruta.parent.mkdir(parents=True, exist_ok=True)
                    ruta.write_bytes(tar.extractfile(m).read())
                    guardadas[nombre] = ruta
    # CSV: image_name,x1,y1,x2,y2,class,image_width,image_height
    for nombre, ruta in guardadas.items():
        cajas = []
        for f in anotaciones.get(nombre, []):
            x1, y1, x2, y2, ancho, alto = (float(f[i]) for i in (1, 2, 3, 4, 6, 7))
            cajas.append((0, (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto, (x2 - x1) / ancho, (y2 - y1) / alto))
        escribir_etiquetas_yolo(ruta.parent.parent / "labels" / f"{ruta.stem}.txt", cajas)
    (destino / "data.yaml").write_text("nc: 1\nnames: ['producto']\n", encoding="utf-8")
    print(f"{len(guardadas)} imágenes guardadas en {destino}")


if __name__ == "__main__":
    main()
