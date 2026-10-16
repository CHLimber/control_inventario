"""Arma el zip que se sube a Google Colab para entrenar los detectores con GPU.

Contenido de <MAIN>/runs/detector/colab/inventario_detector.zip:
  inventario_detector/
    detector_1clase/{images,labels}/{train,val,test}/  + data.yaml (SIN 'path': rutas relativas)
    multiclase/{images,labels}/{train,val,test}/       + data.yaml
    sinteticos/{images,labels}/train/                  (solo con --con-sinteticos y si existen)
    yolo26n.pt (+ yolo26s.pt con --con-s)               pesos COCO de partida, los mismos que en local
    LEEME.txt

Sin 'path' en el data.yaml, Ultralytics toma como raíz la carpeta del propio data.yaml, así que
el zip funciona en cualquier carpeta donde se descomprima (Colab: /content/inventario_detector).
Las imágenes (JPEG) se guardan sin recomprimir (ZIP_STORED): comprimirlas no ahorra nada y tarda.

Uso:
  python scripts/detector_empaquetar_colab.py                       # ambas variantes
  python scripts/detector_empaquetar_colab.py --variantes 1clase --con-sinteticos
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent
if str(RAIZ_REPO) not in sys.path:
    sys.path.insert(0, str(RAIZ_REPO))

from scripts.detector_entrenar import (DIR_RUNS, VARIANTES, buscar_sinteticos, dir_dataset,  # noqa: E402
                                       imagenes_de, nombres_variante)
from src.config import DIR_MODELOS  # noqa: E402

RAIZ_ZIP = "inventario_detector"


def data_yaml_relativo(variante: str, con_sinteticos: bool) -> str:
    import yaml

    train: str | list[str] = "images/train"
    if con_sinteticos:
        train = ["images/train", "../sinteticos/images/train"]
    cfg = {"train": train, "val": "images/val", "test": "images/test",
           "nc": len(nombres_variante(variante)), "names": dict(enumerate(nombres_variante(variante)))}
    return ("# Generado por scripts/detector_empaquetar_colab.py. Sin 'path': la raíz es esta carpeta.\n"
            + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))


def agregar_split(z: zipfile.ZipFile, origen: Path, destino: str, split: str, muestra: int = 0) -> int:
    n = 0
    imgs = imagenes_de(origen / "images" / split)
    if muestra:
        imgs = imgs[:: max(1, len(imgs) // muestra)][:muestra]
    for img in imgs:
        lbl = origen / "labels" / split / (img.stem + ".txt")
        try:
            z.write(img, f"{destino}/images/{split}/{img.name}", compress_type=zipfile.ZIP_STORED)
            if lbl.exists():
                z.write(lbl, f"{destino}/labels/{split}/{lbl.name}", compress_type=zipfile.ZIP_DEFLATED)
            n += 1
        except FileNotFoundError:
            print(f"AVISO: desapareció {img.name} mientras se empaquetaba (¿Datos regenerando?)")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variantes", nargs="+", choices=list(VARIANTES), default=list(VARIANTES))
    ap.add_argument("--con-sinteticos", action="store_true")
    ap.add_argument("--con-s", action="store_true", help="incluir también yolo26s.pt")
    ap.add_argument("--muestra", type=int, default=0, help="solo N imágenes por split (para probar el zip)")
    ap.add_argument("--salida", type=Path, default=DIR_RUNS / "colab" / f"{RAIZ_ZIP}.zip")
    a = ap.parse_args(argv)

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    resumen = []
    with zipfile.ZipFile(a.salida, "w", zipfile.ZIP_DEFLATED) as z:
        sint = []
        if a.con_sinteticos:
            sint = buscar_sinteticos(a.variantes[0])
            if sint:
                n = agregar_split(z, sint[0].parent.parent, f"{RAIZ_ZIP}/sinteticos", "train", a.muestra)
                resumen.append(f"sinteticos/train: {n} imágenes (de {sint[0]})")
            else:
                print("AVISO: no hay sintéticos; se empaqueta sin ellos.")
        for v in a.variantes:
            d = dir_dataset(v)
            carpeta = f"{RAIZ_ZIP}/{VARIANTES[v]}"
            for s in ("train", "val", "test"):
                n = agregar_split(z, d, carpeta, s, a.muestra)
                resumen.append(f"{VARIANTES[v]}/{s}: {n} imágenes")
                print(resumen[-1])
            z.writestr(f"{carpeta}/data.yaml", data_yaml_relativo(v, bool(sint)))
        for pesos in ["yolo26n.pt"] + (["yolo26s.pt"] if a.con_s else []):
            if (DIR_MODELOS / pesos).exists():
                z.write(DIR_MODELOS / pesos, f"{RAIZ_ZIP}/{pesos}")
        z.writestr(f"{RAIZ_ZIP}/LEEME.txt",
                   "Subir este zip a Google Drive en MyDrive/inventario/ (o subirlo directo a Colab)\n"
                   "y abrir notebooks/04_detector_colab.ipynb. Contenido:\n  " + "\n  ".join(resumen) + "\n")
    mb = a.salida.stat().st_size / 1e6
    print(f"\nListo: {a.salida} ({mb:.0f} MB)\nSubilo a Google Drive en MyDrive/inventario/{a.salida.name}")


if __name__ == "__main__":
    main()
