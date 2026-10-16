"""Control de color por clase de todas las cajas con marca (detecta etiquetas equivocadas).

Para cada caja de las 9 clases en los datasets que van al multiclase o a los recortes, mide en la
zona central de la caja la fracción de rojo/naranja/verde/azul/oscuro/gris (src/datos/color.py) y
aplica las reglas por clase. Salida: DIR_DATOS/fusion/_control_color.csv (alias, ruta, caja, clase,
medidas, motivo). datos_fusionar.py y datos_recortes.py la usan para descartar las cajas con color
incoherente (el motivo queda registrado; no se borra nada de raw/).
También imprime percentiles por clase y fuente y guarda una grilla de rechazos en docs/datos/.

Uso:  python scripts/datos_color.py [--procesos N] [--sin-grilla]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import nombres_clases  # noqa: E402
from src.datos.color import MEDIDAS, medir, motivo_color  # noqa: E402
from src.datos.fusion import cajas_remapeadas  # noqa: E402
from src.datos.indice import cargar_indice  # noqa: E402
from src.datos.remapeo import cargar_remapeo  # noqa: E402
from src.datos.rutas import (  # noqa: E402
    DIR_DOCS_DATOS, DIR_FUSION, DIR_RAW, RUTA_INDICE, RUTA_REMAPEO, alias_para, nombres_por_alias,
)
from src.datos.util import a_pixeles, ruta_etiqueta  # noqa: E402

RUTA_COLOR = DIR_FUSION / "_control_color.csv"


def _procesar(args: tuple) -> list[dict]:
    alias, ruta, nombres, remapeo, clases = args
    from PIL import Image, ImageOps

    ruta_img = DIR_RAW / ruta
    cajas = cajas_remapeadas(ruta_etiqueta(ruta_img), nombres, remapeo, alias, "multiclase", clases)
    if not cajas:
        return []
    try:
        with Image.open(ruta_img) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:  # noqa: BLE001
        return []
    filas = []
    for i, (cid, xc, yc, w, h) in enumerate(cajas):
        med = medir(img, a_pixeles(xc, yc, w, h, img.width, img.height))
        clase = clases[cid]
        filas.append({"alias": alias, "ruta": ruta, "caja": i, "clase": clase,
                      **{m: round(v, 3) for m, v in med.items()}, "motivo": motivo_color(clase, med)})
    return filas


def grilla_rechazos(df: pd.DataFrame, destino: Path, por_clase: int = 10) -> None:
    from PIL import Image, ImageDraw

    clases = [c for c in nombres_clases() if (df["clase"] == c).any()]
    lado = 90
    lienzo = Image.new("RGB", (por_clase * lado + 130, max(1, len(clases)) * lado), (255, 255, 255))
    dib = ImageDraw.Draw(lienzo)
    remapeo = cargar_remapeo(RUTA_REMAPEO)
    nombres = nombres_por_alias()
    for i, c in enumerate(clases):
        dib.text((4, i * lado + lado // 2 - 6), c, fill=(0, 0, 0))
        d = df[df["clase"] == c].sample(frac=1, random_state=0).head(por_clase)
        for j, r in enumerate(d.itertuples()):
            ruta_img = DIR_RAW / r.ruta
            cajas = cajas_remapeadas(ruta_etiqueta(ruta_img), nombres[r.alias], remapeo, r.alias, "multiclase",
                                     nombres_clases())
            _, xc, yc, w, h = cajas[r.caja]
            with Image.open(ruta_img) as im:
                im = im.convert("RGB")
                rec = im.crop(a_pixeles(xc, yc, w, h, im.width, im.height))
            rec.thumbnail((lado - 4, lado - 14))
            lienzo.paste(rec, (130 + j * lado + 2, i * lado + 2))
            dib.text((130 + j * lado + 2, i * lado + lado - 12), r.motivo.replace("color_", "")[:14], fill=(200, 0, 0))
    destino.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destino, "JPEG", quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sin-grilla", action="store_true")
    args = ap.parse_args()

    df = cargar_indice(RUTA_INDICE)
    usados = alias_para("multiclase") | alias_para("recortes")
    df = df[df["alias"].isin(usados) & df["split"].isin(["train", "val", "test"]) & (df["n_cajas"] > 0)]
    remapeo = cargar_remapeo(RUTA_REMAPEO)
    nombres = nombres_por_alias()
    clases = nombres_clases()
    tareas = [(r.alias, r.ruta, nombres[r.alias], remapeo, clases) for r in df.itertuples()]
    print(f"Midiendo color en {len(tareas)} imágenes...")
    filas = []
    with ProcessPoolExecutor(args.procesos) as ex:
        for res in ex.map(_procesar, tareas, chunksize=32):
            filas.extend(res)
    col = pd.DataFrame(filas)
    col.to_csv(RUTA_COLOR, index=False)

    col["rechazada"] = col["motivo"] != ""
    print("\nRechazo por clase y fuente (fracción de cajas):")
    print(col.pivot_table(index="clase", columns="alias", values="rechazada", aggfunc="mean")
          .reindex(clases).round(2).to_string())
    print("\nMediana de la medida principal por clase y fuente:")
    for c in clases:
        d = col[col["clase"] == c]
        if d.empty:
            continue
        print(f"  {c:15s}", d.groupby("alias")[list(MEDIDAS)].median().round(2).to_dict("index"))
    if not args.sin_grilla:
        grilla_rechazos(col[col["rechazada"]], DIR_DOCS_DATOS / "color_rechazos.jpg")
        grilla_rechazos(col[~col["rechazada"]], DIR_DOCS_DATOS / "color_aceptados.jpg")
    print(f"\n{int(col['rechazada'].sum())} de {len(col)} cajas con color incoherente. Detalle: {RUTA_COLOR}")


if __name__ == "__main__":
    main()
