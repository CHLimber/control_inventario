"""Genera los recortes para el clasificador: DIR_DATOS/recortes/<split>/<clase>/*.jpg.

Parte de las imágenes del multiclase (mismos splits -> sin fugas) y recorta cada caja de las
9 clases con un margen pequeño. Descarta cajas chicas, borrosas (varianza del Laplaciano) y
duplicadas (pHash). Índice: DIR_DATOS/recortes/indice_recortes.csv
(origen, imagen, imagen_original, split, clase, nitidez, phash, ruta_recorte, motivo_descarte).

Uso:  python scripts/datos_recortes.py [--margen 0.05] [--min-lado 32] [--min-nitidez 40] [--copias-train 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import nombres_clases  # noqa: E402
from src.datos.indice import cargar_indice  # noqa: E402
from src.datos.recortes import generar  # noqa: E402
from src.datos.remapeo import cargar_remapeo  # noqa: E402
from src.datos.rutas import (  # noqa: E402
    DIR_RAW, DIR_RECORTES, RUTA_INDICE, RUTA_REMAPEO, alias_para, nombres_por_alias, prioridad_alias,
)
from src.datos.split import seleccionar, verificar_fugas  # noqa: E402
from src.datos.util import SPLITS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--margen", type=float, default=0.05, help="margen alrededor de la caja (fracción)")
    ap.add_argument("--min-lado", type=int, default=32, help="lado mínimo de la caja en píxeles")
    ap.add_argument("--min-nitidez", type=float, default=40.0, help="varianza mínima del Laplaciano")
    ap.add_argument("--copias-train", type=int, default=1,
                    help="imágenes por foto original y dataset en train (0 = todas las copias aumentadas)")
    args = ap.parse_args()

    df = cargar_indice(RUTA_INDICE)
    if df.empty:
        sys.exit("Falta el índice: corré scripts/datos_indexar.py")
    remapeo = cargar_remapeo(RUTA_REMAPEO)
    clases = nombres_clases()
    nombres = nombres_por_alias()
    marcas = alias_para("recortes")
    # Cada recorte es una caja con marca, así que (a diferencia del multiclase) sirven también las
    # imágenes que además tienen cajas genéricas sin marca.
    df = df[df["alias"].isin(marcas) & (df["n_cajas"] > 0)]
    sel = seleccionar(df, args.copias_train, prioridad_alias())
    print(f"{len(sel)} imágenes candidatas ({sel['split'].value_counts().to_dict()})")
    indice = generar(sel, DIR_RAW, DIR_RECORTES, remapeo, nombres, clases, args.margen, args.min_lado,
                     args.min_nitidez)
    if indice.empty:
        sys.exit("No se generó ningún recorte")

    guardados = indice[indice["motivo_descarte"] == ""]
    print("\nRecortes guardados por clase y split:")
    print(guardados.groupby(["clase", "split"]).size().unstack(fill_value=0)
          .reindex(index=clases, columns=list(SPLITS), fill_value=0).to_string())
    print("\nDescartes por motivo:")
    print(indice[indice["motivo_descarte"] != ""].groupby(["motivo_descarte", "clase"]).size()
          .unstack(fill_value=0).to_string())
    problemas = verificar_fugas(guardados.rename(columns={"imagen_original": "grupo"}))
    if problemas:
        sys.exit("FUGAS entre splits:\n  " + "\n  ".join(problemas))
    print(f"\nSin fugas entre splits. Índice: {DIR_RECORTES / 'indice_recortes.csv'}")


if __name__ == "__main__":
    main()
