"""Arma los datasets YOLO fusionados a partir del índice y del remapeo.

  DIR_DATOS/fusion/detector_1clase/   todo lo que no se descarta -> 'producto'
  DIR_DATOS/fusion/multiclase/        8 clases (ids = orden de config/clases.json)
Qué dataset entra a cada uno lo decide 'usar_en' en src/datos/fuentes.py.

Cada uno con images/ labels/ {train,val,test}, data.yaml y _manifiesto.csv. Las imágenes son
enlaces duros a DIR_DATOS/raw (no ocupan espacio extra). Usa los splits de datos_indexar.py y
verifica que no haya fugas.

Uso:  python scripts/datos_fusionar.py [--solo detector_1clase|multiclase] [--copias-train 1] [--max-negativos 0.1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import nombres_clases  # noqa: E402
from src.datos.fusion import MODOS, construir, conteos  # noqa: E402
from src.datos.color import cargar_rechazos  # noqa: E402
from src.datos.indice import cargar_indice  # noqa: E402
from src.datos.remapeo import cargar_remapeo, validar_remapeo  # noqa: E402
from src.datos.rutas import (  # noqa: E402
    DIR_FUSION, DIR_RAW, RUTA_INDICE, RUTA_REMAPEO, alias_para, nombres_por_alias, prioridad_alias,
)
from src.datos.split import verificar_fugas  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", choices=MODOS, help="arma solo uno de los dos")
    ap.add_argument("--copias-train", type=int, default=1,
                    help="imágenes por foto original y dataset en train (0 = todas las copias aumentadas)")
    ap.add_argument("--max-negativos", type=float, default=0.1,
                    help="imágenes sin cajas útiles, como fracción de las positivas de cada split")
    args = ap.parse_args()

    df = cargar_indice(RUTA_INDICE)
    if df.empty or "split" not in df.columns:
        sys.exit("Falta el índice: corré scripts/datos_indexar.py")
    remapeo = cargar_remapeo(RUTA_REMAPEO)
    rechazos = cargar_rechazos(DIR_FUSION / "_control_color.csv")
    if not rechazos:
        print("AVISO: no hay control de color (corré scripts/datos_color.py)")
    clases = nombres_clases()
    errores = validar_remapeo(remapeo, clases)
    if not remapeo or errores:
        sys.exit("Remapeo vacío o inválido: corré scripts/datos_remapeo.py\n" + "\n".join(errores))

    alias_por_modo = {"detector_1clase": alias_para("detector"), "multiclase": alias_para("multiclase")}
    nombres = nombres_por_alias()
    hubo_fugas = False
    for modo in MODOS:
        if args.solo and modo != args.solo:
            continue
        print(f"\n=== {modo} ===")
        man = construir(df, DIR_RAW, DIR_FUSION / modo, modo, remapeo, nombres, clases, alias_por_modo[modo],
                        args.copias_train, args.max_negativos, prioridad_alias(), rechazos)
        if man.empty:
            print("  (vacío)")
            continue
        nombres_salida = ["producto"] if modo == "detector_1clase" else clases
        tabla = conteos(man, nombres_salida)
        print(man.groupby(["split"]).agg(imagenes=("ruta", "size"), cajas=("n_cajas_salida", "sum")).to_string())
        print(tabla.pivot(index="clase", columns="split", values="cajas")[["train", "val", "test"]]
              .reindex(nombres_salida).to_string())
        problemas = verificar_fugas(man)
        if problemas:
            hubo_fugas = True
            print("  FUGAS:\n   " + "\n   ".join(problemas))
        else:
            print("  Sin fugas entre splits.")
    if hubo_fugas:
        sys.exit(1)


if __name__ == "__main__":
    main()
