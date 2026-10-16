"""Indexa las imágenes crudas, arma los grupos anti-fuga y asigna train/val/test.

Para cada imagen de DIR_DATOS/raw/<alias>/ calcula: id original (sin el sufijo .rf.<hash>),
si es copia aumentada, tamaño, pHash y cajas. Luego:
  1. grupo = misma foto original (mismo proyecto + id original) o pHash idéntico;
  2. split por grupo (los datasets con uso 'test_externo' van enteros a test). Los splits de la
     corrida anterior se CONSERVAN (el clasificador y los sintéticos se generaron sobre ellos);
     solo las imágenes nuevas se sortean (--resortear para empezar de cero);
  3. verifica que ningún grupo ni pHash quede en dos splits.
Salida: DIR_DATOS/fusion/_indice_imagenes.csv (el pHash se cachea: correrlo de nuevo es rápido).

Uso:  python scripts/datos_indexar.py [--semilla 42] [--proporciones 0.8 0.1 0.1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.datos.indice import cargar_indice, indexar_dataset  # noqa: E402
from src.datos.rutas import DIR_RAW, RUTA_INDICE, fuentes_descargadas, usos  # noqa: E402
from src.datos.split import asignar_grupos, asignar_splits, verificar_fugas  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--proporciones", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--resortear", action="store_true",
                    help="ignora los splits de la corrida anterior (por defecto se mantienen estables)")
    ap.add_argument("--procesos", type=int, default=None, help="procesos para el pHash (por defecto: núcleos-1)")
    args = ap.parse_args()

    cache = cargar_indice(RUTA_INDICE)
    partes = []
    for f in fuentes_descargadas():
        df = indexar_dataset(DIR_RAW, f["alias"], f"{f['workspace']}/{f['project']}", cache, args.procesos)
        ilegibles = int((df["phash"] == "").sum())
        print(f"[{f['alias']}] {len(df)} imágenes, {df['id_original'].nunique()} originales, "
              f"{int(df['copia_aumentada'].sum())} con sufijo .rf, {ilegibles} ilegibles")
        df = df[df["phash"] != ""]
        if f.get("splits_origen"):
            fuera = ~df["split_origen"].isin(f["splits_origen"])
            print(f"   se excluyen {int(fuera.sum())} imágenes de carpetas fuera de {f['splits_origen']}")
            df = df[~fuera]
        partes.append(df)
    if not partes:
        sys.exit("No hay datasets descargados: corré scripts/datos_descargar.py")
    df = pd.concat(partes, ignore_index=True)
    # Los datasets que no se usan (usar_en vacío) se indexan para el reporte, pero no participan de
    # los grupos: si no, sus duplicados con otros datasets inflarían los grupos sin motivo.
    usados = {f["alias"] for f in fuentes_descargadas() if f.get("usar_en")}
    en_uso = df["alias"].isin(usados)
    previos = {} if args.resortear or "split" not in cache.columns else dict(zip(cache["ruta"], cache["split"]))
    df_uso = asignar_splits(asignar_grupos(df[en_uso]), usos(), tuple(args.proporciones), args.semilla, previos)
    if previos:
        antes = df_uso["ruta"].map(previos)
        conocidas = antes.isin(["train", "val", "test"])
        cambios = int((antes[conocidas] != df_uso.loc[conocidas, "split"]).sum())
        print(f"Splits estables: {int(conocidas.sum())} imágenes ya asignadas, {cambios} cambiaron de split; "
              f"{int((~conocidas).sum())} nuevas")
    df_fuera = asignar_grupos(df[~en_uso]).assign(split="no_usado")
    RUTA_INDICE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df_uso, df_fuera], ignore_index=True).to_csv(RUTA_INDICE, index=False)
    df = df_uso

    entre = df.groupby("grupo")["alias"].nunique()
    print(f"\n{len(df)} imágenes, {df['grupo'].nunique()} grupos (fotos originales); "
          f"{int((entre > 1).sum())} grupos abarcan más de un dataset")
    print(df.groupby(["alias", "split"]).size().unstack(fill_value=0).to_string())
    problemas = verificar_fugas(df)
    if problemas:
        sys.exit("FUGAS entre splits:\n  " + "\n  ".join(problemas))
    print(f"\nSin fugas entre splits. Índice: {RUTA_INDICE}")


if __name__ == "__main__":
    main()
