"""Arma DIR_DATOS/fusion/_biblioteca: imágenes de train de las fuentes que alimentan la biblioteca de envases.

Las fuentes con "biblioteca": True en src/datos/fuentes.py (cold_drinks_2, hccb_overall) no entran al
multiclase, pero sus góndolas tienen envases completos y de frente. Después:

  python scripts/sinteticos_objetos.py --fuente <DATOS>/fusion/_biblioteca --max-por-clase 400

recorta esos envases sin fondo y los suma a la biblioteca de los sintéticos (es incremental).

Uso:  python scripts/datos_biblioteca.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import nombres_clases  # noqa: E402
from src.datos.biblioteca import construir_fuente_biblioteca  # noqa: E402
from src.datos.fuentes import FUENTES  # noqa: E402
from src.datos.indice import cargar_indice  # noqa: E402
from src.datos.remapeo import cargar_remapeo  # noqa: E402
from src.datos.rutas import DIR_FUSION, DIR_RAW, RUTA_INDICE, RUTA_REMAPEO, fuentes_descargadas, nombres_por_alias  # noqa: E402


def main() -> None:
    df = cargar_indice(RUTA_INDICE)
    if df.empty:
        sys.exit("Falta el índice: corré scripts/datos_indexar.py")
    descargadas = {f["alias"] for f in fuentes_descargadas()}
    aliases = {f["alias"] for f in FUENTES if f.get("biblioteca") and f["alias"] in descargadas}
    if not aliases:
        sys.exit("Ninguna fuente descargada tiene \"biblioteca\": True")
    destino = DIR_FUSION / "_biblioteca"
    conteo = construir_fuente_biblioteca(df, DIR_RAW, destino, cargar_remapeo(RUTA_REMAPEO), nombres_por_alias(),
                                         nombres_clases(), aliases)
    n_img = len(list((destino / "images" / "train").glob("*")))
    print(f"{destino}: {n_img} imágenes de train de {sorted(aliases)}")
    for clase, n in conteo.items():
        print(f"  {clase:15s} {n} cajas")


if __name__ == "__main__":
    main()
