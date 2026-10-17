"""Genera/actualiza el BORRADOR config/remapeo_clases.yaml a partir de las clases de cada dataset.

- Las entradas que ya están en el YAML se RESPETAN (ediciones a mano o aprobadas por el orquestador);
  solo se agregan las clases nuevas, con una sugerencia automática (src/datos/remapeo.py).
- Con --regenerar se recalcula todo desde cero.
- Con --validar solo comprueba que los destinos existan (sale con error si no).

Uso:  python scripts/datos_remapeo.py [--regenerar | --validar]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import nombres_clases  # noqa: E402
from src.datos.remapeo import cargar_remapeo, escribir_remapeo, sugerir, validar_remapeo  # noqa: E402
from src.datos.rutas import RUTA_REMAPEO, fuentes_descargadas, nombres_por_alias  # noqa: E402

ENCABEZADO = """\
# BORRADOR generado por scripts/datos_remapeo.py y revisado a mano por el agente Datos.
# 3.ª ronda (cold_drinks_2, hccb_overall): propuesta pendiente de aprobación.
# Lo APRUEBA el orquestador. Las entradas existentes se respetan al regenerar (salvo --regenerar).
#
# Formato:  <alias_dataset>: { <clase_original>: <clase_nuestra> | solo_detector | descartar }
#   - <clase_nuestra> debe existir en config/clases.json.
#   - solo_detector = es una bebida/envase y sirve como "producto" para el detector de 1 clase,
#     pero su marca no es una de las nuestras o no es confiable (no entra al multiclase ni a recortes;
#     en el multiclase queda como fondo).
#   - auto:<marca> = el dataset da la marca pero no el formato (p. ej. auto:pepsi): botella o lata
#     según la forma de la caja (src/datos/remapeo.py: RATIO_LATA / RATIO_BOTELLA); si es dudosa, solo_detector.
#     auto:<marca>:botella acepta solo botellas por la forma (lo que no encaje queda como solo_detector).
#   - descartar = no es un producto de interés (snacks, vasos, fondo...): la caja se elimina.
#   - Una clase original que no figure acá se trata como "descartar".
#
# Formato por defecto de cada dataset (para clases sin "can"/"bottle" en el nombre):
#   soda_bottles -> botella (botellas PET vistas desde arriba en heladeras).
#
# Qué dataset entra a cada salida (detector / multiclase / recortes) lo decide "usar_en" en
# src/datos/fuentes.py: lamar_grocery (v9, grises) no se usa; de lamar_grocery_color solo la carpeta train;
# soda_bottles solo en el detector.
"""

FORMATO_DEFECTO = {"soda_bottles": "botella"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--regenerar", action="store_true", help="recalcula todo e ignora ediciones previas")
    g.add_argument("--validar", action="store_true", help="solo valida el YAML existente")
    args = ap.parse_args()

    clases = nombres_clases()
    remapeo = {} if args.regenerar else cargar_remapeo(RUTA_REMAPEO)
    if not args.validar:
        nombres = nombres_por_alias()
        nuevas = 0
        for f in fuentes_descargadas():
            actual = remapeo.setdefault(f["alias"], {})
            for nombre in nombres[f["alias"]]:
                if nombre not in actual:
                    actual[nombre] = sugerir(nombre, clases, FORMATO_DEFECTO.get(f["alias"]))
                    nuevas += 1
        # Solo se reescribe si hay algo nuevo: así se conservan los comentarios hechos a mano.
        if nuevas or args.regenerar or not RUTA_REMAPEO.exists():
            escribir_remapeo(RUTA_REMAPEO, remapeo, ENCABEZADO)
            print(f"Remapeo guardado en {RUTA_REMAPEO} ({nuevas} clases nuevas)")
        else:
            print("Sin clases nuevas: el YAML queda como está.")

    for alias, m in remapeo.items():
        print(f"\n[{alias}]")
        for nombre, dest in m.items():
            print(f"  {nombre!r:45s} -> {dest}")
    errores = validar_remapeo(remapeo, clases)
    if errores:
        sys.exit("Errores en el remapeo:\n  " + "\n  ".join(errores))
    print("\nRemapeo válido.")


if __name__ == "__main__":
    main()
