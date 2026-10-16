"""Genera un lote de estantes sintéticos (3 niveles, una fila) con etiquetas YOLO y conteos exactos.

Requiere la biblioteca de objetos sin fondo (``scripts/sinteticos_objetos.py``).

Ejemplos:
  set INVENTARIO_DATOS=<MAIN>\\datos
  python scripts/sinteticos_generar.py --n 500 --semilla 0 --salida lote_500
  python scripts/sinteticos_generar.py --n 3000 --semilla 1 --salida lote_3000 --procesos 4 --figuras

``--salida`` es un nombre de lote (se crea en DIR_DATOS/sinteticos/<nombre>) o una ruta absoluta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import DIR_DATOS, DIR_MAQUETA, cargar_clases, nombres_clases  # noqa: E402
from src.sinteticos import fondos as fnd  # noqa: E402
from src.sinteticos import objetos as obj  # noqa: E402
from src.sinteticos.composicion import Parametros  # noqa: E402
from src.sinteticos.lote import generar_lote  # noqa: E402
from src.sinteticos.visualizar import distribucion, grilla  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default="lote_500")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--objetos", type=Path, default=DIR_DATOS / "sinteticos" / "_objetos")
    ap.add_argument("--fusion", type=Path, default=DIR_DATOS / "fusion",
                    help="dataset fusionado (fondos Background_* de train y rutas de data_mixto.yaml)")
    ap.add_argument("--figuras", action="store_true", help="también escribe docs/sinteticos/muestras.png y distribucion.png")
    ap.add_argument("--dir-figuras", type=Path, default=RAIZ / "docs" / "sinteticos")
    ap.add_argument("--param", action="append", default=[], metavar="CLAVE=VALOR",
                    help="cambia un campo de Parametros (p. ej. --param p_acercamiento=0.3); se puede repetir")
    a = ap.parse_args()
    params = Parametros()
    for kv in a.param:
        clave, valor = kv.split("=", 1)
        actual = getattr(params, clave)   # AttributeError si la clave no existe
        setattr(params, clave, type(actual)(float(valor)) if isinstance(actual, (int, float)) else valor)

    salida = Path(a.salida) if Path(a.salida).is_absolute() else DIR_DATOS / "sinteticos" / a.salida
    dir_train = a.fusion / "multiclase" / "images" / "train"
    bib = obj.cargar_biblioteca(a.objetos, dir_train) if a.objetos.exists() else {}
    if not bib:
        sys.exit(f"No hay objetos en {a.objetos}: correr antes scripts/sinteticos_objetos.py")
    fondos = fnd.listar_fondos_reales(a.fusion / "multiclase" / "images" / "train", DIR_MAQUETA / "fondos")
    print(f"biblioteca: { {k: len(v) for k, v in sorted(bib.items())} }")
    print(f"fondos reales: {len(fondos)}")
    faltan = [c for c in nombres_clases() if not bib.get(c)]
    if faltan:
        print(f"AVISO: clases sin objetos (no aparecerán): {faltan}")
    res = generar_lote(salida, a.n, a.semilla, bib, cargar_clases()["clases"], fondos, params,
                       procesos=a.procesos, dir_fusion=a.fusion if a.fusion.exists() else None)
    print(json.dumps({k: v for k, v in res.items() if k != "parametros"}, indent=2, ensure_ascii=False))
    print(f"-> {salida}")
    if a.figuras:
        print(grilla(salida, a.dir_figuras / "muestras.png", nombres_clases()))
        print(distribucion(salida, a.dir_figuras / "distribucion.png", nombres_clases()))


if __name__ == "__main__":
    main()
