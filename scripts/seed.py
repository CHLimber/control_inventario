"""Carga los datos de ejemplo en la BD de URL_BD (por defecto datos/inventario.db).

Uso (Anaconda Prompt, entorno inventariovision, desde la raíz del repo):
    python scripts/seed.py                      # productos + movimientos de hoy
    python scripts/seed.py --fecha 2026-10-14   # movimientos de ese día
    python scripts/seed.py --solo-productos
    python scripts/seed.py --reiniciar          # BORRA todas las tablas y vuelve a sembrar
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import url_bd  # noqa: E402
from src.db import borrar_tablas, crear_fabrica_sesiones, crear_motor, crear_tablas  # noqa: E402
from src.db.seed import sembrar, stock_final_esperado  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Siembra productos y un día de movimientos.")
    parser.add_argument("--fecha", type=date.fromisoformat, default=None,
                        help="Día ficticio de los movimientos (YYYY-MM-DD); por defecto hoy")
    parser.add_argument("--solo-productos", action="store_true")
    parser.add_argument("--reiniciar", action="store_true", help="Borra y recrea las tablas")
    parser.add_argument("--url", default=None, help="URL de la BD (por defecto URL_BD)")
    args = parser.parse_args()

    url = args.url or url_bd()
    motor = crear_motor(url)
    if args.reiniciar:
        borrar_tablas(motor)
    crear_tablas(motor)
    with crear_fabrica_sesiones(motor)() as sesion:
        info = sembrar(sesion, args.fecha, con_movimientos=not args.solo_productos)
    print(f"BD: {url}")
    print(f"Productos: {info['productos']} · movimientos nuevos: {info['movimientos']} "
          f"· día: {info['dia']}")
    if info["movimientos"]:
        print("Stock al final del día (si la BD estaba vacía):")
        for clase, stock in stock_final_esperado().items():
            print(f"  {clase:<15} {stock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
