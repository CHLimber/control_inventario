"""Ida y vuelta con Google Colab para entrenar el clasificador con GPU.

1) empaquetar: arma UN zip con el código mínimo + datos/recortes para subir a Colab.
       python scripts/clasificador_empaquetar_colab.py empaquetar
   → modelos/clasificador/colab/clasificador_colab.zip   (subirlo en notebooks/05_clasificador_colab.ipynb)

2) importar: trae lo que devuelve Colab (resultados_clasificador.zip) a su lugar.
       python scripts/clasificador_empaquetar_colab.py importar <ruta>/resultados_clasificador.zip
   → modelos/clasificador/*  (clasificador.keras, clases.json, metricas.json, comparacion.csv, …)
   → logs/clasificador/*     (TensorBoard)

Dentro del zip todo cuelga de "control_inventario/" con la misma estructura del repo, así el
notebook de Colab corre exactamente el mismo scripts/clasificador_entrenar.py que en local.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402

PREFIJO = "control_inventario"
CODIGO = [
    "src/__init__.py", "src/config.py", "src/vision/__init__.py", "src/vision/clasificador.py",
    "src/vision/clasificador_modelos.py", "scripts/clasificador_entrenar.py", "scripts/clasificador_sintetico.py",
    "config/clases.json",
]
EXT_IMAGEN = {".jpg", ".jpeg", ".png"}


def empaquetar(destino: Path, dir_recortes: Path, con_datos: bool = True) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    n_img = 0
    with zipfile.ZipFile(destino, "w") as z:
        for rel in CODIGO:
            z.write(RAIZ / rel, f"{PREFIJO}/{rel}", compress_type=zipfile.ZIP_DEFLATED)
        if con_datos:
            if not (dir_recortes / "train").is_dir():
                raise SystemExit(f"No hay recortes en {dir_recortes} (falta correr el agente Datos).")
            indice = dir_recortes / "indice_recortes.csv"
            if indice.exists():
                z.write(indice, f"{PREFIJO}/datos/recortes/indice_recortes.csv", compress_type=zipfile.ZIP_DEFLATED)
            for ruta in sorted(dir_recortes.rglob("*")):
                if ruta.is_file() and ruta.suffix.lower() in EXT_IMAGEN:
                    rel = ruta.relative_to(dir_recortes).as_posix()
                    # los JPEG ya vienen comprimidos: se guardan tal cual (más rápido, mismo tamaño)
                    z.write(ruta, f"{PREFIJO}/datos/recortes/{rel}", compress_type=zipfile.ZIP_STORED)
                    n_img += 1
    print(f"Zip: {destino}  ({destino.stat().st_size / 1e6:.1f} MB, {n_img} recortes)")
    print("Subilo en Colab con la celda 'Subir el zip' de notebooks/05_clasificador_colab.ipynb.")
    return destino


def _destino_seguro(nombre: str) -> Path | None:
    """modelos/clasificador/... → DIR_MODELOS/clasificador/...; logs/clasificador/... → DIR_LOGS/...

    Ignora cualquier otra ruta (y las que intenten salir de la carpeta con '..').
    """
    partes = PurePosixPath(nombre).parts
    if PREFIJO in partes[:1]:
        partes = partes[1:]
    if len(partes) < 3 or ".." in partes or partes[1] != "clasificador":
        return None
    base = {"modelos": config.DIR_MODELOS, "logs": config.DIR_LOGS}.get(partes[0])
    return base.joinpath(*partes[1:]) if base else None


def importar(zip_resultados: Path) -> None:
    n = 0
    with zipfile.ZipFile(zip_resultados) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            destino = _destino_seguro(info.filename)
            if destino is None:
                print(f"  (se ignora {info.filename})")
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(z.read(info))
            n += 1
    print(f"{n} archivos copiados a {config.DIR_MODELOS / 'clasificador'} y {config.DIR_LOGS / 'clasificador'}")
    met = config.DIR_MODELOS / "clasificador" / "metricas.json"
    if met.exists():
        m = json.loads(met.read_text(encoding="utf-8"))
        test = m.get("test") or {}
        print(f"Modelo elegido: {m['modelo']} ({m['tamano_entrada']} px) · val acc {m['val']['accuracy']:.3f}"
              f" · test acc {test.get('accuracy', float('nan')):.3f} · umbral sugerido {m['umbral_sugerido']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="accion", required=True)
    e = sub.add_parser("empaquetar")
    e.add_argument("--salida", type=Path, default=config.DIR_MODELOS / "clasificador" / "colab" / "clasificador_colab.zip")
    e.add_argument("--datos", type=Path, default=config.DIR_DATOS / "recortes")
    e.add_argument("--sin-datos", action="store_true", help="solo el código (si los recortes ya están en Drive)")
    i = sub.add_parser("importar")
    i.add_argument("zip", type=Path)
    a = ap.parse_args()
    if a.accion == "empaquetar":
        empaquetar(a.salida, a.datos, not a.sin_datos)
    else:
        importar(a.zip)


if __name__ == "__main__":
    main()
