"""Descarga un conjunto chico de datasets públicos de Roboflow para la prueba de viabilidad.

Guarda en <RAIZ>/datos/viabilidad/salidas/raw/<workspace>__<proyecto>/ (ignorado por git).
La clave ROBOFLOW_API_KEY se lee de <RAIZ>/.env (RAIZ = repo principal) y nunca se imprime.

Uso:
    python scripts/viabilidad_descargar.py --raiz "<MAIN>" [--datasets ws/proj ws/proj ...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CANDIDATOS = [
    "rosd/final_hr",
    "darians-workspace-4fake/bottle-detection-q9eas",
    "roboflow-100/soda-bottles",
    "ai-k6ynk/bottle-can-cup-dataset-pvesi",
]


def descargar(raiz: Path, slug: str, formato: str = "yolov8") -> Path | None:
    from roboflow import Roboflow

    ws, proj = slug.split("/")
    destino = raiz / "datos" / "viabilidad" / "salidas" / "raw" / f"{ws}__{proj}"
    if (destino / "data.yaml").exists():
        print(f"[ya existe] {slug} -> {destino}")
        return destino
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proyecto = rf.workspace(ws).project(proj)
    versiones = proyecto.versions()
    if not versiones:
        print(f"[sin versiones] {slug}")
        return None
    nums = [int(str(v.version).split("/")[-1]) for v in versiones]
    num = max(nums)
    print(f"[descargando] {slug} v{num} -> {destino}")
    proyecto.version(num).download(formato, location=str(destino), overwrite=True)
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", required=True, help="Repo principal (donde están .env y datos/)")
    ap.add_argument("--datasets", nargs="*", default=CANDIDATOS)
    ap.add_argument("--formato", default="yolov8")
    a = ap.parse_args()
    raiz = Path(a.raiz)
    from dotenv import load_dotenv

    load_dotenv(raiz / ".env")
    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise SystemExit("Falta ROBOFLOW_API_KEY en .env")
    for slug in a.datasets:
        try:
            descargar(raiz, slug, a.formato)
        except Exception as e:  # seguir con los demás
            print(f"[error] {slug}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
