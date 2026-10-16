"""Descarga los datasets de src/datos/fuentes.py desde Roboflow Universe (formato YOLOv8).

- Destino: DIR_DATOS/raw/<alias>/ (respeta INVENTARIO_DATOS).
- Idempotente: si ya existe raw/<alias>/data.yaml no se vuelve a descargar (salvo --forzar).
- Elige la última versión del proyecto y registra licencia, versión, cantidad de imágenes y
  clases en raw/<alias>/_fuente.json (y el resumen de todo en raw/_inspeccion.json).
- Usa el paquete `roboflow` si está instalado; si no, la API REST equivalente.
- La clave ROBOFLOW_API_KEY se lee del entorno o de <DIR_DATOS>/../.env. Nunca se imprime.

Uso:
  python scripts/datos_descargar.py                 # descarga todo lo que falte
  python scripts/datos_descargar.py --inspeccionar  # solo consulta metadatos (no descarga)
  python scripts/datos_descargar.py --dataset rosd_final_hr soda_bottles --forzar
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.config import DIR_DATOS  # noqa: E402
from src.datos.fuentes import FUENTES  # noqa: E402
from src.datos.util import cargar_env_principal, listar_imagenes  # noqa: E402

API = "https://api.roboflow.com"
DIR_RAW = DIR_DATOS / "raw"


def clave_api() -> str:
    cargar_env_principal(DIR_DATOS)
    clave = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not clave:
        sys.exit("Falta ROBOFLOW_API_KEY (en el entorno o en el .env del repo principal)")
    return clave


def _sin_clave(texto: str, clave: str) -> str:
    return texto.replace(clave, "***")


def metadatos(ws: str, proy: str, clave: str) -> dict:
    """Consulta el proyecto: licencia, tipo, clases y versiones."""
    r = requests.get(f"{API}/{ws}/{proy}", params={"api_key": clave}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_sin_clave(r.text[:200], clave)}")
    datos = r.json()
    p = datos.get("project", {})
    versiones = []
    for v in datos.get("versions", []):
        versiones.append({
            "version": int(str(v["id"]).split("/")[-1]),
            "imagenes": v.get("images"),
            "splits": v.get("splits"),
            "aumentos": v.get("augmentation") or {},
            "preprocesado": v.get("preprocessing") or {},
            "exportaciones": v.get("exports") or [],
        })
    versiones.sort(key=lambda v: v["version"])
    return {
        "nombre": p.get("name"),
        "tipo_proyecto": p.get("type"),
        "licencia": p.get("license"),
        "imagenes_proyecto": p.get("images"),
        "clases_proyecto": p.get("classes") or {},
        "versiones": versiones,
    }


def _descargar_zip_rest(ws: str, proy: str, version: int, formato: str, clave: str, destino: Path) -> None:
    """Pide la exportación (esperando si Roboflow la está generando) y descomprime el zip."""
    url = f"{API}/{ws}/{proy}/{version}/{formato}"
    enlace = None
    for _ in range(120):  # hasta ~10 minutos
        r = requests.get(url, params={"api_key": clave}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {_sin_clave(r.text[:200], clave)}")
        datos = r.json()
        enlace = (datos.get("export") or {}).get("link")
        if enlace:
            break
        print(f"   generando exportación... {datos.get('progress', '')}")
        time.sleep(5)
    if not enlace:
        raise RuntimeError("Roboflow no entregó el enlace de exportación")

    destino.mkdir(parents=True, exist_ok=True)
    ruta_zip = destino.parent / f"_{destino.name}.zip"
    with requests.get(enlace, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(ruta_zip, "wb") as f:
            for trozo in r.iter_content(1 << 20):
                f.write(trozo)
    raiz = destino.resolve()
    with zipfile.ZipFile(ruta_zip) as z:
        for miembro in z.namelist():  # protección contra rutas que escapan del destino
            if not (raiz / miembro).resolve().is_relative_to(raiz):
                raise RuntimeError(f"Ruta sospechosa en el zip: {miembro}")
        z.extractall(raiz)
    ruta_zip.unlink()


def _descargar_con_paquete(ws: str, proy: str, version: int, formato: str, clave: str, destino: Path) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=clave)
    rf.workspace(ws).project(proy).version(version).download(formato, location=str(destino), overwrite=True)


def descargar(fuente: dict, clave: str, forzar: bool, solo_inspeccionar: bool) -> dict:
    alias = fuente["alias"]
    destino = DIR_RAW / alias
    info = {"alias": alias, "id": f"{fuente['workspace']}/{fuente['project']}", **fuente}
    meta = metadatos(fuente["workspace"], fuente["project"], clave)
    info.update(meta)
    if not meta["versiones"]:
        raise RuntimeError("el proyecto no tiene versiones publicadas")
    version = fuente.get("version") or meta["versiones"][-1]["version"]
    info["version_usada"] = version
    v = next((x for x in meta["versiones"] if x["version"] == version), {})
    print(f"[{alias}] {info['id']} v{version}: licencia={meta['licencia']} tipo={meta['tipo_proyecto']} "
          f"imagenes_version={v.get('imagenes')} aumentos={'si' if v.get('aumentos') else 'no'} "
          f"clases={len(meta['clases_proyecto'])}")
    if solo_inspeccionar:
        return info

    if (destino / "data.yaml").exists() and not forzar:
        print(f"[{alias}] ya descargado ({len(listar_imagenes(destino))} imágenes), se salta")
    else:
        shutil.rmtree(destino, ignore_errors=True)
        try:
            import roboflow  # noqa: F401
            _descargar_con_paquete(fuente["workspace"], fuente["project"], version, "yolov8", clave, destino)
        except ImportError:
            _descargar_zip_rest(fuente["workspace"], fuente["project"], version, "yolov8", clave, destino)
        print(f"[{alias}] descargado: {len(listar_imagenes(destino))} imágenes")
    info["imagenes_descargadas"] = len(listar_imagenes(destino))
    (destino / "_fuente.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga datasets de Roboflow Universe (YOLOv8)")
    ap.add_argument("--dataset", nargs="*", help="solo estos alias")
    ap.add_argument("--forzar", action="store_true", help="vuelve a descargar aunque ya exista")
    ap.add_argument("--inspeccionar", action="store_true", help="solo consulta metadatos, no descarga")
    args = ap.parse_args()

    clave = clave_api()
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    fuentes = [f for f in FUENTES if f.get("origen", "roboflow") == "roboflow"
               and (not args.dataset or f["alias"] in args.dataset)]
    resumen, errores = [], []
    for f in fuentes:
        try:
            resumen.append(descargar(f, clave, args.forzar, args.inspeccionar))
        except Exception as e:  # noqa: BLE001 — un dataset roto no frena a los demás
            msg = _sin_clave(str(e), clave)
            errores.append({"alias": f["alias"], "error": msg})
            print(f"[{f['alias']}] ERROR: {msg}")

    ruta = DIR_RAW / "_inspeccion.json"
    previo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for info in resumen:
        previo[info["alias"]] = info
    for e in errores:
        previo[e["alias"]] = {**previo.get(e["alias"], {}), "error": e["error"]}
    ruta.write_text(json.dumps(previo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumen en {ruta}")
    if errores:
        print(f"Fallaron: {[e['alias'] for e in errores]}")


if __name__ == "__main__":
    main()
