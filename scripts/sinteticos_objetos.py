"""Construye la biblioteca de objetos sin fondo para los sintéticos (cache en DIR_DATOS/sinteticos/_objetos/).

Lee SOLO el split train del dataset fusionado multiclase, filtra cajas (envase completo, parado, poco
tapado), quita el fondo con rembg y guarda PNG RGBA + indice.csv + rechazos.csv. Es incremental: lo que
ya está en el índice no se vuelve a procesar (rembg es lo lento).

Ejemplo:
  set INVENTARIO_DATOS=<MAIN>\\datos
  python scripts/sinteticos_objetos.py --max-por-clase 400 --procesos 4 --modelo u2net
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import DIR_DATOS, cargar_clases  # noqa: E402
from src.sinteticos import objetos as obj  # noqa: E402

_SEG = None
_CACHE: Path | None = None


def _iniciar(modelo: str, cache: str) -> None:
    global _SEG, _CACHE
    _SEG = obj.segmentador_rembg(modelo)
    _CACHE = Path(cache)


def _procesar(c: obj.Candidato) -> dict:
    try:
        img, motivo, cob = obj.extraer_objeto(c, _SEG)
    except (OSError, ValueError) as e:  # p. ej. la fusión se regeneró mientras corríamos
        img, motivo, cob = None, f"error:{type(e).__name__}", 0.0
    fila = {"clase": c.clase, "formato": c.formato, "origen": f"{c.imagen.name}|{','.join(map(str, c.caja))}"}
    if img is None:
        fila["motivo"] = motivo
        return fila
    rel = Path(c.clase) / obj.nombre_cache(c)
    (_CACHE / c.clase).mkdir(parents=True, exist_ok=True)
    img.save(_CACHE / rel)
    fila.update(archivo=rel.as_posix(), ancho=img.width, alto=img.height, cobertura=f"{cob:.3f}")
    return fila


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", type=Path, default=DIR_DATOS / "fusion" / "multiclase",
                    help="árbol YOLO multiclase (se usa SOLO images/train y labels/train)")
    ap.add_argument("--cache", type=Path, default=DIR_DATOS / "sinteticos" / "_objetos")
    ap.add_argument("--max-por-clase", type=int, default=400)
    ap.add_argument("--modelo", default="u2net", help="modelo de rembg: u2net (mejor) | u2netp (rápido)")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--rehacer", action="store_true", help="borra la cache y vuelve a pasar todo por rembg")
    a = ap.parse_args()
    if a.rehacer and a.cache.exists():
        import shutil
        shutil.rmtree(a.cache)

    clases = cargar_clases()["clases"]
    t0 = time.time()
    cands = obj.candidatos_yolo(a.fuente / "images" / "train", a.fuente / "labels" / "train", clases)
    print(f"{len(cands)} cajas en train; motivos del filtro geométrico:")
    for (cl, m), n in sorted(Counter((c.clase, c.motivo or "OK") for c in cands).items()):
        print(f"  {cl:15s} {m:18s} {n}")
    elegidos = obj.seleccionar(cands, a.max_por_clase, a.semilla)

    previos = obj.leer_indice(a.cache)
    ya = {f["origen"] for f in previos}
    rechazos_previos = []
    ruta_rech = a.cache / "rechazos.csv"
    if ruta_rech.exists():
        with ruta_rech.open(encoding="utf-8", newline="") as f:
            rechazos_previos = list(csv.DictReader(f))
    ya |= {f["origen"] for f in rechazos_previos}
    pendientes = [c for c in elegidos if f"{c.imagen.name}|{','.join(map(str, c.caja))}" not in ya]
    print(f"elegidos {len(elegidos)}; ya procesados {len(elegidos) - len(pendientes)}; pendientes {len(pendientes)}")

    a.cache.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with Pool(a.procesos, initializer=_iniciar, initargs=(a.modelo, str(a.cache))) as pool:
        res = list(pool.imap_unordered(_procesar, pendientes, chunksize=4))
    dt = time.time() - t1
    ok = [r for r in res if "archivo" in r]
    errores = [r for r in res if r.get("motivo", "").startswith("error")]
    rech = [r for r in res if "archivo" not in r and r not in errores]
    if errores:
        print(f"AVISO: {len(errores)} recortes con error de lectura (se reintentan en la próxima corrida)")
    # se re-aplica el control de color a TODO el índice (las reglas pueden haber cambiado)
    final = []
    for f in previos + ok:
        ruta = a.cache / f["archivo"]
        if not ruta.exists():
            continue
        motivo = obj.motivo_color(f["clase"], np.asarray(Image.open(ruta).convert("RGBA")))
        if motivo:
            rech.append({**f, "motivo": motivo})   # el PNG queda en disco, solo sale del índice
        else:
            final.append(f)
    print(f"control de color: {len(previos) + len(ok) - len(final)} objetos descartados")
    previos, ok = final, []
    obj.escribir_indice(a.cache, final)
    with ruta_rech.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "formato", "origen", "motivo"])
        w.writeheader()
        for r in rechazos_previos + rech:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    if pendientes:
        print(f"rembg ({a.modelo}, {a.procesos} procesos): {len(pendientes)} recortes en {dt:.0f} s "
              f"= {len(pendientes) / dt:.2f} recortes/s")
    print("rechazos de máscara:", dict(Counter(r["motivo"].split("_0")[0] for r in rech)))
    total = Counter(f["clase"] for f in previos + ok)
    print("biblioteca final por clase:")
    for c in clases:
        print(f"  {c['clase']:15s} {total.get(c['clase'], 0)}")
    print(f"listo en {time.time() - t0:.0f} s -> {a.cache}")


if __name__ == "__main__":
    main()
