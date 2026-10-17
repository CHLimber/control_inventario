"""Genera un lote de estantes sintéticos listo para Ultralytics.

Estructura de salida (``<salida>`` = DIR_DATOS/sinteticos/<nombre_lote>):

    multiclase/images/*.jpg   multiclase/labels/*.txt   multiclase/data.yaml   multiclase/data_mixto.yaml
    1clase/images/*.jpg       1clase/labels/*.txt       1clase/data.yaml       1clase/data_mixto.yaml
    conteos.csv   (imagen + 8 clases + total: la verdad EXACTA de cada imagen)
    meta.csv      (modo, vista, luz, fondo... de cada imagen, para analizar errores)
    resumen.json  (parámetros, semilla, tiempo, imágenes/segundo)

Ultralytics busca las etiquetas reemplazando ``/images/`` por ``/labels/`` en la ruta de cada imagen,
por eso cada variante es un árbol propio. Las imágenes de ``1clase/images`` son enlaces duros (hard
links) a las de ``multiclase/images``: no ocupan el doble de disco (si el disco no lo permite, se copian).

Reproducible: la imagen ``i`` usa ``np.random.default_rng([semilla, i])``, así que el resultado no
depende de cuántos procesos se usen.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.sinteticos.composicion import Parametros, componer, guardar_jpg

_G: dict = {}


def _iniciar(biblioteca, clases, fondos_reales, params, salida) -> None:
    _G.update(bib=biblioteca, clases=clases, fondos=fondos_reales, p=params, salida=Path(salida))


def nombre_imagen(semilla: int, i: int) -> str:
    return f"sint_s{semilla}_{i:05d}"


def lineas_yolo(cajas, ancho: int, alto: int, una_clase: bool) -> list[str]:
    filas = []
    for cid, x1, y1, x2, y2 in cajas:
        x1, x2 = max(0.0, x1), min(float(ancho), x2)
        y1, y2 = max(0.0, y1), min(float(alto), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto
        w, h = (x2 - x1) / ancho, (y2 - y1) / alto
        filas.append(f"{0 if una_clase else int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return filas


def generar_una(i: int, semilla: int) -> dict:
    rng = np.random.default_rng([semilla, i])
    esc = componer(rng, _G["bib"], _G["clases"], _G["fondos"], _G["p"])
    nombre = nombre_imagen(semilla, i)
    sal: Path = _G["salida"]
    alto, ancho = esc.imagen.shape[:2]
    calidad = int(rng.integers(70, 96))
    ruta_mc = sal / "multiclase" / "images" / f"{nombre}.jpg"
    guardar_jpg(ruta_mc, esc.imagen, calidad)
    ruta_1c = sal / "1clase" / "images" / f"{nombre}.jpg"
    if ruta_1c.exists():
        ruta_1c.unlink()
    try:
        os.link(ruta_mc, ruta_1c)
    except OSError:
        shutil.copyfile(ruta_mc, ruta_1c)
    mc = lineas_yolo(esc.cajas, ancho, alto, una_clase=False)
    uc = lineas_yolo(esc.cajas, ancho, alto, una_clase=True)
    (sal / "multiclase" / "labels" / f"{nombre}.txt").write_text("\n".join(mc) + ("\n" if mc else ""), encoding="utf-8")
    (sal / "1clase" / "labels" / f"{nombre}.txt").write_text("\n".join(uc) + ("\n" if uc else ""), encoding="utf-8")
    nombres = [c["clase"] for c in _G["clases"]]
    conteo = {c: 0 for c in nombres}
    for linea in mc:
        conteo[nombres[int(linea.split()[0])]] += 1
    return {"imagen": f"{nombre}.jpg", **conteo, "total": sum(conteo.values()),
            "_meta": {"imagen": f"{nombre}.jpg", "ancho": ancho, "alto": alto, "jpg_calidad": calidad, **esc.meta}}


def _generar_una_tupla(args) -> dict:
    return generar_una(*args)


def _yaml(ruta: Path, path: Path, train, val, test, names: dict[int, str], comentario: str) -> None:
    def fmt(x):
        if isinstance(x, list):
            return "\n" + "\n".join(f"  - {Path(v).as_posix()}" for v in x)
        return f" {Path(x).as_posix() if os.path.isabs(str(x)) else x}"

    lineas = [f"# {comentario}", f"path: {path.as_posix()}", f"train:{fmt(train)}", f"val:{fmt(val)}"]
    if test is not None:
        lineas.append(f"test:{fmt(test)}")
    lineas.append(f"nc: {len(names)}")
    lineas.append("names:")
    lineas += [f"  {k}: {v}" for k, v in names.items()]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def escribir_yamls(salida: Path, clases: list[dict], dir_fusion: Path | None) -> None:
    nombres_mc = {c["id"]: c["clase"] for c in clases}
    for arbol, names, fusion_sub in (("multiclase", nombres_mc, "multiclase"), ("1clase", {0: "producto"}, "detector_1clase")):
        base = salida / arbol
        _yaml(base / "data.yaml", base, "images", "images", None, names,
              "Solo sintéticos (val = las mismas imágenes: SOLO para probar que entrena). Validar SIEMPRE con fotos reales.")
        if dir_fusion is not None:
            f = dir_fusion / fusion_sub
            _yaml(base / "data_mixto.yaml", f,
                  [f / "images" / "train", base / "images"], f / "images" / "val", f / "images" / "test", names,
                  "Entrenamiento mixto: train real + sintéticos; val/test SOLO reales.")


def generar_lote(salida: Path, n: int, semilla: int, biblioteca: dict[str, list], clases: list[dict],
                 fondos_reales: list, params: Parametros | None = None, procesos: int = 1,
                 dir_fusion: Path | None = None) -> dict:
    params = params or Parametros()
    for arbol in ("multiclase", "1clase"):
        for sub in ("images", "labels"):
            (salida / arbol / sub).mkdir(parents=True, exist_ok=True)
    bib = {k: [str(p) for p in v] for k, v in biblioteca.items()}
    fondos = [str(f) if not isinstance(f, np.ndarray) else f for f in fondos_reales]
    tareas = [(i, semilla) for i in range(n)]
    t0 = time.time()
    if procesos <= 1:
        _iniciar(bib, clases, fondos, params, salida)
        filas = [generar_una(i, s) for i, s in tareas]
    else:
        with Pool(procesos, initializer=_iniciar, initargs=(bib, clases, fondos, params, salida)) as pool:
            filas = pool.map(_generar_una_tupla, tareas, chunksize=8)
    dt = time.time() - t0
    nombres = [c["clase"] for c in clases]
    with (salida / "conteos.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imagen", *nombres, "total"])
        w.writeheader()
        for fila in filas:
            w.writerow({k: fila[k] for k in w.fieldnames})
    metas = [fila["_meta"] for fila in filas]
    with (salida / "meta.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()) if metas else ["imagen"])
        w.writeheader()
        w.writerows(metas)
    escribir_yamls(salida, clases, dir_fusion)
    totales = {c: int(sum(f[c] for f in filas)) for c in nombres}
    resumen = {"n": n, "semilla": semilla, "procesos": procesos, "segundos": round(dt, 1),
               "imagenes_por_segundo": round(n / dt, 2) if dt > 0 else None,
               "objetos_por_clase": totales, "imagenes_vacias": sum(1 for f in filas if f["total"] == 0),
               "clases_sin_objetos": [c for c in nombres if not biblioteca.get(c)],
               "biblioteca": {c: len(biblioteca.get(c, [])) for c in nombres},
               "parametros": asdict(params)}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen
