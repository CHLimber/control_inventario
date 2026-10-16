"""Evaluación del detector entrenado (Fase 2).

Subcomandos (todos aceptan --variante 1clase|multiclase y --pesos; por defecto
DIR_MODELOS/detector/detector_<variante>.pt):

  metricas   mAP@0.5, mAP@0.5:0.95, P y R de Ultralytics en val y test (+ matriz de confusión,
             curvas PR/F1). -> runs/detector/eval_<variante>/metricas.json
  conteo     Error de conteo por imagen (MAE, % exacto, sesgo) en el split test del fusionado,
             total y por dataset de origen; en multiclase también por clase.
             -> runs/detector/eval_<variante>/conteo_test_por_imagen.csv, conteo_test.csv
  seleccion  Misma selección de 90 fotos de la viabilidad (datos/viabilidad/resultados/seleccion.csv),
             para comparar con la línea base COCO (MAE 9.54, 12 % exactos, P 0.99, R 0.20).
             Antes verifica FUGA: si esas fotos (o copias aumentadas de la misma foto) están en el
             train del fusionado, el número es optimista; se reporta también sin las fotos con fuga.
  fuga       Solo la verificación de fuga (no necesita pesos).
  curvas     Gráfico de pérdidas y métricas por época desde results.csv de una corrida.
  todo       metricas + conteo + seleccion

Ejemplo:
  python scripts/detector_evaluar.py todo --variante 1clase
  python scripts/detector_evaluar.py curvas --corrida runs/detector/1clase_yolo26n
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parent.parent
if str(RAIZ_REPO) not in sys.path:
    sys.path.insert(0, str(RAIZ_REPO))

from scripts.detector_entrenar import (DIR_PESOS, DIR_RUNS, armar_data_yaml, dir_dataset,  # noqa: E402
                                       imagenes_de, nombres_variante)
from src.config import DIR_DATOS, DIR_MODELOS, UMBRAL_DET  # noqa: E402

DIR_VIAB = DIR_DATOS / "viabilidad"
LINEA_BASE = {"modelo": "yolo26n COCO (bottle), sin entrenar", "MAE": 9.54, "exacto_%": 12.2,
              "P": 0.991, "R": 0.202}


def dir_eval(variante: str) -> Path:
    d = DIR_RUNS / f"eval_{variante}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resumir_conteo(err: pd.Series, n_real: pd.Series) -> dict:
    return {"imagenes": int(len(err)), "objetos_reales": int(n_real.sum()),
            "MAE": round(float(err.abs().mean()), 3),
            "MAE_rel_%": round(float(100 * (err.abs() / n_real.clip(lower=1)).mean()), 1),
            "exacto_%": round(float(100 * (err == 0).mean()), 1),
            "sesgo": round(float(err.mean()), 3)}


# ------------------------------------------------------------------ metricas
def cmd_metricas(a):
    from ultralytics import YOLO

    out = dir_eval(a.variante)
    data_yaml, _ = armar_data_yaml(a.variante, DIR_RUNS / "tmp" / f"eval_{a.variante}")
    modelo = YOLO(str(a.pesos))
    res = {}
    for split in a.splits:
        m = modelo.val(data=str(data_yaml), split=split, imgsz=a.imgsz, batch=a.batch, device=a.device,
                       workers=a.workers, plots=True, project=str(out), name=f"val_{split}{a.sufijo}", exist_ok=True,
                       verbose=False)
        b = m.box
        fila = {"mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4),
                "P": round(float(b.mp), 4), "R": round(float(b.mr), 4)}
        if len(m.names) > 1:
            fila["mAP50_95_por_clase"] = {m.names[int(c)]: round(float(v), 4)
                                          for c, v in zip(b.ap_class_index, b.maps[b.ap_class_index])}
            cm = m.confusion_matrix.matrix  # filas = predicho, columnas = real (+ fondo al final)
            nombres = [m.names[i] for i in range(len(m.names))] + ["fondo"]
            pd.DataFrame(cm, index=nombres, columns=nombres).to_csv(out / f"confusion_{split}{a.sufijo}.csv")
        res[split] = fila
        print(split, fila)
    (out / f"metricas{a.sufijo}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


# ------------------------------------------------------------------ conteo
def filtrar(cajas, clases):
    """Deja solo las cajas de esas clases (p. ej. 'bottle' para medir la línea base COCO)."""
    return [c for c in cajas if c.clase in clases] if clases else cajas


def cajas_reales(lbl: Path) -> list[int]:
    if not lbl.exists():
        return []
    return [int(float(l.split()[0])) for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_conteo(a):
    from src.vision.detector import Detector

    out = dir_eval(a.variante)
    d = dir_dataset(a.variante)
    nombres = nombres_variante(a.variante)
    det = Detector(a.pesos, conf=a.conf, imgsz=a.imgsz, hilos=a.hilos)
    imgs = imagenes_de(d / "images" / a.split)
    if a.origen:
        imgs = [p for p in imgs if p.name.split("__")[0] in a.origen]
    if a.n_max:
        imgs = imgs[:: max(1, len(imgs) // a.n_max)][: a.n_max]
    filas = []
    for i, p in enumerate(imgs):
        reales = cajas_reales(d / "labels" / a.split / (p.stem + ".txt"))
        cajas = filtrar(det.detectar(p), a.filtrar_clase)
        fila = {"imagen": p.name, "origen": p.name.split("__")[0], "n_real": len(reales), "n_det": len(cajas)}
        if a.variante == "multiclase":
            pred = [c.clase for c in cajas]
            fila["error_por_clase_abs"] = sum(abs(pred.count(n) - reales.count(k)) for k, n in enumerate(nombres))
        filas.append(fila)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(imgs)}")
    pi = pd.DataFrame(filas)
    pi["error"] = pi.n_det - pi.n_real
    pi.to_csv(out / f"conteo_{a.split}{a.sufijo}_por_imagen.csv", index=False)
    resumen = [{"grupo": "TOTAL", **resumir_conteo(pi.error, pi.n_real)}]
    for origen, g in pi.groupby("origen"):
        resumen.append({"grupo": origen, **resumir_conteo(g.error, g.n_real)})
    tabla = pd.DataFrame(resumen)
    if a.variante == "multiclase":
        tabla["MAE_suma_por_clase"] = [round(pi.error_por_clase_abs.mean(), 3)] + [
            round(g.error_por_clase_abs.mean(), 3) for _, g in pi.groupby("origen")]
    tabla.to_csv(out / f"conteo_{a.split}{a.sufijo}.csv", index=False)
    print(tabla.to_string(index=False))
    return tabla


# ------------------------------------------------------------------ fuga
def _norm(original: str) -> str:
    return re.sub(r"_(jpg|jpeg|png|bmp)$", "", original, flags=re.I)


def verificar_fuga(variante: str) -> pd.DataFrame:
    """Para cada foto de la selección de viabilidad: en qué split del fusionado quedó ella (o sus
    copias aumentadas de la misma foto original / mismo grupo)."""
    sel = pd.read_csv(DIR_VIAB / "resultados" / "seleccion.csv")
    idx = pd.read_csv(DIR_DATOS / "fusion" / "_indice_imagenes.csv",
                      usecols=["proyecto", "ruta", "id_original", "foto", "grupo", "split"])
    idx["archivo"] = idx.ruta.str.split("/").str[-1]
    idx["proy"] = idx.proyecto.str.replace("/", "__", regex=False)
    por_archivo = idx.set_index(["proy", "archivo"])
    filas = []
    for f in sel.itertuples():
        archivo = f.imagen.split("/")[-1]
        try:
            fila_idx = por_archivo.loc[(f.dataset, archivo)]
            if isinstance(fila_idx, pd.DataFrame):
                fila_idx = fila_idx.iloc[0]
            grupo, split_archivo = fila_idx.grupo, fila_idx.split
        except KeyError:  # otra versión del dataset: se busca por foto original
            cand = idx[(idx.proy == f.dataset) & (idx.id_original == _norm(f.original))]
            grupo = cand.grupo.iloc[0] if len(cand) else None
            split_archivo = "no_encontrada" if not len(cand) else "otra_copia"
        splits_grupo = set(idx.split[idx.grupo == grupo]) if grupo is not None else set()
        filas.append({"imagen": f.imagen, "dataset": f.dataset, "split_viabilidad": f.split,
                      "split_fusion": split_archivo, "splits_del_grupo": ";".join(sorted(splits_grupo)),
                      "fuga": "train" in splits_grupo})
    fu = pd.DataFrame(filas)
    out = dir_eval(variante)
    fu.to_csv(out / "fuga_seleccion.csv", index=False)
    print(fu.groupby(["dataset", "split_fusion"]).size().rename("fotos").to_string())
    print(f"Fotos de la selección con la misma foto en TRAIN del fusionado: {int(fu.fuga.sum())} de {len(fu)}")
    return fu


def cmd_fuga(a):
    return verificar_fuga(a.variante)


# ------------------------------------------------------------------ seleccion
def cmd_seleccion(a):
    from scripts.viabilidad_detector import emparejar, gt_de
    from src.vision.detector import Detector

    out = dir_eval(a.variante)
    fu = verificar_fuga(a.variante).set_index("imagen")
    R = {"raw": DIR_VIAB / "salidas" / "raw"}
    sel = pd.read_csv(DIR_VIAB / "resultados" / "seleccion.csv")
    cl = pd.read_csv(DIR_VIAB / "resultados" / "clases_datasets.csv")
    tipos_ds = {ds: dict(zip(g.id, g.tipo)) for ds, g in cl.groupby("dataset")}
    det = Detector(a.pesos, conf=a.conf, imgsz=a.imgsz, hilos=a.hilos)
    filas = []
    for f in sel.itertuples():
        gt, _ = gt_de(R, f, tipos_ds)
        cajas = filtrar(det.detectar(R["raw"] / f.imagen), a.filtrar_clase)
        pb = np.array([[c.x, c.y, c.x + c.w, c.y + c.h] for c in cajas], float).reshape(-1, 4)
        pc = np.array([c.conf_det for c in cajas])
        tp, _ = emparejar(pb, pc, gt)
        filas.append({"imagen": f.imagen, "dataset": f.dataset, "n_real": len(gt), "n_det": len(pb),
                      "tp": tp, "fuga": bool(fu.loc[f.imagen, "fuga"])})
    pi = pd.DataFrame(filas)
    pi["error"] = pi.n_det - pi.n_real
    pi.to_csv(out / f"seleccion{a.sufijo}_por_imagen.csv", index=False)

    def fila(nombre, g):
        tp, nd, nr = g.tp.sum(), g.n_det.sum(), g.n_real.sum()
        return {"subconjunto": nombre, **resumir_conteo(g.error, g.n_real),
                "P": round(tp / nd, 3) if nd else 0.0, "R": round(tp / nr, 3) if nr else 0.0}

    tabla = [fila("todas (90)", pi), fila("sin fuga", pi[~pi.fuga])]
    tabla += [fila(ds, g) for ds, g in pi.groupby("dataset")]
    tabla = pd.DataFrame(tabla)
    tabla.to_csv(out / f"seleccion{a.sufijo}.csv", index=False)
    print("Línea base:", LINEA_BASE)
    print(tabla.to_string(index=False))
    return tabla


# ------------------------------------------------------------------ curvas
def graficar_curvas(results_csv: Path, salida: Path | None = None):
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    perdidas = [c for c in df.columns if c.startswith(("train/", "val/"))]
    metricas = [c for c in df.columns if c.startswith("metrics/")]
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in perdidas:
        ejes[0].plot(df.epoch, df[c], label=c, ls="-" if c.startswith("train/") else "--")
    ejes[0].set(title="Pérdidas (— train, -- val)", xlabel="época")
    ejes[0].legend(fontsize=7)
    for c in metricas:
        ejes[1].plot(df.epoch, df[c], label=c.replace("metrics/", ""))
    ejes[1].set(title="Métricas en val", xlabel="época", ylim=(0, 1))
    ejes[1].legend(fontsize=7)
    fig.tight_layout()
    salida = salida or results_csv.with_name("curvas_es.png")
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return salida


def cmd_curvas(a):
    ruta = Path(a.corrida) / "results.csv"
    print("Gráfico:", graficar_curvas(ruta))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["metricas", "conteo", "seleccion", "fuga", "curvas", "todo"])
    ap.add_argument("--variante", choices=["1clase", "multiclase"], default="1clase")
    ap.add_argument("--pesos", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--split", default="test", help="split para el conteo")
    ap.add_argument("--conf", type=float, default=UMBRAL_DET, help="umbral para contar")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=0, help="limitar imágenes del conteo (prueba rápida)")
    ap.add_argument("--origen", nargs="*", default=None,
                    help="conteo solo de esos datasets de origen (prefijo del archivo, p. ej. rosd_final_hr)")
    ap.add_argument("--filtrar-clase", nargs="*", default=None,
                    help="contar solo esas clases (línea base COCO: --pesos yolo26n.pt --filtrar-clase bottle)")
    ap.add_argument("--sufijo", default="", help="sufijo de los archivos de salida")
    ap.add_argument("--corrida", default=None, help="carpeta de la corrida (para 'curvas')")
    a = ap.parse_args(argv)
    if a.pesos is None:
        a.pesos = DIR_PESOS / f"detector_{a.variante}.pt"
    elif not a.pesos.exists() and (DIR_MODELOS / a.pesos).exists():
        a.pesos = DIR_MODELOS / a.pesos  # p. ej. --pesos yolo26n.pt
    if a.comando == "curvas":
        return cmd_curvas(a)
    if a.comando == "fuga":
        return cmd_fuga(a)
    if not a.pesos.exists():
        sys.exit(f"No existen los pesos {a.pesos}. Entrená primero (scripts/detector_entrenar.py o Colab).")
    if a.comando in ("metricas", "todo"):
        cmd_metricas(a)
    if a.comando in ("conteo", "todo"):
        cmd_conteo(a)
    if a.comando in ("seleccion", "todo"):
        cmd_seleccion(a)


if __name__ == "__main__":
    main()
