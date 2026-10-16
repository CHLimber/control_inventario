"""Viabilidad (Detector-0): ¿cuánto rinde un YOLO preentrenado en COCO, sin entrenar, para contar bebidas?

Subcomandos (todos aceptan --raiz = repo principal, donde viven datos/ y modelos/):

  seleccionar  Arma la muestra de prueba desde los datasets de Roboflow descargados en
               <RAIZ>/datos/viabilidad/salidas/raw/ (split test/valid, >= N bebidas por foto,
               una sola copia por foto original) -> datos/viabilidad/resultados/seleccion.csv
  detectar     Corre el modelo COCO sobre la muestra con conf baja y TODAS las clases; guarda
               las detecciones crudas -> salidas/detecciones_<modelo>.csv
  evaluar      A partir de las detecciones: barrido de umbrales x variantes de clases ->
               métricas de conteo (MAE, % exacto, sesgo) y de detección (P/R con IoU>=0.5),
               análisis de latas y botellas -> resultados/*.csv
  tiempos      Mide ms/imagen en CPU (640, con calentamiento) con 1 y con N hilos
  anotar       Dibuja ~12 casos ilustrativos (real en verde, detectado en rojo) -> salidas/anotadas/
  todo         seleccionar + detectar + evaluar + tiempos + anotar

Ejemplo:
  python scripts/viabilidad_detector.py todo --raiz "<MAIN>" --modelos yolo26n.pt yolo26s.pt
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- clases COCO de interés -------------------------------------------------
COCO_BOTTLE = 39
COCO_CUP = 41
VARIANTES = {
    "bottle": {COCO_BOTTLE},
    "bottle+cup": {COCO_BOTTLE, COCO_CUP},
}
UMBRALES = (0.25, 0.4, 0.5)
IOU_MATCH = 0.5
CONF_CRUDA = 0.10  # se guardan todas las detecciones >= 0.10 y se filtra después

# --- clasificación de nombres de clase de los datasets ----------------------
PAT_LATA = re.compile(r"(^|[^a-z])(can|cans|lata|latas|tin)([^a-z]|$)|_can$|can_", re.I)
PAT_BOTELLA = re.compile(r"bottle|bottel|botella", re.I)
PAT_NEGADO = re.compile(r"^(non|no_|not_)", re.I)  # p. ej. 'NonBottle'


# Datasets cuyas clases son marcas sin formato en el nombre; se verificó a ojo que son todas botellas
# (RF100 soda-bottles: botellas PET de Coca-Cola/Fanta/Sprite dentro de una heladera).
TIPO_FORZADO = {"roboflow-100__soda-bottles": "botella"}


def tipo_de_clase(nombre: str, dataset: str = "") -> str:
    """'lata', 'botella' o 'excluir' según el nombre de clase del dataset."""
    if dataset in TIPO_FORZADO:
        return TIPO_FORZADO[dataset]
    n = nombre.lower().replace("-", "_").replace(" ", "_")
    if PAT_NEGADO.search(n):
        return "excluir"
    if PAT_LATA.search(n):
        return "lata"
    if PAT_BOTELLA.search(n):
        return "botella"
    return "excluir"


# --- utilidades -----------------------------------------------------------------
def rutas(raiz: Path, repo: Path) -> dict:
    return {
        "raw": raiz / "datos" / "viabilidad" / "salidas" / "raw",
        "salidas": raiz / "datos" / "viabilidad" / "salidas",
        "anotadas": raiz / "datos" / "viabilidad" / "salidas" / "anotadas",
        "resultados": repo / "datos" / "viabilidad" / "resultados",
        "modelos": raiz / "modelos",
    }


def leer_yaml_nombres(data_yaml: Path) -> list[str]:
    import yaml

    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nombres = d["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres)]
    return [str(x) for x in nombres]


def leer_etiquetas(lbl: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    """YOLO txt -> [(clase, x1, y1, x2, y2)] en píxeles. Acepta cajas y polígonos."""
    out = []
    if not lbl.exists():
        return out
    for linea in lbl.read_text(encoding="utf-8").splitlines():
        p = linea.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        v = list(map(float, p[1:]))
        if len(v) == 4:
            xc, yc, bw, bh = v
            x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
        else:  # polígono
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out.append((c, x1 * w, y1 * h, x2 * w, y2 * h))
    return out


def iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def emparejar(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, umbral=IOU_MATCH):
    """Emparejamiento voraz por confianza (estilo COCO). Devuelve (tp, idx_gt_por_pred)."""
    asign = -np.ones(len(pred), dtype=int)
    if len(pred) == 0 or len(gt) == 0:
        return 0, asign
    m = iou_matriz(pred, gt)
    usado = np.zeros(len(gt), bool)
    for i in np.argsort(-conf):
        cand = np.where(~usado & (m[i] >= umbral))[0]
        if len(cand):
            j = cand[np.argmax(m[i, cand])]
            usado[j] = True
            asign[i] = j
    return int(usado.sum()), asign


# --- 1. seleccionar ----------------------------------------------------------------
def cmd_seleccionar(a, R):
    from PIL import Image

    filas, clases_info = [], []
    for ds in sorted(p for p in R["raw"].iterdir() if (p / "data.yaml").exists()):
        nombres = leer_yaml_nombres(ds / "data.yaml")
        tipos = {i: tipo_de_clase(n, ds.name) for i, n in enumerate(nombres)}
        for i, n in enumerate(nombres):
            clases_info.append({"dataset": ds.name, "id": i, "clase": n, "tipo": tipos[i]})
        cand = []
        vistos = set()
        for split in a.splits:
            for img in sorted((ds / split / "images").glob("*")):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                orig = img.name.split(".rf.")[0]
                if orig in vistos:  # misma foto original (copia aumentada): no repetir
                    continue
                cajas = leer_etiquetas(ds / split / "labels" / (img.stem + ".txt"), 1, 1)
                nb = sum(tipos.get(c) == "botella" for c, *_ in cajas)
                nl = sum(tipos.get(c) == "lata" for c, *_ in cajas)
                nx = sum(tipos.get(c, "excluir") == "excluir" for c, *_ in cajas)
                if nb + nl >= a.min_obj:
                    with Image.open(img) as im:
                        w, h = im.size
                    vistos.add(orig)
                    cand.append({"dataset": ds.name, "split": split, "original": orig,
                                 "imagen": str(img.relative_to(R["raw"])).replace("\\", "/"),
                                 "ancho": w, "alto": h, "n_real": nb + nl, "n_botella": nb,
                                 "n_lata": nl, "n_excluidas": nx})
        rng = np.random.default_rng(a.semilla)
        if len(cand) > a.max_por_dataset:
            idx = sorted(rng.choice(len(cand), a.max_por_dataset, replace=False))
            cand = [cand[i] for i in idx]
        print(f"{ds.name}: {len(cand)} imágenes elegidas")
        filas += cand
    R["resultados"].mkdir(parents=True, exist_ok=True)
    sel = pd.DataFrame(filas)
    sel.to_csv(R["resultados"] / "seleccion.csv", index=False)
    pd.DataFrame(clases_info).to_csv(R["resultados"] / "clases_datasets.csv", index=False)
    print(f"Total: {len(sel)} imágenes, {sel.n_real.sum()} bebidas "
          f"({sel.n_botella.sum()} botellas, {sel.n_lata.sum()} latas)")
    return sel


# --- 2. detectar -------------------------------------------------------------------
def cargar_modelo(R, nombre: str):
    from ultralytics import YOLO

    ruta = R["modelos"] / nombre
    if ruta.exists():
        return YOLO(str(ruta))
    m = YOLO(nombre)  # se descarga solo; luego se mueve a modelos/
    descargado = Path(nombre)
    if descargado.exists():
        R["modelos"].mkdir(parents=True, exist_ok=True)
        descargado.replace(ruta)
        m = YOLO(str(ruta))
    return m


def cmd_detectar(a, R):
    import torch

    sel = pd.read_csv(R["resultados"] / "seleccion.csv")
    R["salidas"].mkdir(parents=True, exist_ok=True)
    for nombre in a.modelos:
        modelo = cargar_modelo(R, nombre)
        torch.set_num_threads(a.hilos)
        filas = []
        for _, f in sel.iterrows():
            res = modelo.predict(str(R["raw"] / f.imagen), imgsz=a.imgsz, conf=CONF_CRUDA,
                                 device="cpu", verbose=False, max_det=300)[0]
            b = res.boxes
            for (x1, y1, x2, y2), c, k in zip(b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy()):
                filas.append({"imagen": f.imagen, "clase_coco": int(k),
                              "nombre_coco": modelo.names[int(k)], "conf": float(c),
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        det = pd.DataFrame(filas)
        det.to_csv(R["salidas"] / f"detecciones_{Path(nombre).stem}.csv", index=False)
        print(f"{nombre}: {len(det)} detecciones crudas (conf>={CONF_CRUDA})")


# --- 3. evaluar --------------------------------------------------------------------
def gt_de(R, fila, tipos_ds) -> tuple[np.ndarray, list[str]]:
    p = R["raw"] / fila.imagen
    lbl = p.parent.parent / "labels" / (p.stem + ".txt")
    cajas = leer_etiquetas(lbl, fila.ancho, fila.alto)
    tipos = tipos_ds[fila.dataset]
    gt = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in cajas if tipos.get(c) in ("botella", "lata")]
    tt = [tipos[c] for c, *_ in cajas if tipos.get(c) in ("botella", "lata")]
    return np.array(gt, float).reshape(-1, 4), tt


def cmd_evaluar(a, R):
    sel = pd.read_csv(R["resultados"] / "seleccion.csv")
    cl = pd.read_csv(R["resultados"] / "clases_datasets.csv")
    tipos_ds = {ds: dict(zip(g.id, g.tipo)) for ds, g in cl.groupby("dataset")}
    gts = {f.imagen: gt_de(R, f, tipos_ds) for f in sel.itertuples()}

    por_imagen, metricas, latas = [], [], []
    for nombre in a.modelos:
        stem = Path(nombre).stem
        det = pd.read_csv(R["salidas"] / f"detecciones_{stem}.csv")
        grupos = {k: g for k, g in det.groupby("imagen")}
        vacio = det.iloc[0:0]
        # (a) qué clase COCO "ve" el modelo en cada lata/botella real (conf>=0.25, cualquier clase)
        for f in sel.itertuples():
            gt, tt = gts[f.imagen]
            g = grupos.get(f.imagen, vacio)
            g = g[g.conf >= 0.25]
            pb = g[["x1", "y1", "x2", "y2"]].to_numpy()
            m = iou_matriz(gt, pb)
            for j in range(len(gt)):
                if m.shape[1] and m[j].max() >= IOU_MATCH:
                    k = int(np.argmax(m[j]))
                    clase, conf = g.iloc[k].nombre_coco, g.iloc[k].conf
                else:
                    clase, conf = "(nada)", np.nan
                latas.append({"modelo": stem, "dataset": f.dataset, "imagen": f.imagen,
                              "tipo_real": tt[j], "clase_coco": clase, "conf": conf,
                              "orientacion": "de pie" if (gt[j, 3] - gt[j, 1]) >= (gt[j, 2] - gt[j, 0])
                              else "acostada"})
        # (b) barrido umbral x variante
        for umbral in UMBRALES:
            for var, clases in VARIANTES.items():
                for f in sel.itertuples():
                    gt, tt = gts[f.imagen]
                    g = grupos.get(f.imagen, vacio)
                    g = g[(g.conf >= umbral) & g.clase_coco.isin(clases)]
                    pb, pc = g[["x1", "y1", "x2", "y2"]].to_numpy(), g.conf.to_numpy()
                    tp, asign = emparejar(pb, pc, gt)
                    tp_b = sum(1 for j in asign if j >= 0 and tt[j] == "botella")
                    tp_l = sum(1 for j in asign if j >= 0 and tt[j] == "lata")
                    por_imagen.append({
                        "modelo": stem, "umbral": umbral, "variante": var,
                        "dataset": f.dataset, "imagen": f.imagen,
                        "n_real": len(gt), "n_det": len(pb), "error": len(pb) - len(gt),
                        "tp": tp, "fp": len(pb) - tp, "fn": len(gt) - tp,
                        "n_botella": tt.count("botella"), "n_lata": tt.count("lata"),
                        "tp_botella": tp_b, "tp_lata": tp_l})
    pi = pd.DataFrame(por_imagen)

    def resumir(g):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return pd.Series({
            "imagenes": len(g), "objetos_reales": int(g.n_real.sum()),
            "objetos_detectados": int(g.n_det.sum()),
            "MAE": g.error.abs().mean(), "MAE_rel_%": 100 * (g.error.abs() / g.n_real).mean(),
            "exacto_%": 100 * (g.error == 0).mean(), "sesgo": g.error.mean(),
            "sub_%": 100 * (g.error < 0).mean(), "sobre_%": 100 * (g.error > 0).mean(),
            "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
            "R_botellas": g.tp_botella.sum() / max(g.n_botella.sum(), 1),
            "R_latas": g.tp_lata.sum() / max(g.n_lata.sum(), 1),
        })

    claves = ["modelo", "umbral", "variante"]
    met = pi.groupby(claves).apply(resumir, include_groups=False).reset_index()
    met_ds = pi.groupby(claves + ["dataset"]).apply(resumir, include_groups=False).reset_index()
    la = pd.DataFrame(latas)
    conf_tab = (la.groupby(["modelo", "tipo_real", "clase_coco"]).size()
                  .rename("n").reset_index())
    conf_tab["%"] = 100 * conf_tab.n / conf_tab.groupby(["modelo", "tipo_real"]).n.transform("sum")

    out = R["resultados"]
    pi.to_csv(out / "por_imagen.csv", index=False)
    met.round(4).to_csv(out / "metricas.csv", index=False)
    met_ds.round(4).to_csv(out / "metricas_por_dataset.csv", index=False)
    conf_tab.round(2).to_csv(out / "latas_clase_coco.csv", index=False)
    # ¿influye la orientación? (caja más alta que ancha = "de pie")
    orient = (la.assign(es_bottle=la.clase_coco.eq("bottle"), alguna=la.clase_coco.ne("(nada)"))
                .groupby(["modelo", "dataset", "tipo_real", "orientacion"])
                .agg(n=("es_bottle", "size"), pct_como_bottle=("es_bottle", "mean"),
                     pct_con_alguna_clase=("alguna", "mean")).reset_index())
    orient[["pct_como_bottle", "pct_con_alguna_clase"]] *= 100
    orient.round(1).to_csv(out / "orientacion_clase_coco.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(met.round(3).to_string(index=False))
        print(conf_tab.round(1).to_string(index=False))
    return met


# --- 4. tiempos --------------------------------------------------------------------
def cmd_tiempos(a, R):
    import cv2
    import torch

    sel = pd.read_csv(R["resultados"] / "seleccion.csv")
    imgs = [cv2.imread(str(R["raw"] / p)) for p in sel.imagen.head(a.n_tiempo)]
    filas = []
    for nombre in a.modelos:
        modelo = cargar_modelo(R, nombre)
        # La 1.ª llamada a predict arma el predictor y ultralytics.select_device() hace
        # torch.set_num_threads(min(8, cpu-1)); por eso se fija la cantidad de hilos DESPUÉS.
        modelo.predict(imgs[0], imgsz=a.imgsz, conf=0.25, device="cpu", verbose=False)
        for hilos in a.hilos_tiempo:
            torch.set_num_threads(hilos)
            for im in imgs[: a.calentamiento]:  # calentamiento con esta cantidad de hilos
                modelo.predict(im, imgsz=a.imgsz, conf=0.25, device="cpu", verbose=False)
            assert torch.get_num_threads() == hilos, "ultralytics cambió la cantidad de hilos"
            ms, pre, inf, post = [], [], [], []
            for im in imgs:
                t0 = time.perf_counter()
                r = modelo.predict(im, imgsz=a.imgsz, conf=0.25, device="cpu", verbose=False)[0]
                ms.append(1000 * (time.perf_counter() - t0))
                pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"])
                post.append(r.speed["postprocess"])
            ms = np.array(ms)
            fila = {"modelo": Path(nombre).stem, "hilos": hilos, "torch_hilos": torch.get_num_threads(),
                    "imgsz": a.imgsz, "imagenes": len(ms), "ms_prom": ms.mean(),
                    "ms_p50": np.percentile(ms, 50), "ms_p95": np.percentile(ms, 95),
                    "ms_preproceso": np.mean(pre), "ms_inferencia": np.mean(inf),
                    "ms_postproceso": np.mean(post), "img_por_s": 1000 / ms.mean()}
            print(fila)
            filas.append(fila)
    t = pd.DataFrame(filas).round(2)
    t.to_csv(R["resultados"] / "tiempos.csv", index=False)
    info = {"cpu": _cpu(), "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    import ultralytics
    info["ultralytics"] = ultralytics.__version__
    (R["resultados"] / "entorno_tiempos.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return t


def _cpu() -> str:
    import platform
    import subprocess

    try:
        s = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return s or platform.processor()
    except Exception:
        return platform.processor()


# --- 5. anotar ---------------------------------------------------------------------
def cmd_anotar(a, R):
    import cv2

    sel = pd.read_csv(R["resultados"] / "seleccion.csv").set_index("imagen")
    cl = pd.read_csv(R["resultados"] / "clases_datasets.csv")
    tipos_ds = {ds: dict(zip(g.id, g.tipo)) for ds, g in cl.groupby("dataset")}
    pi = pd.read_csv(R["resultados"] / "por_imagen.csv")
    stem = Path(a.modelos[0]).stem
    q = pi[(pi.modelo == stem) & (pi.umbral == a.umbral_anotar) & (pi.variante == a.variante_anotar)]
    q = q.assign(abs_err=q.error.abs())
    n = a.n_anotar
    # 1/3 buenos (conteo exacto, los más poblados); el resto, repartido por dataset:
    # mitad los peores y mitad casos "típicos" (error cercano a la mediana).
    buenos = q[q.error == 0].sort_values("n_real", ascending=False).head(n // 3)
    partes = [buenos]
    resto_q = q[~q.imagen.isin(buenos.imagen)]
    dss = sorted(resto_q.dataset.unique())
    por_ds = max(1, (n - len(buenos)) // max(len(dss), 1))
    for ds in dss:
        g = resto_q[resto_q.dataset == ds]
        peores = g.sort_values("abs_err", ascending=False).head(por_ds - por_ds // 2)
        g = g[~g.imagen.isin(peores.imagen)]
        med = g.abs_err.median()
        tipicos = g.iloc[(g.abs_err - med).abs().argsort()].head(por_ds // 2)
        partes += [peores, tipicos]
    elegidos = pd.concat(partes).head(n)
    if len(elegidos) < n:  # completar con los de mayor error
        resto = q[~q.imagen.isin(elegidos.imagen)].sort_values("abs_err", ascending=False)
        elegidos = pd.concat([elegidos, resto.head(n - len(elegidos))])
    det = pd.read_csv(R["salidas"] / f"detecciones_{stem}.csv")
    clases = VARIANTES[a.variante_anotar]
    dest = R["anotadas"]
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()
    lista = []
    for k, e in enumerate(elegidos.itertuples(), 1):
        f = sel.loc[e.imagen]
        im = cv2.imread(str(R["raw"] / e.imagen))
        fila = pd.Series({**f.to_dict(), "imagen": e.imagen})
        gt, tt = gt_de(R, fila, tipos_ds)
        esc = max(1, round(max(im.shape[:2]) / 640))
        for (x1, y1, x2, y2), t in zip(gt.astype(int), tt):
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), 2 * esc)
            cv2.putText(im, t, (x1, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * esc, (0, 200, 0), esc)
        g = det[(det.imagen == e.imagen) & (det.conf >= a.umbral_anotar) & det.clase_coco.isin(clases)]
        for d in g.itertuples():
            x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), esc)
            cv2.putText(im, f"{d.nombre_coco} {d.conf:.2f}", (x1, y1 + 14 * esc),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * esc, (0, 0, 255), esc)
        txt = f"real={e.n_real} detectado={e.n_det} error={e.error:+d}"
        cv2.rectangle(im, (0, 0), (im.shape[1], 26 * esc), (0, 0, 0), -1)
        cv2.putText(im, txt, (6, 19 * esc), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * esc, (255, 255, 255), esc)
        tipo = "exacto" if e.error == 0 else ("subconteo" if e.error < 0 else "sobreconteo")
        nombre = f"{k:02d}_{tipo}_real{e.n_real}_det{e.n_det}.jpg"
        cv2.imwrite(str(dest / nombre), im)
        lista.append({"archivo": nombre, "imagen": e.imagen, "dataset": e.dataset, "caso": tipo,
                      "n_real": e.n_real, "n_det": e.n_det, "error": e.error})
    pd.DataFrame(lista).to_csv(R["resultados"] / "casos_anotados.csv", index=False)
    print(f"{len(lista)} imágenes anotadas en {dest}")


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["seleccionar", "detectar", "evaluar", "tiempos", "anotar", "todo"])
    ap.add_argument("--raiz", required=True, help="repo principal (datos pesados y modelos/)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="checkout donde se escriben resultados versionados (por defecto, el de este script)")
    ap.add_argument("--modelos", nargs="+", default=["yolo26n.pt"])
    ap.add_argument("--splits", nargs="+", default=["test", "valid"])
    ap.add_argument("--min-obj", type=int, default=3)
    ap.add_argument("--max-por-dataset", type=int, default=45)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--hilos", type=int, default=8, help="hilos de torch para 'detectar'")
    ap.add_argument("--hilos-tiempo", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--n-tiempo", type=int, default=60)
    ap.add_argument("--calentamiento", type=int, default=5)
    ap.add_argument("--n-anotar", type=int, default=12)
    ap.add_argument("--umbral-anotar", type=float, default=0.25)
    ap.add_argument("--variante-anotar", default="bottle+cup", choices=list(VARIANTES))
    a = ap.parse_args(argv)
    R = rutas(Path(a.raiz), Path(a.repo))
    pasos = ["seleccionar", "detectar", "evaluar", "tiempos", "anotar"] if a.comando == "todo" else [a.comando]
    for p in pasos:
        print(f"\n=== {p} ===")
        globals()[f"cmd_{p}"](a, R)


if __name__ == "__main__":
    main()
