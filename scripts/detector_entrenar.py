"""Fine-tuning de YOLO26 (pesos COCO) para el detector de bebidas.

Variantes:
  1clase      -> datos/fusion/detector_1clase  (una clase "producto"; variante A del sistema)
  multiclase  -> datos/fusion/multiclase       (9 clases de config/clases.json; variante B)

Qué hace:
  1. Arma un data.yaml PROPIO en runs/detector/tmp/<nombre>/ (no toca datos/): rutas absolutas
     a la carpeta del dataset, y opcionalmente suma los sintéticos como carpeta extra de train.
     Con --prueba arma listas .txt con un subconjunto chico (para medir tiempos en CPU).
  2. Entrena con los hiperparámetros de HIPER (+ aumento según la variante, ver AUMENTO).
  3. Mide el tiempo de cada época (entrenamiento y validación por separado) y estima cuánto
     tardaría el entrenamiento completo -> runs/detector/<nombre>/tiempos.json
  4. Copia best.pt a DIR_MODELOS/detector/detector_<variante>.pt (salvo en --prueba).

Ejemplos (desde la raíz del repo; INVENTARIO_DATOS/INVENTARIO_MODELOS apuntan al repo principal):
  python scripts/detector_entrenar.py --variante 1clase --prueba
  python scripts/detector_entrenar.py --variante multiclase --prueba --n-prueba 256
  python scripts/detector_entrenar.py --variante 1clase --epocas 60 --device 0      # con GPU
  python scripts/detector_entrenar.py --variante 1clase --con-sinteticos
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent
if str(RAIZ_REPO) not in sys.path:
    sys.path.insert(0, str(RAIZ_REPO))

from src.config import DIR_DATOS, DIR_MODELOS, nombres_clases  # noqa: E402

VARIANTES = {"1clase": "detector_1clase", "multiclase": "multiclase"}
DIR_RUNS = DIR_MODELOS.parent / "runs" / "detector"   # <MAIN>/runs/detector
DIR_PESOS = DIR_MODELOS / "detector"
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Hiperparámetros comunes (los mismos que usa notebooks/04_detector_colab.ipynb).
HIPER = dict(
    epochs=60,          # con early stopping (patience) casi nunca se llega al final
    patience=15,        # corta si mAP@0.5:0.95 de val no mejora en 15 épocas
    imgsz=640,
    batch=16,
    optimizer="auto",   # Ultralytics elige SGD/AdamW según la cantidad de iteraciones
    cos_lr=False,
    seed=0,
    deterministic=False,
    plots=True,
)

# Aumento de datos. flipud=0 SIEMPRE: en un estante las botellas nunca están cabeza abajo.
# mosaic=1: junta 4 imágenes en una -> más objetos chicos y contextos variados por lote;
# se apaga en las últimas `close_mosaic` épocas para que el modelo termine viendo fotos "reales".
# fliplr: en 1clase el espejo es inofensivo (una botella espejada sigue siendo "producto") y duplica
# la variedad; en multiclase el texto/logo espejado no existe en la realidad y la marca se reconoce
# en parte por el texto -> se apaga para no enseñarle patrones falsos.
AUMENTO_COMUN = dict(flipud=0.0, mosaic=1.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                     degrees=0.0, translate=0.1, scale=0.5, mixup=0.0)
AUMENTO = {
    "1clase": dict(AUMENTO_COMUN, fliplr=0.5),
    "multiclase": dict(AUMENTO_COMUN, fliplr=0.0),
}


# ---------------------------------------------------------------- datos
def dir_dataset(variante: str) -> Path:
    d = DIR_DATOS / "fusion" / VARIANTES[variante]
    for s in ("train", "val"):
        if not (d / "images" / s).is_dir():
            raise FileNotFoundError(f"Falta {d / 'images' / s}: ¿terminó el agente Datos de generar la fusión?")
    return d


def nombres_variante(variante: str) -> list[str]:
    return ["producto"] if variante == "1clase" else nombres_clases()


def imagenes_de(carpeta: Path) -> list[Path]:
    return sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXT_IMG) if carpeta.is_dir() else []


def buscar_sinteticos(variante: str, raiz: Path | None = None) -> list[Path]:
    """Carpetas images/train de sintéticos. Busca, en orden:
    <sinteticos>/<detector_1clase|multiclase>/images/train, <sinteticos>/<1clase|multiclase>/images/train,
    <sinteticos>/images/train. Las etiquetas deben estar en la carpeta labels/ hermana, con los ids de
    config/clases.json (en 1clase da igual: se entrena con single_cls=True y toda caja es 'producto').
    También acepta el formato del generador (src/sinteticos): <lote>/<1clase|multiclase>/images/ sin
    subcarpeta train. Si `raiz` no es un lote, se usa el lote más reciente dentro de ella."""
    raiz = raiz or DIR_DATOS / "sinteticos"
    lotes = sorted((p for p in raiz.glob("lote_*") if p.is_dir()), key=lambda p: p.stat().st_mtime,
                   reverse=True) if raiz.is_dir() else []
    for base in [raiz] + lotes:
        for c in (base / VARIANTES[variante], base / variante, base):
            for d in (c / "images" / "train", c / "images"):
                if imagenes_de(d):
                    return [d]
    return []


def etiquetas_de(dir_imagenes: Path) -> Path:
    """Carpeta de etiquetas YOLO que corresponde a una de imágenes (convención de Ultralytics)."""
    if dir_imagenes.name == "images":
        return dir_imagenes.parent / "labels"
    return dir_imagenes.parent.parent / "labels" / dir_imagenes.name


def muestra(imgs: list[Path], n: int, semilla: int = 0) -> list[Path]:
    """Muestra aleatoria reproducible, estratificada por dataset de origen (prefijo 'alias__')."""
    if n <= 0 or n >= len(imgs):
        return imgs
    por_origen: dict[str, list[Path]] = {}
    for p in imgs:
        por_origen.setdefault(p.name.split("__")[0], []).append(p)
    rng = random.Random(semilla)
    elegidas = []
    for grupo in por_origen.values():
        k = max(1, round(n * len(grupo) / len(imgs)))
        elegidas += rng.sample(grupo, min(k, len(grupo)))
    rng.shuffle(elegidas)
    return sorted(elegidas[:n])


def limitar_por_origen(imgs: list[Path], maximo: int | None, semilla: int = 0) -> list[Path]:
    """Como mucho `maximo` imágenes de cada dataset de origen (prefijo 'alias__' del archivo).

    Sirve para que un dataset enorme y poco parecido al estante (darians_bottle: ~15 000 fotos de
    416 px con UNA botella cada una) no domine las épocas ni el tiempo de entrenamiento."""
    if not maximo:
        return imgs
    por_origen: dict[str, list[Path]] = {}
    for p in imgs:
        por_origen.setdefault(p.name.split("__")[0], []).append(p)
    rng = random.Random(semilla)
    salida = []
    for grupo in por_origen.values():
        salida += rng.sample(grupo, maximo) if len(grupo) > maximo else grupo
    return sorted(salida)


def escribir_lista(ruta: Path, imgs: list[Path]) -> str:
    ruta.write_text("\n".join(p.as_posix() for p in imgs) + "\n", encoding="utf-8")
    return ruta.as_posix()


def copiar_subconjunto(imgs: list[Path], destino: Path, split: str) -> list[Path]:
    """Copia imágenes y etiquetas (labels/ hermana de images/) a destino/{images,labels}/split."""
    di, dl = destino / "images" / split, destino / "labels" / split
    shutil.rmtree(di, ignore_errors=True)
    shutil.rmtree(dl, ignore_errors=True)
    di.mkdir(parents=True)
    dl.mkdir(parents=True)
    copiadas = []
    for p in imgs:
        lbl = etiquetas_de(p.parent) / (p.stem + ".txt")
        try:
            shutil.copy2(p, di / p.name)
            if lbl.exists():
                shutil.copy2(lbl, dl / lbl.name)
            copiadas.append(di / p.name)
        except FileNotFoundError:  # la borró el agente Datos justo ahora
            continue
    return copiadas


def armar_data_yaml(variante: str, destino: Path, n_prueba: int = 0, sinteticos: list[Path] | None = None,
                    semilla: int = 0, max_por_origen: int | None = None) -> tuple[Path, dict]:
    """Escribe destino/data.yaml y devuelve (ruta, resumen con cantidad de imágenes por split)."""
    import yaml

    d = dir_dataset(variante)
    destino.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"path": (destino if n_prueba else d).as_posix()}
    resumen = {}
    for s in ("train", "val", "test"):
        imgs = imagenes_de(d / "images" / s)
        if s == "train":
            imgs = limitar_por_origen(imgs, max_por_origen, semilla)
        if s == "train" and sinteticos:
            for extra in sinteticos:
                imgs += imagenes_de(extra)
        if not imgs:
            continue
        if n_prueba:
            # Se COPIA el subconjunto: la prueba no depende de que el fusionado esté quieto
            # (el agente Datos puede estar regenerándolo mientras tanto).
            imgs = copiar_subconjunto(muestra(imgs, n_prueba if s == "train" else max(16, n_prueba // 4),
                                              semilla), destino, s)
            cfg[s] = f"images/{s}"
        elif s == "train" and (sinteticos or max_por_origen):
            cfg[s] = escribir_lista(destino / f"{s}.txt", imgs)
        else:
            cfg[s] = f"images/{s}"
        resumen[s] = len(imgs)
    cfg["names"] = dict(enumerate(nombres_variante(variante)))
    cfg["nc"] = len(cfg["names"])
    ruta = destino / "data.yaml"
    ruta.write_text("# Generado por scripts/detector_entrenar.py (no editar a mano)\n"
                    + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return ruta, resumen


# ---------------------------------------------------------------- tiempos
class Cronometro:
    """Callbacks de Ultralytics que miden el tiempo de entrenamiento y de validación por época."""

    def __init__(self, hilos: int | None):
        self.hilos = hilos
        self.epocas: list[dict] = []
        self._t0 = self._t1 = 0.0
        self._abierta = False  # Ultralytics vuelve a llamar on_fit_epoch_end tras validar best.pt: se ignora

    def registrar(self, modelo):
        modelo.add_callback("on_pretrain_routine_start", self.fijar_hilos)
        modelo.add_callback("on_train_epoch_start", self.inicio)
        modelo.add_callback("on_train_epoch_end", self.fin_train)
        modelo.add_callback("on_fit_epoch_end", self.fin_val)

    def fijar_hilos(self, _trainer=None):
        if self.hilos:
            import torch

            torch.set_num_threads(self.hilos)

    def inicio(self, trainer):
        self.fijar_hilos()
        self._t0 = time.perf_counter()
        self._abierta = True

    def fin_train(self, trainer):
        self._t1 = time.perf_counter()

    def fin_val(self, trainer):
        t2 = time.perf_counter()
        if not self._abierta:
            return
        self._abierta = False
        self.epocas.append({"epoca": trainer.epoch + 1, "seg_train": round(self._t1 - self._t0, 1),
                            "seg_val": round(t2 - self._t1, 1)})


def estimar(cron: Cronometro, n_train: int, n_val: int, n_train_total: int, n_val_total: int,
            epocas: int) -> dict:
    """Extrapola linealmente (por imagen) el tiempo de una época al dataset completo."""
    if not cron.epocas:
        return {}
    e = cron.epocas[-1]
    s_img_train = e["seg_train"] / max(n_train, 1)
    s_img_val = e["seg_val"] / max(n_val, 1)
    epoca_total = s_img_train * n_train_total + s_img_val * n_val_total
    return {
        "seg_por_img_train": round(s_img_train, 4),
        "seg_por_img_val": round(s_img_val, 4),
        "epoca_completa_estimada_min": round(epoca_total / 60, 1),
        f"entrenamiento_{epocas}_epocas_estimado_h": round(epoca_total * epocas / 3600, 1),
    }


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", choices=list(VARIANTES), required=True)
    ap.add_argument("--modelo", default="yolo26n.pt", help="pesos iniciales (en DIR_MODELOS o ruta)")
    ap.add_argument("--epocas", type=int, default=HIPER["epochs"])
    ap.add_argument("--imgsz", type=int, default=HIPER["imgsz"])
    ap.add_argument("--batch", type=int, default=HIPER["batch"])
    ap.add_argument("--paciencia", type=int, default=HIPER["patience"])
    ap.add_argument("--fraction", type=float, default=1.0, help="fracción del train a usar (Ultralytics)")
    ap.add_argument("--max-por-origen", type=int, default=None,
                    help="tope de imágenes de train por dataset de origen (p. ej. 4000)")
    ap.add_argument("--con-sinteticos", action="store_true", help="suma datos/sinteticos/ al train")
    ap.add_argument("--sinteticos", type=Path, default=None, help="carpeta de sintéticos (por defecto DIR_DATOS/sinteticos)")
    ap.add_argument("--prueba", action="store_true", help="1 época sobre un subconjunto: mide tiempos y detecta errores")
    ap.add_argument("--n-prueba", type=int, default=256, help="imágenes de train en --prueba (val/test: n/4)")
    ap.add_argument("--device", default="cpu", help="'cpu' o '0' (GPU)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hilos", type=int, default=4, help="hilos de torch en CPU (Ultralytics pone 8)")
    ap.add_argument("--cache", default=False, help="False | ram | disk")
    ap.add_argument("--nombre", default=None, help="nombre de la corrida en runs/detector/")
    ap.add_argument("--no-instalar", action="store_true", help="no copiar best.pt a modelos/detector/")
    a = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO, settings
    from ultralytics.utils import torch_utils

    epocas = 1 if a.prueba else a.epocas
    nombre = a.nombre or f"{a.variante}_{Path(a.modelo).stem}" + ("_prueba" if a.prueba else "")
    sint = []
    if a.con_sinteticos:
        sint = buscar_sinteticos(a.variante, a.sinteticos)
        if not sint:
            print("AVISO: --con-sinteticos pero no hay imágenes en datos/sinteticos/; se entrena sin ellos.")
    data_yaml, resumen = armar_data_yaml(a.variante, DIR_RUNS / "tmp" / nombre,
                                         n_prueba=a.n_prueba if a.prueba else 0, sinteticos=sint,
                                         max_por_origen=a.max_por_origen)
    d = dir_dataset(a.variante)
    total = {s: len(imagenes_de(d / "images" / s)) for s in ("train", "val", "test")}
    train_tope = {m: len(limitar_por_origen(imagenes_de(d / "images" / "train"), m)) for m in (2000, 4000, 8000)}
    print(f"data.yaml: {data_yaml}\nimágenes usadas: {resumen} | dataset completo: {total}"
          + (f" | sintéticos: {[str(s) for s in sint]}" if sint else ""))

    settings.update({"tensorboard": True})  # eventos en runs/detector/<nombre>/ -> tensorboard --logdir
    if a.device == "cpu" and a.hilos:
        torch_utils.NUM_THREADS = a.hilos   # select_device() hace torch.set_num_threads(NUM_THREADS)
        torch.set_num_threads(a.hilos)

    ruta_modelo = Path(a.modelo) if Path(a.modelo).exists() else DIR_MODELOS / a.modelo
    modelo = YOLO(str(ruta_modelo))
    cron = Cronometro(a.hilos if a.device == "cpu" else None)
    cron.registrar(modelo)

    params = dict(HIPER, **AUMENTO[a.variante])
    params.update(data=str(data_yaml), epochs=epocas, imgsz=a.imgsz, batch=a.batch, patience=a.paciencia,
                  fraction=a.fraction, device=a.device, workers=a.workers, cache=a.cache,
                  single_cls=(a.variante == "1clase"), project=str(DIR_RUNS), name=nombre, exist_ok=True)
    print("Parámetros:", json.dumps(params, ensure_ascii=False))
    t0 = time.perf_counter()
    modelo.train(**params)
    seg_total = time.perf_counter() - t0

    save_dir = Path(modelo.trainer.save_dir)
    n_train = int(resumen["train"] * a.fraction)
    info = {
        "variante": a.variante, "modelo": ruta_modelo.name, "prueba": a.prueba, "device": a.device,
        "hilos": a.hilos, "workers": a.workers, "imgsz": a.imgsz, "batch": a.batch, "epocas": epocas,
        "imagenes_usadas": resumen, "dataset_completo": total, "seg_total": round(seg_total, 1),
        "epocas_medidas": cron.epocas,
        "estimacion_cpu": estimar(cron, n_train, resumen.get("val", 0), total["train"], total.get("val", 0),
                                  HIPER["epochs"]) if a.device == "cpu" else {},
        "estimacion_cpu_con_tope_por_origen": {
            m: estimar(cron, n_train, resumen.get("val", 0), n, total.get("val", 0), HIPER["epochs"])
            for m, n in train_tope.items()} if a.device == "cpu" else {},
        "train_con_tope_por_origen": train_tope,
        "torch": torch.__version__, "cuda": torch.cuda.is_available(),
    }
    (save_dir / "tiempos.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))

    best = save_dir / "weights" / "best.pt"
    if best.exists() and not a.prueba and not a.no_instalar:
        DIR_PESOS.mkdir(parents=True, exist_ok=True)
        destino = DIR_PESOS / f"detector_{a.variante}.pt"
        shutil.copy2(best, destino)
        print(f"Pesos instalados en {destino}")
    return save_dir


if __name__ == "__main__":
    main()
