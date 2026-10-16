"""Entrena y compara los clasificadores de envases (variante A): cnn, cnn_ad y mobilenet.

Lee  datos/recortes/{train,val,test}/<clase>/*.jpg  (los produce el agente Datos) y guarda en
modelos/clasificador/ (o en --salida):

  <modelo>.keras              mejor época (ModelCheckpoint + EarlyStopping), con el aumento y
                              la normalización ADENTRO: recibe píxeles 0-255
  <modelo>_metricas.json      val/test: accuracy, loss, precisión/recall por clase, matriz de
                              confusión, confusiones botella↔lata y Sprite↔7up, barrido de umbral
  <modelo>_historial.csv      loss/accuracy por época
  matriz_<modelo>_<split>.png matriz de confusión
  comparacion.csv / .md       tabla comparativa de todos los modelos entrenados en esa carpeta
  clasificador.keras          copia del MEJOR (mayor val_accuracy)  ← lo usa src/vision/clasificador.py
  clases.json                 copia de config/clases.json (orden de las salidas)
  metricas.json               resumen del elegido: tamaño de entrada, umbral sugerido, métricas

TensorBoard: logs/clasificador/<modelo>_<fecha>/   (tensorboard --logdir logs/clasificador)

--prueba hace una corrida corta (pocas imágenes por clase, 2 épocas por fase; la 1.ª incluye
compilar y leer los JPEG, la 2.ª mide la velocidad) en modelos/clasificador/prueba/ y estima
cuánto tardaría el entrenamiento completo en esta máquina.

Uso:
  python scripts/clasificador_entrenar.py --modelo todos --prueba
  python scripts/clasificador_entrenar.py --modelo mobilenet
  python scripts/clasificador_entrenar.py --modelo cnn --epocas 10 --max-por-clase 500
  python scripts/clasificador_entrenar.py --modelo todos --hilos 0        # Colab: sin límite de hilos
  python scripts/clasificador_entrenar.py --solo-comparar --actualizar-doc
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.vision.clasificador_modelos import (MODELOS, PARAMETROS, cargar_imagen,  # noqa: E402
                                             letterbox)

LOTE = 32
SEMILLA = 42
PACIENCIA = {"cnn": 5, "cnn_ad": 8, "mobilenet": 5}
MAX_POR_CLASE_PRUEBA = 16
EPOCAS_PRUEBA = 2
CACHE_RAM_MAX_GB = 2.0
PESO_MAX = 10.0
SPLITS = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
UMBRALES = [round(0.30 + 0.05 * i, 2) for i in range(14)]     # 0.30 … 0.95
PRECISION_OBJETIVO = 0.95
DOC = RAIZ / "docs" / "clasificador.md"
MARCA_INI, MARCA_FIN = "<!-- COMPARACION:INICIO -->", "<!-- COMPARACION:FIN -->"


# ----------------------------------------------------------------------------- entorno
def configurar_tf(hilos: int, semilla: int = SEMILLA) -> None:
    """Limita los hilos de TensorFlow (otros procesos usan la CPU) y fija semillas.

    Debe llamarse ANTES de cualquier operación de TensorFlow. hilos=0 → sin límite (Colab).
    """
    import keras
    import tensorflow as tf

    if hilos > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(hilos)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:      # TF ya inicializado (p. ej. dentro de pytest): se sigue igual
            pass
    keras.utils.set_random_seed(semilla)


def mantener_despierto() -> None:
    """En Windows evita que la PC se suspenda mientras dura el proceso."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


# ----------------------------------------------------------------------------- datos
def carpeta_split(dir_recortes: Path, split: str) -> Path | None:
    for nombre in SPLITS[split]:
        if (dir_recortes / nombre).is_dir():
            return dir_recortes / nombre
    return None


def listar_split(dir_recortes: Path, split: str, max_por_clase: int | None = None,
                 semilla: int = SEMILLA) -> tuple[list[str], np.ndarray]:
    """Rutas y etiquetas (índice en config/clases.json) de <recortes>/<split>/<clase>/*.jpg."""
    clases = config.nombres_clases()
    base = carpeta_split(dir_recortes, split)
    rutas: list[str] = []
    etiquetas: list[int] = []
    if base is None:
        return rutas, np.zeros(0, dtype=np.int32)
    extras = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in clases)
    if extras:
        print(f"  aviso: en {base} hay carpetas que no son clases del modelo (se ignoran): {extras}")
    rng = random.Random(semilla)
    for i, clase in enumerate(clases):
        carpeta = base / clase
        archivos = sorted(p for p in carpeta.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
            if carpeta.is_dir() else []
        if max_por_clase:
            rng.shuffle(archivos)
            archivos = archivos[:max_por_clase]
        rutas += [str(a) for a in archivos]
        etiquetas += [i] * len(archivos)
    return rutas, np.array(etiquetas, dtype=np.int32)


def crear_dataset(rutas: list[str], etiquetas: np.ndarray, tamano: int, lote: int = LOTE,
                  mezclar: bool = False, semilla: int = SEMILLA, archivo_cache: Path | None = None):
    """tf.data: lee cada JPEG y le aplica `letterbox` (el MISMO preprocesamiento de inferencia).

    Se cachea la imagen ya decodificada (uint8) para no releer los JPEG en cada época: en RAM si
    entra en ~2 GB; si no, en el archivo de caché de tf.data `archivo_cache`.
    """
    import tensorflow as tf

    def cargar(ruta: bytes) -> np.ndarray:
        return letterbox(cargar_imagen(ruta.decode("utf-8")), tamano)

    def mapear(ruta, y):
        x = tf.numpy_function(cargar, [ruta], tf.uint8)
        x.set_shape((tamano, tamano, 3))
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    ds = ds.map(mapear, num_parallel_calls=tf.data.AUTOTUNE)
    gb = len(rutas) * tamano * tamano * 3 / 1e9
    if gb <= CACHE_RAM_MAX_GB or archivo_cache is None:
        ds = ds.cache()
    else:
        archivo_cache.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(archivo_cache))
    if mezclar:
        ds = ds.shuffle(min(len(rutas), 5000), seed=semilla, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.batch(lote).prefetch(tf.data.AUTOTUNE)


def pesos_clase(etiquetas: np.ndarray, n_clases: int) -> dict[int, float]:
    """class_weight inverso a la frecuencia: una clase con la mitad de ejemplos pesa el doble.

    Con clases balanceadas todos los pesos dan ≈ 1 (no cambia nada). Se topea en PESO_MAX: una
    clase con 30 ejemplos frente a otras con 1000 pesaría ~30 y cada uno de sus lotes movería
    demasiado los pesos (entrenamiento inestable, sobreajuste a esos pocos recortes).
    """
    conteo = np.bincount(etiquetas, minlength=n_clases)
    presentes = int((conteo > 0).sum())
    return {i: min(PESO_MAX, float(len(etiquetas) / (presentes * c))) if c else 1.0
            for i, c in enumerate(conteo)}


# ----------------------------------------------------------------------------- métricas
def confusiones_clave(matriz: np.ndarray, clases: list[str]) -> dict:
    """Errores que más importan en el inventario: botella↔lata de la MISMA marca y Sprite↔7up."""
    idx = {c: i for i, c in enumerate(clases)}
    soporte = matriz.sum(axis=1)

    def par(a: str, b: str) -> dict:
        ia, ib = idx[a], idx[b]
        return {"de": a, "a": b, "n": int(matriz[ia, ib]),
                "tasa": float(matriz[ia, ib] / soporte[ia]) if soporte[ia] else None}

    marcas = sorted({c.split("_")[0] for c in clases})
    botella_lata = []
    for m in marcas:
        b, l_ = f"{m}_botella", f"{m}_lata"
        if b in idx and l_ in idx:
            botella_lata += [par(b, l_), par(l_, b)]
    sprite_7up = [par(a, b) for a, b in (("sprite_lata", "7up_lata"), ("7up_lata", "sprite_lata"),
                                         ("sprite_botella", "7up_lata"), ("7up_lata", "sprite_botella"))
                  if a in idx and b in idx]
    return {"botella_lata": botella_lata, "total_botella_lata": int(sum(p["n"] for p in botella_lata)),
            "sprite_7up": sprite_7up, "total_sprite_7up": int(sum(p["n"] for p in sprite_7up))}


def barrido_umbral(probs: np.ndarray, etiquetas: np.ndarray) -> list[dict]:
    """Para cada umbral: qué parte se acepta (cobertura) y qué accuracy tiene lo aceptado."""
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    filas = []
    for t in UMBRALES:
        ok = conf >= t
        filas.append({"umbral": t, "cobertura": float(ok.mean()),
                      "accuracy_aceptadas": float((pred[ok] == etiquetas[ok]).mean()) if ok.any() else None})
    return filas


def sugerir_umbral(barrido: list[dict], piso: float = 0.5) -> float:
    """Menor umbral ≥ `piso` cuyo accuracy sobre lo aceptado llega a PRECISION_OBJETIVO.

    Si ninguno llega, el de mejor accuracy sobre lo aceptado manteniendo al menos 50 % de
    cobertura. El piso 0.5 es el UMBRAL_CONF del contrato (no se sugiere aceptar más dudas).
    """
    candidatos = [f for f in barrido if f["umbral"] >= piso and f["accuracy_aceptadas"] is not None]
    for f in candidatos:
        if f["accuracy_aceptadas"] >= PRECISION_OBJETIVO:
            return f["umbral"]
    con_cobertura = [f for f in candidatos if f["cobertura"] >= 0.5] or candidatos
    if not con_cobertura:
        return piso
    return max(con_cobertura, key=lambda f: f["accuracy_aceptadas"])["umbral"]


def evaluar(modelo, ds, etiquetas: np.ndarray) -> dict:
    """accuracy, loss, reporte por clase, matriz de confusión, confusiones clave y barrido."""
    from sklearn.metrics import classification_report, confusion_matrix

    clases = config.nombres_clases()
    if len(etiquetas) == 0:
        return {"n": 0}
    probs = modelo.predict(ds, verbose=0)
    pred = probs.argmax(axis=1)
    p_real = np.clip(probs[np.arange(len(etiquetas)), etiquetas], 1e-7, 1.0)
    matriz = confusion_matrix(etiquetas, pred, labels=list(range(len(clases))))
    reporte = classification_report(etiquetas, pred, labels=list(range(len(clases))), target_names=clases,
                                    output_dict=True, zero_division=0)
    return {
        "n": int(len(etiquetas)),
        "accuracy": float((pred == etiquetas).mean()),
        "loss": float(-np.log(p_real).mean()),            # = sparse_categorical_crossentropy
        "macro_f1": float(reporte["macro avg"]["f1-score"]),
        "reporte": reporte,
        "matriz_confusion": matriz.tolist(),
        "confusiones_clave": confusiones_clave(matriz, clases),
        "barrido_umbral": barrido_umbral(probs, etiquetas),
    }


def graficar_matriz(matriz, clases: list[str], titulo: str, ruta: Path) -> None:
    import matplotlib

    matriz = np.asarray(matriz)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(clases)), clases, rotation=45, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("predicha")
    ax.set_ylabel("real")
    ax.set_title(titulo)
    for i in range(len(clases)):
        for j in range(len(clases)):
            if matriz[i, j]:
                ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- entrenamiento
def cronometro():
    import keras

    class Cronometro(keras.callbacks.Callback):
        """Guarda la duración (s) de cada época."""

        def __init__(self):
            super().__init__()
            self.tiempos: list[float] = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.time()

        def on_epoch_end(self, epoch, logs=None):
            self.tiempos.append(time.time() - self._t)

    return Cronometro()


def callbacks(modelo_nombre: str, ruta_keras: Path, historial: Path, dir_logs: Path,
              mejor_previo: float | None = None) -> list:
    """EarlyStopping + guardado del mejor (val_loss) + TensorBoard + historial CSV."""
    import keras

    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA[modelo_nombre],
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ruta_keras), monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=mejor_previo),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=max(2, PACIENCIA[modelo_nombre] // 2), min_lr=1e-6),
        keras.callbacks.CSVLogger(str(historial), append=True),
        keras.callbacks.TensorBoard(log_dir=str(dir_logs)),
    ]


def entrenar_modelo(nombre: str, dir_recortes: Path, salida: Path, dir_logs: Path, *,
                    epocas: int | None = None, epocas_finetune: int | None = None,
                    max_por_clase: int | None = None, prueba: bool = False,
                    pesos: str | None = "imagenet", tamano: int | None = None) -> dict:
    """Entrena un modelo, lo evalúa en val y test y guarda .keras, métricas, historial y matrices."""
    import keras

    from src.vision import clasificador_modelos as cm

    p = PARAMETROS[nombre]
    tamano = tamano or p["tamano"]
    clases = config.nombres_clases()
    if prueba:
        max_por_clase = max_por_clase or MAX_POR_CLASE_PRUEBA
    splits = {s: listar_split(dir_recortes, s, max_por_clase) for s in SPLITS}
    print(f"\n=== {nombre}  (entrada {tamano}x{tamano}) ===")
    for s, (_, y) in splits.items():
        conteo = np.bincount(y, minlength=len(clases))
        print(f"  {s:5s} {len(y):6d}  " + "  ".join(f"{c}={n}" for c, n in zip(clases, conteo)))
    if len(splits["train"][1]) == 0 or len(splits["val"][1]) == 0:
        raise SystemExit(f"No hay recortes de train o val en {dir_recortes}: falta correr el agente Datos.")

    dir_cache = Path(tempfile.gettempdir()) / f"clasificador_cache_{os.getpid()}_{nombre}"
    ds = {s: crear_dataset(r, y, tamano, mezclar=(s == "train"), archivo_cache=dir_cache / s)
          for s, (r, y) in splits.items() if len(y)}
    salida.mkdir(parents=True, exist_ok=True)
    ruta_keras = salida / f"{nombre}.keras"
    historial = salida / f"{nombre}_historial.csv"
    historial.unlink(missing_ok=True)
    ruta_keras.unlink(missing_ok=True)
    fecha = datetime.now()
    logs = dir_logs / f"{nombre}_{fecha:%Y%m%d-%H%M%S}"
    pesos_c = pesos_clase(splits["train"][1], len(clases))

    modelo = cm.construir(nombre, len(clases), pesos=pesos, tamano=tamano)
    print(f"  {modelo.count_params():,} parámetros ({sum(int(np.prod(w.shape)) for w in modelo.trainable_weights):,} entrenables)")
    n_ep = EPOCAS_PRUEBA if prueba else (epocas or p["epocas"])
    crono, crono_ft = cronometro(), cronometro()
    t0 = time.time()
    hist = modelo.fit(ds["train"], validation_data=ds["val"], epochs=n_ep, class_weight=pesos_c, shuffle=False,
                      callbacks=[*callbacks(nombre, ruta_keras, historial, logs), crono], verbose=2)
    hechas = len(hist.epoch)
    if ruta_keras.exists():
        modelo.load_weights(ruta_keras)       # la mejor época, aunque EarlyStopping no haya cortado

    if nombre == "mobilenet":
        mejor = float(np.min(hist.history["val_loss"]))
        ft = EPOCAS_PRUEBA if prueba else (epocas_finetune or p["epocas_finetune"])
        print(f"\n  Fine-tuning: últimas {p['capas_finetune']} capas de la base (sin BatchNorm), lr {p['lr_finetune']}")
        cm.habilitar_finetune(modelo, p["capas_finetune"], p["lr_finetune"])
        modelo.fit(ds["train"], validation_data=ds["val"], initial_epoch=hechas, epochs=hechas + ft,
                   class_weight=pesos_c, shuffle=False,
                   callbacks=[*callbacks(nombre, ruta_keras, historial, logs, mejor), crono_ft], verbose=2)
        modelo.load_weights(ruta_keras)
    minutos = (time.time() - t0) / 60
    modelo.save(ruta_keras)

    met = {"modelo": nombre, "tamano": tamano, "parametros": int(modelo.count_params()),
           "epocas": len(crono.tiempos) + len(crono_ft.tiempos), "minutos": round(minutos, 2),
           "seg_por_epoca": [round(t, 1) for t in crono.tiempos + crono_ft.tiempos],
           "n": {s: int(len(y)) for s, (_, y) in splits.items()},
           "class_weight": {clases[i]: round(w, 3) for i, w in pesos_c.items()},
           "prueba": prueba, "max_por_clase": max_por_clase, "pesos_base": pesos if nombre == "mobilenet" else None,
           "fecha": fecha.isoformat(timespec="seconds"), "logs": str(logs)}
    for s in ("val", "test"):
        if s in ds:
            met[s] = evaluar(modelo, ds[s], splits[s][1])
            graficar_matriz(met[s]["matriz_confusion"], clases, f"{nombre} · {s}", salida / f"matriz_{nombre}_{s}.png")
    met["umbral_sugerido"] = sugerir_umbral(met["val"]["barrido_umbral"])
    (salida / f"{nombre}_metricas.json").write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  {nombre}: {minutos:.1f} min, {met['epocas']} épocas")
    for s in ("val", "test"):
        if s in met:
            m = met[s]
            print(f"  {s:5s} accuracy {m['accuracy']:.3f}  loss {m['loss']:.3f}  macro-F1 {m['macro_f1']:.3f}  "
                  f"botella↔lata {m['confusiones_clave']['total_botella_lata']}  "
                  f"sprite↔7up {m['confusiones_clave']['total_sprite_7up']}")
    if prueba:
        imprimir_estimacion(nombre, met, crono.tiempos, crono_ft.tiempos, dir_recortes)
    keras.backend.clear_session()
    shutil.rmtree(dir_cache, ignore_errors=True)
    return met


def imprimir_estimacion(nombre: str, met: dict, t_fase1: list[float], t_fase2: list[float],
                        dir_recortes: Path) -> None:
    """Extrapola el tiempo por época de la prueba al dataset completo (lineal en nº de imágenes)."""
    n_prueba = met["n"]["train"] + met["n"]["val"]
    n_total = sum(len(listar_split(dir_recortes, s)[1]) for s in ("train", "val"))
    p = PARAMETROS[nombre]
    seg = t_fase1[-1]
    por_img = seg / max(1, n_prueba)
    print(f"  Velocidad (época {len(t_fase1)}, ya en caché): {por_img * 1000:.1f} s cada 1000 imágenes (train+val)")
    if n_total:
        ep = por_img * n_total / 60
        print(f"  Estimación con {n_total} imágenes (train+val): ~{ep:.1f} min/época × {p['epocas']} épocas "
              f"= ~{ep * p['epocas']:.0f} min como máximo (EarlyStopping suele cortar antes)")
        if t_fase2:
            ep2 = t_fase2[-1] / max(1, n_prueba) * n_total / 60
            print(f"  Fine-tuning: ~{ep2:.1f} min/época × {p['epocas_finetune']} = ~{ep2 * p['epocas_finetune']:.0f} min")
        print("  (la 1.ª época real tarda más: decodifica todos los JPEG antes de cachearlos)")


# ----------------------------------------------------------------------------- comparación
COLUMNAS = ["modelo", "tamano", "parametros", "epocas", "minutos", "val_accuracy", "val_loss", "val_macro_f1",
            "test_accuracy", "test_loss", "test_macro_f1", "test_botella_lata", "test_sprite_7up",
            "umbral_sugerido", "n_train", "prueba", "fecha"]


def fila_comparacion(met: dict) -> dict:
    v, t = met.get("val", {}), met.get("test", {})
    return {
        "modelo": met["modelo"], "tamano": met["tamano"], "parametros": met["parametros"],
        "epocas": met["epocas"], "minutos": met["minutos"],
        "val_accuracy": round(v.get("accuracy", float("nan")), 4), "val_loss": round(v.get("loss", float("nan")), 4),
        "val_macro_f1": round(v.get("macro_f1", float("nan")), 4),
        "test_accuracy": round(t["accuracy"], 4) if t.get("n") else None,
        "test_loss": round(t["loss"], 4) if t.get("n") else None,
        "test_macro_f1": round(t["macro_f1"], 4) if t.get("n") else None,
        "test_botella_lata": t["confusiones_clave"]["total_botella_lata"] if t.get("n") else None,
        "test_sprite_7up": t["confusiones_clave"]["total_sprite_7up"] if t.get("n") else None,
        "umbral_sugerido": met.get("umbral_sugerido"), "n_train": met["n"]["train"],
        "prueba": met.get("prueba", False), "fecha": met.get("fecha"),
    }


def tabla_markdown(filas: list[dict]) -> str:
    cab = ["modelo", "entrada", "parámetros", "épocas", "min", "val acc", "val loss", "test acc", "test loss",
           "test F1 macro", "bot↔lata (test)", "sprite↔7up (test)", "umbral sug.", "n train", "prueba"]
    fmt = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    lineas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for f in filas:
        lineas.append("| " + " | ".join(fmt(x) for x in (
            f["modelo"], f"{f['tamano']}px", f"{f['parametros']:,}".replace(",", "."), f["epocas"], f["minutos"],
            f["val_accuracy"], f["val_loss"], f["test_accuracy"], f["test_loss"], f["test_macro_f1"],
            f["test_botella_lata"], f["test_sprite_7up"], f["umbral_sugerido"], f["n_train"],
            "sí" if f["prueba"] else "no")) + " |")
    return "\n".join(lineas)


def comparar_y_elegir(salida: Path, actualizar_doc: bool = False) -> dict | None:
    """Arma comparacion.csv/.md con todos los <modelo>_metricas.json y copia el mejor a clasificador.keras."""
    import pandas as pd

    metricas = []
    for nombre in MODELOS:
        ruta = salida / f"{nombre}_metricas.json"
        if ruta.exists() and (salida / f"{nombre}.keras").exists():
            metricas.append(json.loads(ruta.read_text(encoding="utf-8")))
    if not metricas:
        print(f"No hay modelos entrenados en {salida}")
        return None
    filas = [fila_comparacion(m) for m in metricas]
    pd.DataFrame(filas, columns=COLUMNAS).to_csv(salida / "comparacion.csv", index=False)
    tabla = tabla_markdown(filas)
    (salida / "comparacion.md").write_text(tabla + "\n", encoding="utf-8")

    mejor = max(metricas, key=lambda m: (m["val"]["accuracy"], -m["val"]["loss"]))
    shutil.copy2(salida / f"{mejor['modelo']}.keras", salida / "clasificador.keras")
    shutil.copy2(config.ARCHIVO_CLASES, salida / "clases.json")
    resumen = {
        "modelo": mejor["modelo"], "archivo": "clasificador.keras", "tamano_entrada": mejor["tamano"],
        "entrada": "RGB uint8 0-255, letterbox a cuadrado (src/vision/clasificador_modelos.letterbox)",
        "clases": config.nombres_clases(), "umbral_sugerido": mejor["umbral_sugerido"],
        "umbral_contrato": config.UMBRAL_CONF,
        "val": {k: mejor["val"][k] for k in ("n", "accuracy", "loss", "macro_f1", "confusiones_clave", "barrido_umbral")},
        "test": ({k: mejor["test"][k] for k in ("n", "accuracy", "loss", "macro_f1", "confusiones_clave")}
                 if mejor.get("test", {}).get("n") else None),
        "prueba": mejor.get("prueba", False), "fecha": mejor["fecha"], "comparacion": filas,
    }
    (salida / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nComparación:\n" + tabla)
    print(f"\nElegido: {mejor['modelo']} → {salida / 'clasificador.keras'} (umbral sugerido {mejor['umbral_sugerido']})")
    if actualizar_doc and DOC.exists():
        texto = DOC.read_text(encoding="utf-8")
        if MARCA_INI in texto and MARCA_FIN in texto:
            ini, fin = texto.index(MARCA_INI) + len(MARCA_INI), texto.index(MARCA_FIN)
            nota = f"\n\n_Generada por `scripts/clasificador_entrenar.py` el {datetime.now():%Y-%m-%d %H:%M} desde `{salida}`._\n\n"
            DOC.write_text(texto[:ini] + nota + tabla + "\n\n" + texto[fin:], encoding="utf-8")
            print(f"Tabla actualizada en {DOC}")
    return resumen


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=[*MODELOS, "todos"], default="todos")
    ap.add_argument("--datos", type=Path, default=config.DIR_DATOS / "recortes", help="carpeta de recortes")
    ap.add_argument("--salida", type=Path, help="carpeta de modelos (defecto: modelos/clasificador[/prueba])")
    ap.add_argument("--logs", type=Path, help="carpeta de TensorBoard (defecto: logs/clasificador[/prueba])")
    ap.add_argument("--epocas", type=int, help="épocas (fase 1 en mobilenet)")
    ap.add_argument("--epocas-finetune", type=int, help="épocas de fine-tuning (mobilenet)")
    ap.add_argument("--max-por-clase", type=int, help="limitar imágenes por clase y split")
    ap.add_argument("--tamano", type=int, help="lado de la entrada (pisa el de cada modelo)")
    ap.add_argument("--prueba", action="store_true", help="corrida corta para medir tiempos")
    ap.add_argument("--hilos", type=int, default=4, help="hilos de TensorFlow (0 = sin límite, p. ej. Colab)")
    ap.add_argument("--sin-pesos", action="store_true", help="MobileNetV2 sin pesos de ImageNet (tests)")
    ap.add_argument("--solo-comparar", action="store_true", help="no entrena: rehace la comparación y elige")
    ap.add_argument("--actualizar-doc", action="store_true", help="escribe la tabla en docs/clasificador.md")
    a = ap.parse_args(argv)
    sub = "prueba" if a.prueba else ""
    salida = a.salida or (config.DIR_MODELOS / "clasificador" / sub)
    dir_logs = a.logs or (config.DIR_LOGS / "clasificador" / sub)
    if not a.solo_comparar:
        configurar_tf(a.hilos)
        mantener_despierto()
        indice = a.datos / "indice_recortes.csv"
        print(f"Recortes: {a.datos}  (indice_recortes.csv {'presente' if indice.exists() else 'NO encontrado'})")
        for nombre in (MODELOS if a.modelo == "todos" else [a.modelo]):
            entrenar_modelo(nombre, a.datos, salida, dir_logs, epocas=a.epocas, epocas_finetune=a.epocas_finetune,
                            max_por_clase=a.max_por_clase, prueba=a.prueba,
                            pesos=None if a.sin_pesos else "imagenet", tamano=a.tamano)
    comparar_y_elegir(salida, a.actualizar_doc)
    print(f"TensorBoard: tensorboard --logdir \"{dir_logs}\"")


if __name__ == "__main__":
    main()
