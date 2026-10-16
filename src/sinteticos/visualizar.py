"""Figuras para revisar un lote a ojo: grilla de muestras con cajas y distribución por clase."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.sinteticos.composicion import dibujar_cajas


def leer_cajas_yolo(ruta: Path, ancho: int, alto: int):
    cajas = []
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            p = linea.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                cajas.append((c, (cx - w / 2) * ancho, (cy - h / 2) * alto, (cx + w / 2) * ancho, (cy + h / 2) * alto))
    return cajas


def grilla(dir_lote: Path, salida: Path, nombres: list[str], n: int = 12, columnas: int = 4,
           lado: int = 480, semilla: int = 0) -> Path:
    """Muestra ``n`` imágenes del árbol multiclase con sus cajas y el conteo exacto como título."""
    imgs = sorted((dir_lote / "multiclase" / "images").glob("*.jpg"))
    rng = np.random.default_rng(semilla)
    elegidas = [imgs[i] for i in sorted(rng.choice(len(imgs), size=min(n, len(imgs)), replace=False))]
    celdas = []
    for ruta in elegidas:
        im = cv2.cvtColor(cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        cajas = leer_cajas_yolo(dir_lote / "multiclase" / "labels" / (ruta.stem + ".txt"), w, h)
        dib = dibujar_cajas(im, cajas, nombres)
        esc = lado / max(h, w)
        dib = cv2.resize(dib, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
        celda = np.full((lado + 22, lado, 3), 255, np.uint8)
        y0, x0 = 22 + (lado - dib.shape[0]) // 2, (lado - dib.shape[1]) // 2
        celda[y0:y0 + dib.shape[0], x0:x0 + dib.shape[1]] = dib
        txt = f"{ruta.stem[-5:]}  n={len(cajas)}"
        cv2.putText(celda, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        celdas.append(celda)
    while len(celdas) % columnas:
        celdas.append(np.full_like(celdas[0], 255))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]
    lam = np.vstack(filas)
    salida.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(lam, cv2.COLOR_RGB2BGR))
    salida.write_bytes(buf.tobytes())
    return salida


def distribucion(dir_lote: Path, salida: Path, nombres: list[str]) -> Path:
    """Barras: envases por clase en el lote + histograma de envases por imagen (con los ceros)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (dir_lote / "conteos.csv").open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    tot = [sum(int(r[c]) for r in filas) for c in nombres]
    ceros = [sum(1 for r in filas if int(r[c]) == 0) / max(1, len(filas)) for c in nombres]
    por_img = [int(r["total"]) for r in filas]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    colores = ["#c0392b" if "botella" in c else "#2471a3" for c in nombres]
    ax[0].bar(nombres, tot, color=colores)
    ax[0].set_title(f"Envases por clase ({len(filas)} imágenes; rojo=botella, azul=lata)")
    ax[0].tick_params(axis="x", rotation=60)
    for i, v in enumerate(tot):
        ax[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax[1].bar(nombres, ceros, color=colores)
    ax[1].set_title("Fracción de imágenes con CERO de la clase")
    ax[1].set_ylim(0, 1)
    ax[1].tick_params(axis="x", rotation=60)
    ax[2].hist(por_img, bins=range(0, max(por_img + [1]) + 2), color="#7f8c8d", align="left")
    ax[2].set_title(f"Envases por imagen (vacías: {sum(1 for v in por_img if v == 0)})")
    ax[2].set_xlabel("envases etiquetados en la imagen")
    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=110)
    plt.close(fig)
    return salida
