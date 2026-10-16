"""Utilidades comunes para datasets en formato YOLO (rutas, etiquetas, ids originales)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import yaml

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

# Roboflow exporta copias aumentadas como "<foto>_jpg.rf.<hash>.jpg": el sufijo .rf.<hash>
# cambia en cada copia, pero el prefijo identifica la foto original.
_RE_RF = re.compile(r"\.rf\.[0-9a-z]+$", re.IGNORECASE)
_RE_EXT_EMBEBIDA = re.compile(r"_(jpe?g|png|bmp|webp)$", re.IGNORECASE)


def id_original(nombre_archivo: str) -> str:
    """Identificador de la foto original: quita extensión, sufijo .rf.<hash> y el '_jpg' embebido."""
    tallo = Path(nombre_archivo).stem
    tallo = _RE_RF.sub("", tallo)
    return _RE_EXT_EMBEBIDA.sub("", tallo)


def es_copia_roboflow(nombre_archivo: str) -> bool:
    return bool(_RE_RF.search(Path(nombre_archivo).stem))


def listar_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXT_IMAGEN)


def ruta_etiqueta(ruta_img: Path) -> Path:
    """images/xxx.jpg -> labels/xxx.txt (convención YOLO)."""
    partes = list(ruta_img.parts)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i] == "images":
            partes[i] = "labels"
            break
    return Path(*partes).with_suffix(".txt")


def pares_imagen_etiqueta(dir_ds: Path) -> Iterator[tuple[Path, Path]]:
    """Recorre todas las imágenes de un dataset YOLO con su archivo de etiquetas (puede no existir)."""
    for img in listar_imagenes(dir_ds):
        if "images" not in img.parts:
            continue
        yield img, ruta_etiqueta(img)


def split_origen(ruta_img: Path, dir_ds: Path) -> str:
    """Split que traía el dataset original (train/valid/test), según la carpeta."""
    rel = ruta_img.relative_to(dir_ds).parts
    return rel[0] if len(rel) > 2 else "sin_split"


def nombres_clases_yolo(dir_ds: Path) -> list[str]:
    """Lee 'names' de data.yaml (lista o dict)."""
    ruta = dir_ds / "data.yaml"
    if not ruta.exists():
        return []
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    nombres = datos.get("names", [])
    if isinstance(nombres, dict):
        return [str(nombres[k]) for k in sorted(nombres, key=int)]
    return [str(n) for n in nombres]


def leer_etiquetas_yolo(ruta_txt: Path) -> list[tuple[int, float, float, float, float]]:
    """Lee cajas YOLO normalizadas (cid, xc, yc, w, h). Los polígonos se convierten a su caja."""
    if not ruta_txt.exists():
        return []
    cajas = []
    for linea in ruta_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        partes = linea.split()
        if len(partes) < 5:
            continue
        try:
            cid = int(float(partes[0]))
            valores = [float(v) for v in partes[1:]]
        except ValueError:
            continue
        if len(valores) == 4:
            xc, yc, w, h = valores
        else:  # polígono de segmentación: x1 y1 x2 y2 ...
            xs, ys = valores[0::2], valores[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cajas.append((cid, xc, yc, w, h))
    return cajas


def escribir_etiquetas_yolo(ruta_txt: Path, cajas: list[tuple[int, float, float, float, float]]) -> None:
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)
    lineas = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in cajas]
    ruta_txt.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")


def a_pixeles(xc: float, yc: float, w: float, h: float, ancho: int, alto: int) -> tuple[int, int, int, int]:
    x1 = max(0, round((xc - w / 2) * ancho))
    y1 = max(0, round((yc - h / 2) * alto))
    x2 = min(ancho, round((xc + w / 2) * ancho))
    y2 = min(alto, round((yc + h / 2) * alto))
    return x1, y1, x2, y2


def cargar_env_principal(dir_datos: Path) -> None:
    """Carga el .env del repo principal (el padre de DIR_DATOS) sin pisar variables ya definidas.

    Así un worktree que apunta INVENTARIO_DATOS a <MAIN>/datos encuentra <MAIN>/.env.
    """
    from dotenv import load_dotenv

    for ruta in (Path(os.getenv("INVENTARIO_ENV", "")), dir_datos.parent / ".env"):
        if str(ruta) and ruta.is_file():
            load_dotenv(ruta, override=False)
