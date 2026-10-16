"""Fondos y materiales para los estantes sintéticos.

- Texturas procedurales que imitan la maqueta (ver maqueta/GUIA.md): cartulina blanca, negra/gris,
  papel contact de madera y cartón. Se generan con ruido suave (barato y sin archivos).
- Fotos reales SIN productos: fotos de la maqueta vacía en ``DIR_MAQUETA/fondos/`` (si existen) y las
  imágenes ``Background_*`` del split train del dataset fusionado (darians: fotos sin envases).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# color base (RGB) y variación por tipo de fondo/material
COLORES = {
    "blanco": ((236, 234, 228), 14),
    "negro": ((32, 32, 34), 10),
    "gris": ((128, 128, 130), 30),
    "madera": ((160, 112, 70), 30),
    "carton": ((176, 140, 98), 18),
    "mdf": ((190, 150, 108), 18),
}


def ruido_suave(rng: np.random.Generator, h: int, w: int, escala: int) -> np.ndarray:
    """Ruido en [-1, 1] interpolado desde una grilla gruesa (aspecto de 'manchas' suaves)."""
    gh, gw = max(2, h // escala + 2), max(2, w // escala + 2)
    g = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(-1, 1)


def _color(rng: np.random.Generator, tipo: str) -> np.ndarray:
    """Color base: variación de brillo común + un tinte chico por canal (los neutros siguen neutros)."""
    base, var = COLORES[tipo]
    tinte = 4 if tipo in ("blanco", "negro", "gris") else var / 3
    return np.clip(np.array(base, np.float32) + rng.uniform(-var, var) + rng.uniform(-tinte, tinte, 3), 0, 255)


def textura(tipo: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Textura RGB uint8 de tamaño (h, w). Las grandes se generan a media resolución (más rápido)."""
    h, w = max(1, int(h)), max(1, int(w))
    if h * w > 250_000:
        chica = textura(tipo, (h + 1) // 2, (w + 1) // 2, rng)
        return cv2.resize(chica, (w, h), interpolation=cv2.INTER_LINEAR)
    color = _color(rng, tipo)
    if tipo == "madera":
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        frec = rng.uniform(0.04, 0.12)
        dist = ruido_suave(rng, h, w, max(8, w // 6)) * rng.uniform(8, 25)
        vetas = np.sin((y + dist) * frec + ruido_suave(rng, h, w, 4) * 0.6)
        fibra = ruido_suave(rng, h, w, 3) * 0.04
        f = 1 + 0.10 * vetas + fibra + 0.06 * ruido_suave(rng, h, w, max(8, w // 3))
        img = color[None, None, :] * f[..., None]
    else:
        grano = {"blanco": 0.012, "negro": 0.05, "gris": 0.02, "carton": 0.05, "mdf": 0.03}[tipo]
        f = 1 + grano * rng.standard_normal((h, w)).astype(np.float32) + 0.04 * ruido_suave(rng, h, w, max(8, min(h, w) // 2))
        img = color[None, None, :] * f[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def listar_fondos_reales(dir_fusion_train: Path | None, dir_maqueta_fondos: Path | None) -> list[Path]:
    """Fotos reales sin productos que se pueden usar como fondo (SOLO train / maqueta vacía)."""
    rutas: list[Path] = []
    if dir_maqueta_fondos and dir_maqueta_fondos.is_dir():
        rutas += sorted(p for p in dir_maqueta_fondos.rglob("*") if p.suffix.lower() in EXT_IMG)
    if dir_fusion_train and dir_fusion_train.is_dir():
        rutas += sorted(p for p in dir_fusion_train.glob("*") if p.suffix.lower() in EXT_IMG and "background" in p.name.lower())
    return rutas


def recorte_foto(foto: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Recorte aleatorio de una foto reescalada para cubrir (h, w)."""
    fh, fw = foto.shape[:2]
    esc = max(h / fh, w / fw) * rng.uniform(1.0, 1.6)
    nh, nw = int(np.ceil(fh * esc)), int(np.ceil(fw * esc))
    g = cv2.resize(foto, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = int(rng.integers(0, nh - h + 1))
    x0 = int(rng.integers(0, nw - w + 1))
    return g[y0:y0 + h, x0:x0 + w].copy()
