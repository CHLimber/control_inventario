"""Biblioteca de objetos recortados SIN fondo (PNG RGBA) para pegar en los estantes.

Fuente: SOLO el split ``train`` del dataset fusionado multiclase (formato YOLO). Si usáramos val/test,
los sintéticos filtrarían imágenes de evaluación al entrenamiento.

Pasos por caja etiquetada:
  1. Filtro geométrico (barato): caja completa (no cortada por el borde de la foto), envase PARADO
     (relación alto/ancho típica de botella o lata), tamaño mínimo y poco tapada por otras cajas.
  2. Quitar el fondo con ``rembg`` sobre el recorte con un margen de contexto.
  3. Limpieza de la máscara: se recorta a la caja, se queda la componente conexa más grande y
     se rellenan huecos (botellas transparentes).
  4. Control de calidad de la máscara: cubre una fracción razonable de la caja y ocupa casi todo
     su alto (si rembg se "comió" medio envase, se descarta).
  5. Se guarda el PNG RGBA recortado a la máscara en ``<cache>/<clase>/`` y una fila en ``indice.csv``.
"""
from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

# Relación alto/ancho aceptada por formato (envase parado y visto de frente).
ASPECTO = {"botella": (2.0, 4.6), "lata": (1.35, 2.5)}
MIN_ALTO_PX = {"botella": 90, "lata": 60}
MARGEN_BORDE_PX = 3          # caja a menos de esto del borde de la foto = probablemente cortada
MAX_TAPADO = 0.12            # fracción máxima del área de la caja cubierta por otras cajas
CONTEXTO = 0.15              # margen de contexto alrededor de la caja para rembg
COBERTURA = (0.40, 0.97)     # fracción de la caja cubierta por la máscara
ALTO_MASCARA_MIN = 0.85      # la máscara debe ocupar >= 85 % del alto de la caja
ANCHO_MASCARA_MIN = 0.70

# Control de color por clase: (tono, fracción mínima, fracción máxima) de los píxeles del envase.
# Detecta etiquetas equivocadas del dataset de origen, vistas a ojo en la biblioteca: latas con dibujos
# naranjas/verdes rotuladas "coca_lata", botellas VACÍAS o de Sprite rotuladas "coca_botella", una Coca
# rotulada "sprite_botella", etc. "oscuro" = brillo bajo (líquido de cola). Umbrales elegidos mirando
# la distribución por clase (ver docs/sinteticos.md).
REGLAS_COLOR = {
    "coca_lata": [("rojo", 0.35, 1.0)],
    "coca_botella": [("rojo", 0.10, 1.0), ("oscuro", 0.15, 1.0)],
    "fanta_lata": [("naranja", 0.35, 1.0)],
    "fanta_botella": [("naranja", 0.25, 1.0)],
    "sprite_lata": [("verde", 0.25, 1.0)],
    "sprite_botella": [("verde", 0.15, 1.0), ("oscuro", 0.0, 0.30)],
    "pepsi_lata": [("azul", 0.15, 1.0)],
    "pepsi_botella": [("azul", 0.15, 1.0)],
    "7up_lata": [("verde", 0.30, 1.0)],
}
TONOS = {"rojo": [(0, 10), (165, 180)], "naranja": [(8, 25)], "verde": [(35, 85)], "azul": [(95, 130)]}

COLUMNAS_INDICE = ["archivo", "clase", "formato", "origen", "ancho", "alto", "cobertura"]


@dataclass
class Candidato:
    imagen: Path
    clase: str
    formato: str
    caja: tuple[int, int, int, int]  # x1, y1, x2, y2 en píxeles
    grupo: str                       # foto original (las copias aumentadas de Roboflow comparten grupo)
    motivo: str = ""                 # vacío = aceptado por el filtro geométrico
    extra: dict = field(default_factory=dict)


def grupo_de(nombre: str) -> str:
    """'alias__Foto_123_jpg.rf.abc.jpg' -> 'alias__Foto_123' (las copias aumentadas comparten grupo)."""
    base = Path(nombre).stem
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", base, maxsplit=1)[0]


def _leer_yolo(ruta: Path) -> list[tuple[int, float, float, float, float]]:
    filas = []
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            p = linea.split()
            if len(p) >= 5:
                filas.append((int(p[0]), *map(float, p[1:5])))
    return filas


def _tapado(i: int, cajas: list[tuple[int, int, int, int]]) -> float:
    x1, y1, x2, y2 = cajas[i]
    area = max(1, (x2 - x1) * (y2 - y1))
    m = np.zeros((y2 - y1, x2 - x1), bool)
    for j, (a1, b1, a2, b2) in enumerate(cajas):
        if j == i:
            continue
        ix1, iy1, ix2, iy2 = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)
        if ix2 > ix1 and iy2 > iy1:
            m[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = True
    return float(m.sum()) / area


def candidatos_yolo(dir_imagenes: Path, dir_etiquetas: Path, clases: list[dict]) -> list[Candidato]:
    """Todas las cajas del árbol YOLO con el motivo de rechazo geométrico (si lo hay)."""
    salida: list[Candidato] = []
    for ruta_img in sorted(dir_imagenes.glob("*")):
        if ruta_img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        filas = _leer_yolo(dir_etiquetas / (ruta_img.stem + ".txt"))
        if not filas:
            continue
        with Image.open(ruta_img) as im:
            ancho, alto = im.size
        cajas = []
        for _, cx, cy, w, h in filas:
            cajas.append((int(round((cx - w / 2) * ancho)), int(round((cy - h / 2) * alto)),
                          int(round((cx + w / 2) * ancho)), int(round((cy + h / 2) * alto))))
        for i, (cid, *_r) in enumerate(filas):
            if not 0 <= cid < len(clases):
                continue
            info = clases[cid]
            x1, y1, x2, y2 = cajas[i]
            c = Candidato(ruta_img, info["clase"], info["formato"], (x1, y1, x2, y2), grupo_de(ruta_img.name))
            w, h = x2 - x1, y2 - y1
            lo, hi = ASPECTO[info["formato"]]
            if w <= 0 or h <= 0:
                c.motivo = "caja_vacia"
            elif x1 < MARGEN_BORDE_PX or y1 < MARGEN_BORDE_PX or x2 > ancho - MARGEN_BORDE_PX or y2 > alto - MARGEN_BORDE_PX:
                c.motivo = "cortada_por_borde"
            elif h < MIN_ALTO_PX[info["formato"]]:
                c.motivo = "chica"
            elif not lo <= h / w <= hi:
                c.motivo = "no_parada"   # acostada, inclinada o caja deformada
            elif _tapado(i, cajas) > MAX_TAPADO:
                c.motivo = "tapada"
            salida.append(c)
    return salida


def seleccionar(cands: Iterable[Candidato], max_por_clase: int, semilla: int = 0) -> list[Candidato]:
    """Aceptados, como mucho UNO por (foto original, clase) y ``max_por_clase`` por clase."""
    rng = random.Random(semilla)
    por_clase: dict[str, dict[str, list[Candidato]]] = {}
    for c in cands:
        if not c.motivo:
            por_clase.setdefault(c.clase, {}).setdefault(c.grupo, []).append(c)
    elegidos = []
    for clase in sorted(por_clase):
        grupos = por_clase[clase]
        uno_por_grupo = [rng.choice(grupos[g]) for g in sorted(grupos)]
        rng.shuffle(uno_por_grupo)
        elegidos.extend(uno_por_grupo[:max_por_clase])
    return elegidos


# --------------------------------------------------------------------------- máscaras

def segmentador_rembg(modelo: str = "u2netp") -> Callable[[Image.Image], np.ndarray]:
    """Devuelve f(imagen RGB) -> máscara uint8 (0-255) usando rembg con el modelo indicado."""
    from rembg import new_session, remove

    sesion = new_session(modelo)

    def segmentar(img: Image.Image) -> np.ndarray:
        m = remove(img, session=sesion, only_mask=True)
        return np.asarray(m.convert("L"))

    return segmentar


def _rellenar_filas(binaria: np.ndarray) -> np.ndarray:
    """Hace la máscara 'convexa por filas' y rellena filas vacías entre la tapa y el cuerpo.

    Un envase parado visto de frente es convexo fila a fila. rembg suele separar la tapa del cuerpo
    (el cuello de una botella PET es transparente): se interpola el ancho entre ambas partes.
    """
    salida = np.zeros_like(binaria)
    filas = np.where(binaria.any(axis=1))[0]
    if filas.size == 0:
        return salida
    izq = np.full(binaria.shape[0], np.nan)
    der = np.full(binaria.shape[0], np.nan)
    for y in filas:
        xs = np.where(binaria[y])[0]
        izq[y], der[y] = xs[0], xs[-1]
    rango = np.arange(filas[0], filas[-1] + 1)
    izq_i = np.interp(rango, filas, izq[filas])
    der_i = np.interp(rango, filas, der[filas])
    for y, a, b in zip(rango, izq_i, der_i):
        salida[y, int(round(a)):int(round(b)) + 1] = True
    return salida


def limpiar_mascara(mascara: np.ndarray, caja_local: tuple[int, int, int, int]) -> np.ndarray:
    """Recorta la máscara a la caja (+2 px), descarta manchas chicas y la vuelve convexa por filas.

    Se conservan las componentes con >= 3 % del área de la mayor que no bajan más que ella (así la
    tapa no se pierde, pero no se pega una mano o un objeto de abajo) y se
    rellena el hueco del cuello. El borde conserva la transición suave de rembg; el interior queda sólido.
    """
    import cv2
    from scipy import ndimage

    x1, y1, x2, y2 = caja_local
    m = np.zeros_like(mascara)
    a1, b1 = max(0, x1 - 2), max(0, y1 - 2)
    a2, b2 = min(m.shape[1], x2 + 2), min(m.shape[0], y2 + 2)
    m[b1:b2, a1:a2] = mascara[b1:b2, a1:a2]
    binaria = m > 80
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mascara)
    areas = stats[1:, cv2.CC_STAT_AREA]
    mayor = 1 + int(np.argmax(areas))
    fondo_mayor = stats[mayor, cv2.CC_STAT_TOP] + stats[mayor, cv2.CC_STAT_HEIGHT]
    # piezas chicas que NO bajan del cuerpo (la tapa está arriba; nada de un envase cuelga por debajo)
    validas = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.03 * areas.max()
               and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] <= fondo_mayor + 2]
    binaria = _rellenar_filas(np.isin(etiquetas, validas))
    interior = ndimage.binary_erosion(binaria, iterations=2)
    alfa = np.where(binaria, np.maximum(m, 160), 0).astype(np.uint8)
    alfa[interior] = 255
    return alfa


def calidad_mascara(alfa: np.ndarray, caja_local: tuple[int, int, int, int]) -> str:
    """'' si la máscara es aceptable; si no, el motivo del rechazo."""
    x1, y1, x2, y2 = caja_local
    sub = alfa[y1:y2, x1:x2] > 127
    if sub.size == 0 or not sub.any():
        return "mascara_vacia"
    cob = sub.mean()
    if not COBERTURA[0] <= cob <= COBERTURA[1]:
        return f"cobertura_{cob:.2f}"
    filas = np.where(sub.any(axis=1))[0]
    cols = np.where(sub.any(axis=0))[0]
    if (filas[-1] - filas[0] + 1) < ALTO_MASCARA_MIN * sub.shape[0]:
        return "mascara_baja"
    if (cols[-1] - cols[0] + 1) < ANCHO_MASCARA_MIN * sub.shape[1]:
        return "mascara_angosta"
    return ""


def inclinacion(alfa: np.ndarray) -> float:
    """Ángulo (grados) del eje mayor de la máscara respecto de la vertical (momentos de 2.º orden)."""
    import cv2

    mo = cv2.moments((alfa > 127).astype(np.uint8), binaryImage=True)
    if mo["m00"] == 0:
        return 0.0
    ang = 0.5 * np.degrees(np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"]))  # respecto de la horizontal
    return float(ang - 90 if ang > 0 else ang + 90)


def enderezar(rgba: np.ndarray, max_grados: float = 20.0) -> np.ndarray:
    """Rota el recorte para dejar el envase vertical (si está inclinado <= max_grados)."""
    import cv2

    ang = inclinacion(rgba[..., 3])
    if abs(ang) < 1.0 or abs(ang) > max_grados:
        return rgba
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def fraccion_tono(rgba: np.ndarray, tono: str) -> float:
    """Fracción de píxeles opacos y saturados del envase cuyo tono cae en ``tono``."""
    import cv2

    hsv = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2HSV)
    opaco = rgba[..., 3] > 200
    if not opaco.any():
        return 0.0
    h, sat, val = hsv[..., 0][opaco], hsv[..., 1][opaco], hsv[..., 2][opaco]
    if tono == "oscuro":
        return float((val < 70).mean())
    ok = np.zeros(h.shape, bool)
    for a, b in TONOS[tono]:
        ok |= (h >= a) & (h < b)
    ok &= (sat > 90) & (val > 50)
    return float(ok.mean())


def motivo_color(clase: str, rgba: np.ndarray) -> str:
    """'' si el color del envase es coherente con su clase; si no, el motivo (p. ej. 'color_rojo_0.21')."""
    for tono, lo, hi in REGLAS_COLOR.get(clase, []):
        f = fraccion_tono(rgba, tono)
        if not lo <= f <= hi:
            return f"color_{tono}_{f:.2f}"
    return ""


def recortar_a_alfa(rgba: np.ndarray, umbral: int = 20) -> np.ndarray:
    ys, xs = np.where(rgba[..., 3] > umbral)
    if ys.size == 0:
        return rgba[:0, :0]
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def extraer_objeto(cand: Candidato, segmentar: Callable[[Image.Image], np.ndarray]) -> tuple[Image.Image | None, str, float]:
    """Recorta con contexto, quita el fondo, endereza y devuelve (PNG RGBA, motivo de rechazo, cobertura)."""
    with Image.open(cand.imagen) as im:
        im = im.convert("RGB")
        W, H = im.size
        x1, y1, x2, y2 = cand.caja
        mx, my = int((x2 - x1) * CONTEXTO), int((y2 - y1) * CONTEXTO)
        a1, b1, a2, b2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
        rec = im.crop((a1, b1, a2, b2))
    caja_local = (x1 - a1, y1 - b1, x2 - a1, y2 - b1)
    alfa = limpiar_mascara(segmentar(rec), caja_local)
    motivo = calidad_mascara(alfa, caja_local)
    if motivo:
        return None, motivo, 0.0
    lx1, ly1, lx2, ly2 = caja_local
    cob = float((alfa[ly1:ly2, lx1:lx2] > 127).mean())
    rgba = recortar_a_alfa(enderezar(recortar_a_alfa(np.dstack([np.asarray(rec), alfa]))))
    motivo = motivo_color(cand.clase, rgba)
    if motivo:
        return None, motivo, 0.0
    h, w = rgba.shape[:2]
    lo, hi = ASPECTO[cand.formato]
    if w == 0 or not lo <= h / w <= hi * 1.1:
        return None, "aspecto_final", 0.0
    return Image.fromarray(rgba, "RGBA"), "", cob


def leer_indice(dir_cache: Path) -> list[dict]:
    ruta = dir_cache / "indice.csv"
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir_indice(dir_cache: Path, filas: list[dict]) -> None:
    dir_cache.mkdir(parents=True, exist_ok=True)
    with (dir_cache / "indice.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_INDICE)
        w.writeheader()
        for fila in filas:
            w.writerow({k: fila.get(k, "") for k in COLUMNAS_INDICE})


def nombre_cache(cand: Candidato) -> str:
    x1, y1, x2, y2 = cand.caja
    return f"{cand.imagen.stem[:80]}_{x1}_{y1}_{x2}_{y2}.png"


def cargar_biblioteca(dir_cache: Path, dir_train: Path | None = None) -> dict[str, list[Path]]:
    """{clase: [rutas PNG]} según indice.csv (si no hay índice, se recorren las carpetas).

    Con ``dir_train`` se descartan los objetos cuya foto de origen ya NO está en train (por si el
    dataset fusionado se regeneró y una foto pasó a val/test: evita filtrar evaluación al entrenamiento).
    """
    filas = leer_indice(dir_cache)
    bib: dict[str, list[Path]] = {}
    en_train = {p.name for p in dir_train.iterdir()} if dir_train and dir_train.is_dir() else None
    if filas:
        for f in filas:
            if en_train is not None and f["origen"].split("|")[0] not in en_train:
                continue
            p = dir_cache / f["archivo"]
            if p.exists():
                bib.setdefault(f["clase"], []).append(p)
    else:
        for carpeta in sorted(p for p in dir_cache.iterdir() if p.is_dir() and not p.name.startswith("_")):
            pngs = sorted(carpeta.glob("*.png"))
            if pngs:
                bib[carpeta.name] = pngs
    return bib
