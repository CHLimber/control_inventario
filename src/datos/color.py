"""Control de color por clase sobre la caja: detecta etiquetas equivocadas del dataset de origen.

Ejemplos reales (darians_bottle, vistos por el agente Sintéticos): latas con dibujos naranjas/verdes
rotuladas coca_lata, botellas vacías o de Sprite rotuladas coca_botella, una Coca rotulada fanta_lata.

Se mide sobre la zona CENTRAL de la caja (sin bordes, donde suele asomar el fondo) la fracción de
píxeles de cada tono (HSV de OpenCV: H en 0..179) y la de píxeles oscuros (líquido de cola). Las reglas
son mínimos por clase; los umbrales se eligieron mirando la distribución por clase y fuente
(scripts/datos_color.py imprime percentiles y arma una grilla de los rechazados).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Verde hasta 95 (no 85): las latas de Sprite de catálogo son verde azulado.
TONOS = {"rojo": [(0, 10), (165, 180)], "naranja": [(8, 25)], "verde": [(35, 95)], "azul": [(95, 130)]}
MEDIDAS = ("rojo", "naranja", "verde", "azul", "oscuro", "gris")
ZONA_CENTRAL = (0.20, 0.15)  # se recorta este margen (fracción del ancho y del alto) a cada lado
LADO_ANALISIS = 64

# Reglas: lista de (medida, mínimo, máximo). Todas deben cumplirse. Calibradas con la distribución por
# clase y fuente (datos_color.py) y las grillas docs/datos/color_rechazos.jpg / color_aceptados.jpg:
# umbrales bajos, para sacar solo lo claramente incoherente (variantes de otro color, botellas vacías,
# cajas de otra marca), no las fotos con poca saturación.
REGLAS_COLOR: dict[str, list[tuple[str, float, float]]] = {
    "coca_lata": [("rojo", 0.15, 1.0)],
    "coca_botella": [("rojo", 0.05, 1.0), ("oscuro", 0.10, 1.0)],   # sin líquido oscuro = botella vacía
    "fanta_lata": [("naranja", 0.10, 1.0)],
    "fanta_botella": [("naranja", 0.10, 1.0)],
    "sprite_lata": [("verde", 0.08, 1.0)],
    "sprite_botella": [("verde", 0.08, 1.0), ("oscuro", 0.0, 0.35)],  # oscura = probablemente cola
    "pepsi_lata": [("azul", 0.08, 1.0)],
    "pepsi_botella": [("azul", 0.08, 1.0)],
}
# Imagen casi sin color (grises o muy desaturada): no se puede juzgar la marca por color -> se descarta.
MAX_GRIS = 0.97


def medir(img: Image.Image, caja_px: tuple[int, int, int, int]) -> dict[str, float]:
    """Fracciones de tono sobre la zona central de la caja (x1, y1, x2, y2 en píxeles)."""
    import cv2

    x1, y1, x2, y2 = caja_px
    mx, my = (x2 - x1) * ZONA_CENTRAL[0], (y2 - y1) * ZONA_CENTRAL[1]
    zona = img.crop((round(x1 + mx), round(y1 + my), round(x2 - mx), round(y2 - my)))
    if zona.width < 2 or zona.height < 2:
        return {m: 0.0 for m in MEDIDAS}
    zona = zona.convert("RGB").resize((LADO_ANALISIS, LADO_ANALISIS))
    hsv = cv2.cvtColor(np.asarray(zona), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    color = (s > 90) & (v > 50)
    res = {}
    for tono, rangos in TONOS.items():
        ok = np.zeros(h.shape, bool)
        for a, b in rangos:
            ok |= (h >= a) & (h < b)
        res[tono] = float((ok & color).mean())
    res["oscuro"] = float((v < 70).mean())
    res["gris"] = float((s < 40).mean())
    return res


def motivo_color(clase: str, medidas: dict[str, float]) -> str:
    """'' si el color es coherente con la clase; si no, el motivo (p. ej. 'color_rojo_0.12')."""
    if medidas.get("gris", 0.0) > MAX_GRIS:
        return f"color_gris_{medidas['gris']:.2f}"
    for medida, lo, hi in REGLAS_COLOR.get(clase, []):
        f = medidas.get(medida, 0.0)
        if not lo <= f <= hi:
            return f"color_{medida}_{f:.2f}"
    return ""


def cargar_rechazos(ruta) -> dict[tuple[str, int], str]:
    """{(ruta_imagen, índice_de_caja): motivo} de las cajas rechazadas en _control_color.csv."""
    import pandas as pd
    from pathlib import Path

    ruta = Path(ruta)
    if not ruta.exists():
        return {}
    d = pd.read_csv(ruta, keep_default_na=False, dtype={"motivo": str})
    d = d[d["motivo"] != ""]
    return {(r.ruta, int(r.caja)): r.motivo for r in d.itertuples()}
