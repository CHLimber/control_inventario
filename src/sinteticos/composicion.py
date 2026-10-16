"""Compone UNA imagen sintética de estante (3 niveles, una sola fila) con sus cajas exactas.

Geometría (en cm, como la maqueta de maqueta/GUIA.md; se convierte a píxeles con ``s`` px/cm):

    techo (tabla)
    nivel 3  latas                 ~15 cm libres
    tabla + tira portaprecios
    nivel 2  botellas chicas       ~25 cm   (a veces latas: ver ``p_latas_medio``)
    tabla + tira portaprecios
    nivel 1  botellas grandes      ~35 cm
    tabla base + tira portaprecios

Orden de dibujo: pared -> fondo del estante -> sombreado interior -> piso de cada nivel -> sombras
-> envases (en orden z) -> tablas y tiras portaprecios -> laterales -> perspectiva -> luz/ruido/blur.

Las cajas salen de un "mapa de ids" que se pinta igual que la imagen: cada píxel guarda qué envase se
ve ahí (o -1). Así la caja es la del objeto VISIBLE (sin la parte tapada por la tira o por el vecino),
y la perspectiva se aplica al mapa con vecino más cercano: las cajas quedan exactas por construcción.

Regla de oclusión: un envase con menos de ``min_visible`` (40 %) de su silueta a la vista NO se
etiqueta ni se cuenta. La colocación limita el solapamiento con el vecino (<= 12 % del ancho) para
que esto casi nunca pase; si pasa, el envase se quita de la escena (no queda pegado sin etiqueta).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.sinteticos import fondos as fnd

# Medidas reales aproximadas (cm) por "tipo de envase en el nivel": (alto min, alto max, diámetro min, diámetro max)
MEDIDAS = {
    "botella_grande": (29.0, 33.0, 9.5, 11.0),
    "botella_chica": (20.0, 24.0, 6.3, 7.5),
    "lata": (11.8, 12.6, 6.4, 6.8),
}
TOLERANCIA_ASPECTO = 0.20   # se respeta la forma del recorte si difiere <= 20 % de la forma real
LUCES = ("fria", "natural", "calida", "poca_luz", "flash")


@dataclass
class Parametros:
    lado_largo: int = 1024
    lado_corto: int = 768
    p_vertical: float = 0.7          # foto vertical (estante entero) vs horizontal
    p_acercamiento: float = 0.2      # foto de un solo nivel (como 'distancia=nivel_k' del protocolo)
    p_estante_vacio: float = 0.04    # imagen sin ningún envase (todo en cero)
    p_nivel_vacio: float = 0.12      # cada nivel por separado puede quedar vacío
    p_latas_medio: float = 0.35      # el nivel del medio lleva latas en vez de botellas chicas
    llenado_botellas: tuple[float, float] = (2.0, 1.6)   # Beta(a, b) de la fracción de capacidad
    llenado_latas: tuple[float, float] = (2.5, 1.0)
    p_agrupado: float = 0.6          # productos iguales juntos (facings) vs mezclados
    p_solape: float = 0.25           # prob. de solapamiento leve con el vecino
    max_solape: float = 0.12         # fracción del ancho
    max_rotacion: float = 5.0        # grados
    p_tira: float = 0.9              # tira portaprecios
    p_sombra: float = 0.7
    p_fondo_foto: float = 0.15       # fondo del estante = foto real (si hay)
    p_pared_foto: float = 0.4        # pared alrededor del estante = foto real desenfocada (si hay)
    p_perspectiva_lateral: float = 0.5  # foto tomada ~20° de costado (izq/der)
    max_perspectiva: float = 0.07
    p_desenfoque: float = 0.4
    max_ruido: float = 6.0
    min_visible: float = 0.40
    pesos_clase: dict[str, float] = field(default_factory=dict)  # opcional, para rebalancear


@dataclass
class Objeto:
    clase: str
    cid: int
    rgba: np.ndarray        # ya escalado/rotado/ajustado
    x: int                  # esquina sup. izq. en el lienzo
    y: int
    nivel: int


@dataclass
class Escena:
    imagen: np.ndarray                      # RGB uint8
    cajas: list[tuple[int, float, float, float, float]]   # (cid, x1, y1, x2, y2) en píxeles
    meta: dict


# --------------------------------------------------------------------------- utilidades

@lru_cache(maxsize=800)
def cargar_rgba(ruta: str) -> np.ndarray:
    data = np.fromfile(ruta, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        raise ValueError(f"no es un PNG RGBA: {ruta}")
    return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)


@lru_cache(maxsize=64)
def cargar_rgb(ruta: str) -> np.ndarray:
    data = np.fromfile(ruta, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"no se pudo leer: {ruta}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _u(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.uniform(a, b))


def ajustar_color(rgba: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brillo/contraste/saturación/tono leves por objeto (no cambia la marca: tono ±3 en escala 0-180)."""
    rgb = rgba[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + _u(rng, -3, 3)) % 180
    hsv[..., 1] *= _u(rng, 0.85, 1.15)
    hsv[..., 2] *= _u(rng, 0.85, 1.15)
    rgb2 = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    c = _u(rng, 0.9, 1.1)
    media = rgb2.mean(axis=(0, 1), keepdims=True)
    rgb2 = (rgb2 - media) * c + media
    return np.dstack([np.clip(rgb2, 0, 255).astype(np.uint8), rgba[..., 3]])


def rotar(rgba: np.ndarray, grados: float) -> np.ndarray:
    if abs(grados) < 0.2:
        return rgba
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    ys, xs = np.where(out[..., 3] > 20)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if ys.size else out


def suavizar_borde(alfa: np.ndarray, sigma: float) -> np.ndarray:
    """Erosiona 1 px y desenfoca el alfa: borde 'feather' para que no se note el recorte."""
    a = cv2.erode(alfa, np.ones((3, 3), np.uint8), iterations=1)
    k = max(3, int(sigma * 3) | 1)
    return cv2.GaussianBlur(a, (k, k), sigma)


def preparar_objeto(rgba: np.ndarray, tipo: str, s: float, rng: np.random.Generator, p: Parametros) -> np.ndarray:
    """Escala al tamaño real del envase en este nivel, rota levemente, ajusta color y suaviza el borde."""
    hmin, hmax, dmin, dmax = MEDIDAS[tipo]
    alto_cm, diam_cm = _u(rng, hmin, hmax), _u(rng, dmin, dmax)
    asp_real = alto_cm / diam_cm
    asp_rec = rgba.shape[0] / max(1, rgba.shape[1])
    asp = float(np.clip(asp_rec, asp_real * (1 - TOLERANCIA_ASPECTO), asp_real * (1 + TOLERANCIA_ASPECTO)))
    alto = max(8, int(round(alto_cm * s)))
    ancho = max(4, int(round(alto / asp)))
    interp = cv2.INTER_AREA if alto < rgba.shape[0] else cv2.INTER_LINEAR
    o = cv2.resize(rgba, (ancho, alto), interpolation=interp)
    o = ajustar_color(o, rng)
    o = rotar(o, _u(rng, -p.max_rotacion, p.max_rotacion))
    o[..., 3] = suavizar_borde(o[..., 3], _u(rng, 0.6, 1.2))
    return o


def _clases_nivel(rng: np.random.Generator, disponibles: list[str], n: int, p: Parametros) -> list[str]:
    if not disponibles or n <= 0:
        return []
    pesos = np.array([p.pesos_clase.get(c, 1.0) for c in disponibles], float)
    pesos /= pesos.sum()
    if rng.random() < p.p_agrupado:
        salida: list[str] = []
        while len(salida) < n:
            c = disponibles[rng.choice(len(disponibles), p=pesos)]
            salida += [c] * int(rng.integers(1, 5))
        return salida[:n]
    return [disponibles[i] for i in rng.choice(len(disponibles), size=n, p=pesos)]


def _homografia(rng: np.random.Generator, ventana: tuple[int, int, int, int], p: Parametros) -> tuple[np.ndarray, str]:
    """Perspectiva leve que lleva la ventana (x, y, w, h) del lienzo a la foto de salida (w, h).

    Frontal: las 4 esquinas se mueven un poco (±1.2 %). De costado (~20°, como 'izq20/der20' del
    protocolo): el lado lejano se achica. Lo que entra por los bordes sale del margen extra del lienzo
    (pared y estante reales de la escena), nunca de un reflejo de la imagen.
    """
    x, y, w, h = ventana
    src = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    j = 0.012
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) + np.float32(rng.uniform(-j, j, (4, 2)) * [w, h])
    vista = "frontal"
    if rng.random() < p.p_perspectiva_lateral:
        k = _u(rng, 0.02, p.max_perspectiva)
        vista = "izq" if rng.random() < 0.5 else "der"
        lejos = [1, 2] if vista == "izq" else [0, 3]   # cámara a la izquierda: el lado derecho queda lejos
        for i in lejos:
            dst[i, 1] += k * h if i in (0, 1) else -k * h
            dst[i, 0] += (-1 if i in (1, 2) else 1) * k * 0.5 * w
    return cv2.getPerspectiveTransform(src, dst), vista


def _luz(img: np.ndarray, rng: np.random.Generator, p: Parametros) -> tuple[np.ndarray, str]:
    """Iluminación global: tipo de luz (como la columna 'luz' del protocolo), gradiente, brillo/contraste."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    luz = LUCES[rng.choice(len(LUCES), p=[0.3, 0.2, 0.2, 0.15, 0.15])]
    ganancia = {"fria": (0.96, 1.0, 1.07), "natural": (1.0, 1.0, 1.0), "calida": (1.12, 1.0, 0.80),
                "poca_luz": (1.0, 1.0, 1.0), "flash": (1.0, 1.0, 1.02)}[luz]
    f *= np.float32(ganancia) * np.float32(rng.uniform(0.96, 1.04, 3))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # gradiente lineal de iluminación (ventana, foco de un lado)
    ang = _u(rng, 0, 2 * np.pi)
    t = ((x / w - 0.5) * np.cos(ang) + (y / h - 0.5) * np.sin(ang))
    fuerza = _u(rng, 0.0, 0.35 if luz == "natural" else 0.18)
    campo = 1 + fuerza * t
    if luz == "flash":
        cx, cy = _u(rng, 0.35, 0.65) * w, _u(rng, 0.35, 0.65) * h
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (max(w, h) ** 2)
        campo *= 0.75 + 0.55 * np.exp(-r2 / _u(rng, 0.05, 0.15))
    if luz == "poca_luz":
        campo *= _u(rng, 0.4, 0.65)
    f *= campo[..., None]
    brillo, contraste = _u(rng, 0.9, 1.1), _u(rng, 0.85, 1.15)
    media = f.mean()
    f = (f - media) * contraste + media * brillo
    return np.clip(f, 0, 255), luz


# --------------------------------------------------------------------------- composición

def componer(rng: np.random.Generator, biblioteca: dict[str, list], clases: list[dict],
             fondos_reales: list, p: Parametros | None = None) -> Escena:
    p = p or Parametros()
    id_de = {c["clase"]: c["id"] for c in clases}
    formato_de = {c["clase"]: c["formato"] for c in clases}
    disp = {f: [c["clase"] for c in clases if c["formato"] == f and biblioteca.get(c["clase"])]
            for f in ("botella", "lata")}

    acercamiento = rng.random() < p.p_acercamiento
    vertical = (not acercamiento) and rng.random() < p.p_vertical
    W, Hfoto = (p.lado_corto, p.lado_largo) if vertical else (p.lado_largo, p.lado_corto)

    # --- geometría en cm
    ancho_cm = _u(rng, 60, 80)
    alto_niv = {1: 35 + _u(rng, -2, 2), 2: 25 + _u(rng, -1.5, 1.5), 3: 15 + _u(rng, -0.5, 1.5)}
    tabla, lateral, tira = _u(rng, 1.0, 1.8), _u(rng, 1.0, 2.0), _u(rng, 1.8, 2.6)
    alto_total = sum(alto_niv.values()) + 4 * tabla
    ancho_total = ancho_cm + 2 * lateral
    margen = _u(rng, 0.02, 0.08)
    frac_ancho = 1.0
    if acercamiento:
        # foto de cerca de un nivel: se ve entre el 60 % y el 100 % del ancho del estante
        frac_ancho = _u(rng, 0.6, 1.0)
        visible_cm = ancho_total * frac_ancho + (2 * _u(rng, 0, 4) if frac_ancho > 0.95 else 0)
        s = W / visible_cm
        H = int(s * alto_total + 2 * margen * W) + 2
    else:
        H = Hfoto
        s = min(W * (1 - 2 * margen) / ancho_total, H * (1 - 2 * margen * 0.5) / alto_total)
    ex0 = (W - s * ancho_total) / 2 + _u(rng, -0.5, 0.5) * max(0, W - s * ancho_total) * 0.5
    ey0 = (H - s * alto_total) / 2 + _u(rng, -0.5, 0.5) * max(0, H - s * alto_total) * 0.5
    # lienzo con margen extra alrededor de la foto: la perspectiva puede "traer" esa zona al cuadro
    Wout, Hout = W, H
    ancho_px = s * ancho_total
    P = int(max(0.12 * max(W, H), (ancho_px - W) / 2 + 0.12 * W))
    W, H = W + 2 * P, H + 2 * P
    ex0, ey0 = ex0 + P, ey0 + P
    vx = P
    if ancho_px > Wout:   # acercamiento: la ventana recorre el estante
        vx = int(round(_u(rng, ex0, ex0 + ancho_px - Wout)))
    ventana = (vx, P, Wout, Hout)

    def X(cm: float) -> int:
        return int(round(ex0 + cm * s))

    def Y(cm: float) -> int:
        return int(round(ey0 + cm * s))

    # niveles de arriba (3) hacia abajo (1): (techo interior, piso) en cm desde arriba del estante
    niveles = {}
    yc = tabla
    for k in (3, 2, 1):
        niveles[k] = (yc, yc + alto_niv[k])
        yc += alto_niv[k] + tabla

    # --- pared y fondo del estante
    tipos_mat = ["mdf", "madera", "blanco", "carton", "negro"]
    material = tipos_mat[rng.choice(len(tipos_mat), p=[0.3, 0.25, 0.2, 0.15, 0.1])]
    if fondos_reales and rng.random() < p.p_pared_foto:
        pared = fnd.recorte_foto(_foto(fondos_reales[rng.integers(len(fondos_reales))]), H, W, rng)
        pared = cv2.GaussianBlur(pared, (0, 0), _u(rng, 2, 6))
        tipo_pared = "foto"
    else:
        tipo_pared = ["blanco", "gris", "carton", "madera"][rng.integers(4)]
        pared = fnd.textura(tipo_pared, H, W, rng)
    lienzo = pared.astype(np.float32)

    ix0, ix1 = X(lateral), X(lateral + ancho_cm)
    iy0, iy1 = Y(0), Y(alto_total)
    hi, wi = iy1 - iy0, ix1 - ix0
    if fondos_reales and rng.random() < p.p_fondo_foto:
        tipo_fondo = "foto"
        fondo = fnd.recorte_foto(_foto(fondos_reales[rng.integers(len(fondos_reales))]), hi, wi, rng)
    else:
        tipo_fondo = ["blanco", "madera", "negro", "gris"][rng.choice(4, p=[0.35, 0.3, 0.2, 0.15])]
        fondo = fnd.textura(tipo_fondo, hi, wi, rng)
    lienzo[iy0:iy1, ix0:ix1] = fondo

    # sombreado interior: más oscuro bajo cada tabla y hacia los costados
    sombra_int = _u(rng, 0.1, 0.35)
    for k, (t, b) in niveles.items():
        y0, y1 = Y(t), Y(b)
        yy = np.arange(y1 - y0, dtype=np.float32)[:, None]
        xx = np.arange(wi, dtype=np.float32)[None, :]
        g = 1 - sombra_int * np.exp(-yy / max(1.0, 0.25 * (y1 - y0)))
        g = g * (1 - 0.5 * sombra_int * (np.exp(-xx / (0.06 * wi)) + np.exp(-(wi - xx) / (0.06 * wi))))
        lienzo[y0:y1, ix0:ix1] *= g[..., None]

    # piso visible de cada nivel (franja delante del fondo)
    pisos = {}
    for k, (t, b) in niveles.items():
        banda = _u(rng, 0.4, 2.5) if k < 3 else _u(rng, 0.0, 0.8)
        y0, y1 = Y(b - banda), Y(b)
        if y1 > y0:
            piso = fnd.textura(material if material != "negro" else "gris", y1 - y0, wi, rng).astype(np.float32)
            lienzo[y0:y1, ix0:ix1] = piso * _u(rng, 0.8, 1.05)
        pisos[k] = banda

    # --- envases por nivel
    escena_vacia = rng.random() < p.p_estante_vacio
    latas_medio = rng.random() < p.p_latas_medio
    plan = {3: ("lata", "lata"), 2: ("lata", "lata") if latas_medio else ("botella", "botella_chica"),
            1: ("botella", "botella_grande")}
    objetos: list[Objeto] = []
    capacidad = {}
    for k in (3, 2, 1):
        formato, tipo = plan[k]
        diam_med = (MEDIDAS[tipo][2] + MEDIDAS[tipo][3]) / 2
        cap = int((ancho_cm - 2) // (diam_med + 0.5))
        capacidad[k] = cap
        if escena_vacia or not disp[formato] or rng.random() < p.p_nivel_vacio:
            continue
        a, b = p.llenado_latas if formato == "lata" else p.llenado_botellas
        n = int(np.clip(round(rng.beta(a, b) * cap), 1, cap))
        nombres = _clases_nivel(rng, disp[formato], n, p)
        prep = []
        for c in nombres:
            ruta = biblioteca[c][rng.integers(len(biblioteca[c]))]
            prep.append((c, preparar_objeto(cargar_rgba(str(ruta)), tipo, s, rng, p)))
        # que entren en el ancho interior
        libre_px = (ix1 - ix0) - 2 - sum(o.shape[1] for _, o in prep)
        while prep and libre_px < 0:
            _, o = prep.pop()
            libre_px += o.shape[1]
        if not prep:
            continue
        conc = _u(rng, 0.3, 6.0)   # alta = parejo; baja = huecos grandes (nivel medio vacío)
        huecos = rng.dirichlet(np.full(len(prep) + 1, conc)) * libre_px
        x = ix0 + 1 + huecos[0]
        piso_y = Y(niveles[k][1])
        for i, (c, o) in enumerate(prep):
            if i > 0 and rng.random() < p.p_solape:
                x -= _u(rng, 0.03, p.max_solape) * min(o.shape[1], prep[i - 1][1].shape[1])
            base = piso_y - int(round(_u(rng, 0.0, max(0.1, pisos[k] * 0.8)) * s))
            xi = int(round(min(max(x, ix0), ix1 - o.shape[1])))
            objetos.append(Objeto(c, id_de[c], o, xi, base - o.shape[0], k))
            x += o.shape[1] + huecos[i + 1]

    # orden z: aleatorio dentro de cada nivel (quién tapa a quién en un solape)
    orden = list(rng.permutation(len(objetos)))
    mapa = np.full((H, W), -1, np.int32)
    total_px = np.zeros(len(objetos), np.int64)
    for i in orden:
        o = objetos[i]
        sub = o.rgba[..., 3] > 127
        total_px[i] = sub.sum()
        _pegar_mascara(mapa, sub, o.x, o.y, i)

    # tablas, tiras y laterales tapan (mapa = -1)
    elementos_frente = []   # (y0, y1, x0, x1, tipo)
    for k, (t, b) in niveles.items():
        elementos_frente.append((Y(t - tabla), Y(t), X(0), X(ancho_total), "tabla"))
    elementos_frente.append((Y(alto_total - tabla), Y(alto_total), X(0), X(ancho_total), "tabla"))
    con_tira = rng.random() < p.p_tira
    if con_tira:
        for k, (t, b) in niveles.items():
            elementos_frente.append((Y(b + tabla - tira), Y(b + tabla), X(lateral), X(lateral + ancho_cm), "tira"))
    elementos_frente.append((Y(0), Y(alto_total), X(0), X(lateral), "lateral"))
    elementos_frente.append((Y(0), Y(alto_total), X(lateral + ancho_cm), X(ancho_total), "lateral"))
    for y0, y1, x0, x1, _ in elementos_frente:
        mapa[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = -1

    # regla de oclusión: si queda < min_visible a la vista, el envase se quita de la escena
    visibles = np.bincount(mapa[mapa >= 0].ravel(), minlength=len(objetos)) if objetos else np.zeros(0)
    quitar = {i for i in range(len(objetos)) if total_px[i] == 0 or visibles[i] / total_px[i] < p.min_visible}
    if quitar:   # se repinta sin ellos (un quitado podía tapar a otro)
        mapa_tmp = np.full((H, W), -1, np.int32)
        for i in orden:
            if i not in quitar:
                _pegar_mascara(mapa_tmp, objetos[i].rgba[..., 3] > 127, objetos[i].x, objetos[i].y, i)
        for y0, y1, x0, x1, _ in elementos_frente:
            mapa_tmp[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = -1
        mapa = mapa_tmp

    # --- sombras suaves de los envases sobre el fondo
    con_sombra = rng.random() < p.p_sombra and objetos
    if con_sombra:
        sombra = np.zeros((H, W), np.float32)
        dx, dy = int(_u(rng, -1.0, 1.0) * s), int(_u(rng, -0.6, 0.2) * s)
        for i in orden:
            if i in quitar:
                continue
            o = objetos[i]
            _pegar_max(sombra, o.rgba[..., 3].astype(np.float32) / 255, o.x + dx, o.y + dy)
        sig = _u(rng, 0.4, 1.5) * s
        sombra = cv2.GaussianBlur(sombra, (0, 0), sig)
        lienzo *= (1 - _u(rng, 0.15, 0.45) * sombra)[..., None]

    # --- envases (alfa suavizado)
    for i in orden:
        if i in quitar:
            continue
        _pegar_rgba(lienzo, objetos[i].rgba, objetos[i].x, objetos[i].y)

    # --- tablas, tiras portaprecios y laterales
    tono_mat = fnd.textura(material, H, W, rng).astype(np.float32)
    for y0, y1, x0, x1, tipo in elementos_frente:
        y0c, y1c, x0c, x1c = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
        if y1c <= y0c or x1c <= x0c:
            continue
        if tipo == "tira":
            lienzo[y0c:y1c, x0c:x1c] = _tira(rng, y1c - y0c, x1c - x0c)
        else:
            lienzo[y0c:y1c, x0c:x1c] = tono_mat[y0c:y1c, x0c:x1c] * (0.92 if tipo == "lateral" else 1.0)
            if tipo == "tabla" and y1c - y0c > 2:   # canto superior más claro
                lienzo[y0c:y0c + 1, x0c:x1c] = np.clip(lienzo[y0c:y0c + 1, x0c:x1c] * 1.15, 0, 255)

    # --- acercamiento: recorte a un solo nivel (sin envases de otros niveles)
    nivel_foto = None
    if acercamiento:
        nivel_foto = int(rng.integers(1, 4))
        t, b = niveles[nivel_foto]
        y0 = max(0, Y(t - _u(rng, 0, 0.6) * tabla))
        y1 = min(H, Y(b + tabla + _u(rng, 0, 0.3)))
        Hout = y1 - y0
        ventana = (vx, y0, Wout, Hout)

    # --- perspectiva leve (imagen con interpolación lineal, mapa de ids con vecino más cercano)
    Hm, vista = _homografia(rng, ventana, p)
    img = cv2.warpPerspective(lienzo, Hm, (Wout, Hout), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mapa = cv2.warpPerspective(mapa.astype(np.float32), Hm, (Wout, Hout), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=-1).astype(np.int32)

    # --- luz, desenfoque y ruido globales
    img, luz = _luz(img, rng, p)
    if rng.random() < p.p_desenfoque:
        img = cv2.GaussianBlur(img, (0, 0), _u(rng, 0.3, 1.2))
    ruido = _u(rng, 0, p.max_ruido) * (2 if luz == "poca_luz" else 1)
    if ruido > 0.5:
        img = img + rng.normal(0, ruido, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # --- cajas = bbox del objeto visible en el mapa final
    # (un envase que quedó en su mayoría fuera del cuadro, p. ej. de otro nivel en un acercamiento,
    #  no se etiqueta: misma regla del 40 % visible)
    cajas = []
    parciales = 0
    if objetos:
        ys, xs = np.nonzero(mapa >= 0)
        ids = mapa[ys, xs]
        for i in range(len(objetos)):
            if i in quitar:
                continue
            sel = ids == i
            if not sel.any():
                continue
            if sel.sum() < p.min_visible * total_px[i]:
                parciales += 1
                continue
            xi, yi = xs[sel], ys[sel]
            cajas.append((objetos[i].cid, float(xi.min()), float(yi.min()), float(xi.max() + 1), float(yi.max() + 1)))
    meta = {"modo": f"nivel_{nivel_foto}" if acercamiento else "completo", "vista": vista, "luz": luz,
            "fondo": tipo_fondo, "pared": tipo_pared, "material": material, "latas_medio": latas_medio,
            "tira": con_tira, "sombra": bool(con_sombra), "px_por_cm": round(s, 2), "frac_ancho": round(frac_ancho, 2),
            "quitados_por_oclusion": len(quitar), "parciales_sin_etiqueta": parciales, "vacio": escena_vacia}
    return Escena(img, cajas, meta)


# --------------------------------------------------------------------------- pegado

def _foto(x) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else cargar_rgb(str(x))


def _recorte_destino(H: int, W: int, h: int, w: int, x: int, y: int):
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (y0, y1, x0, x1), (y0 - y, y1 - y, x0 - x, x1 - x)


def _pegar_mascara(mapa: np.ndarray, m: np.ndarray, x: int, y: int, valor: int) -> None:
    r = _recorte_destino(*mapa.shape, *m.shape, x, y)
    if r:
        (a, b, c, d), (e, f, g, h) = r
        mapa[a:b, c:d][m[e:f, g:h]] = valor


def _pegar_max(dst: np.ndarray, m: np.ndarray, x: int, y: int) -> None:
    r = _recorte_destino(*dst.shape, *m.shape, x, y)
    if r:
        (a, b, c, d), (e, f, g, h) = r
        np.maximum(dst[a:b, c:d], m[e:f, g:h], out=dst[a:b, c:d])


def _pegar_rgba(dst: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    r = _recorte_destino(dst.shape[0], dst.shape[1], rgba.shape[0], rgba.shape[1], x, y)
    if r:
        (a, b, c, d), (e, f, g, h) = r
        al = rgba[e:f, g:h, 3:4].astype(np.float32) / 255
        dst[a:b, c:d] = dst[a:b, c:d] * (1 - al) + rgba[e:f, g:h, :3].astype(np.float32) * al


def _tira(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Tira portaprecios de cartulina con precios 'Bs N' escritos."""
    colores = [(240, 238, 230), (245, 232, 150), (205, 230, 205), (250, 250, 250)]
    base = np.array(colores[rng.integers(len(colores))], np.float32)
    t = np.clip(base[None, None, :] * (1 + 0.02 * rng.standard_normal((h, w, 1))), 0, 255).astype(np.uint8)
    t = np.ascontiguousarray(t)
    escala = max(0.25, h / 40)
    grosor = max(1, int(h / 18))
    paso = int(_u(rng, 3.5, 6.0) * h)
    x = int(_u(rng, 0.2, 1.0) * h)
    color_txt = (20, 20, 20) if rng.random() < 0.7 else (170, 20, 20)
    while x < w - h:
        cv2.putText(t, f"Bs {int(rng.integers(4, 18))}", (x, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, color_txt, grosor, cv2.LINE_AA)
        x += paso
    return t.astype(np.float32)


def dibujar_cajas(img: np.ndarray, cajas, nombres: list[str] | None = None) -> np.ndarray:
    """Copia de la imagen con las cajas dibujadas (para revisar a ojo)."""
    out = np.ascontiguousarray(img.copy())
    paleta = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
              (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]
    for cid, x1, y1, x2, y2 in cajas:
        col = paleta[int(cid) % len(paleta)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), col, 2)
        if nombres:
            cv2.putText(out, nombres[int(cid)], (int(x1) + 2, int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def guardar_jpg(ruta: Path, img_rgb: np.ndarray, calidad: int) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
    if not ok:
        raise IOError(f"no se pudo codificar {ruta}")
    ruta.write_bytes(buf.tobytes())
