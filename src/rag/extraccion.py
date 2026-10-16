"""Comprensión de la pregunta por reglas (sin LLM).

Tres piezas independientes y fáciles de probar:
- normalizar(texto): minúsculas, sin acentos, sin signos.
- extraer_fechas(texto, hoy): rango (desde, hasta) a partir de expresiones en español.
- extraer_productos(texto, productos): qué productos de la BD se mencionan.
- clasificar(texto, ...): una de las INTENCIONES.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Protocol

INTENCIONES = ("faltantes_rango", "detalle_cierre", "stock_producto",
               "diferencia_puntual", "desconocida")

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
DIAS_SEMANA = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3,
               "viernes": 4, "sabado": 5, "domingo": 6}

_MES_RE = "|".join(MESES)


# ------------------------------------------------------------------ texto
def normalizar(texto: str) -> str:
    """'¿Cuánta Coca-Cola hay el 12/10?' -> 'cuanta coca cola hay el 12/10'."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"(?<=\d)[-.](?=\d)", "/", t)          # 12-10 o 12.10 -> 12/10
    t = re.sub(r"[^a-z0-9/ ]+", " ", t)                # signos, guiones, '_' -> espacio
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------ fechas
@dataclass
class Rango:
    desde: date
    hasta: date
    texto: str      # expresión reconocida, para mostrar en 'parametros'

    def como_dict(self) -> dict:
        return {"desde": self.desde.isoformat(), "hasta": self.hasta.isoformat(),
                "expresion": self.texto}


def _fecha_segura(anio: int, mes: int, dia: int, hoy: date, anio_explicito: bool) -> date | None:
    try:
        f = date(anio, mes, dia)
    except ValueError:
        return None
    # "el 20 de diciembre" dicho en octubre se refiere al año pasado.
    if not anio_explicito and f > hoy:
        try:
            f = date(anio - 1, mes, dia)
        except ValueError:
            return None
    return f


def _anio(texto: str | None, hoy: date) -> tuple[int, bool]:
    if not texto:
        return hoy.year, False
    a = int(texto)
    return (a + 2000 if a < 100 else a), True


def extraer_fechas(texto: str, hoy: date) -> Rango | None:
    """Devuelve el rango de fechas mencionado, o None si no hay ninguno.

    Reconoce: hoy, ayer, anteayer, esta semana, la semana pasada, este mes, el mes
    pasado, últimos N días, el lunes..., el 12 de octubre (de 2026), 12/10(/2026),
    del 1 al 10 de octubre, entre el 1 y el 10 de octubre, del 1/10 al 10/10.
    """
    t = normalizar(texto)

    # Rangos explícitos: "del 1 al 10 de octubre", "entre el 1 y el 10 de octubre"
    m = re.search(rf"\b(?:del?|entre(?: el)?)\s+(\d{{1,2}})(?: de ({_MES_RE}))?\s+"
                  rf"(?:al?|y(?: el)?|hasta(?: el)?)\s+(\d{{1,2}}) de ({_MES_RE})(?: (?:de )?(\d{{4}}))?", t)
    if m:
        d1, mes1, d2, mes2, anio_t = m.groups()
        anio, expl = _anio(anio_t, hoy)
        f2 = _fecha_segura(anio, MESES[mes2], int(d2), hoy, expl)
        if f2:
            f1 = _fecha_segura(f2.year, MESES[mes1 or mes2], int(d1), f2, True)
            if f1 and f1 <= f2:
                return Rango(f1, f2, m.group(0))
    m = re.search(r"\b(?:del?|entre(?: el)?)\s+(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s+"
                  r"(?:al?|y(?: el)?|hasta(?: el)?)\s+(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", t)
    if m:
        d1, m1, a1, d2, m2, a2 = m.groups()
        anio2, expl2 = _anio(a2, hoy)
        f2 = _fecha_segura(anio2, int(m2), int(d2), hoy, expl2)
        if f2:
            anio1, expl1 = _anio(a1, hoy) if a1 else (f2.year, True)
            f1 = _fecha_segura(anio1, int(m1), int(d1), f2, expl1)
            if f1 and f1 <= f2:
                return Rango(f1, f2, m.group(0))

    # Fecha puntual: "12 de octubre (de 2026)" o "12/10(/2026)"
    m = re.search(rf"\b(\d{{1,2}}) de ({_MES_RE})(?: (?:de |del )?(\d{{4}}))?", t)
    if m:
        anio, expl = _anio(m.group(3), hoy)
        f = _fecha_segura(anio, MESES[m.group(2)], int(m.group(1)), hoy, expl)
        if f:
            return Rango(f, f, m.group(0))
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        anio, expl = _anio(m.group(3), hoy)
        f = _fecha_segura(anio, int(m.group(2)), int(m.group(1)), hoy, expl)
        if f:
            return Rango(f, f, m.group(0))

    # Relativas
    m = re.search(r"\bultimos? (\d{1,3}) dias\b", t)
    if m:
        n = max(1, int(m.group(1)))
        return Rango(hoy - timedelta(days=n - 1), hoy, m.group(0))
    if re.search(r"\bsemana pasada\b", t):
        lunes = hoy - timedelta(days=hoy.weekday() + 7)
        return Rango(lunes, lunes + timedelta(days=6), "la semana pasada")
    if re.search(r"\bultima semana\b", t):
        return Rango(hoy - timedelta(days=6), hoy, "la ultima semana")
    if re.search(r"\b(esta|la) semana\b", t):
        return Rango(hoy - timedelta(days=hoy.weekday()), hoy, "esta semana")
    if re.search(r"\bmes pasado\b", t):
        fin = hoy.replace(day=1) - timedelta(days=1)
        return Rango(fin.replace(day=1), fin, "el mes pasado")
    if re.search(r"\b(este|el) mes\b", t):
        return Rango(hoy.replace(day=1), hoy, "este mes")
    m = re.search(rf"\b(?:en )?({_MES_RE})(?: (?:de |del )?(\d{{4}}))?\b", t)
    if m:   # "en octubre": el mes entero (hasta hoy si es el mes en curso)
        anio, expl = _anio(m.group(2), hoy)
        mes = MESES[m.group(1)]
        if not expl and (anio, mes) > (hoy.year, hoy.month):
            anio -= 1
        ini = date(anio, mes, 1)
        fin = (date(anio + (mes == 12), mes % 12 + 1, 1) - timedelta(days=1))
        return Rango(ini, min(fin, hoy) if ini <= hoy else fin, m.group(0))
    if re.search(r"\b(anteayer|antes de ayer|antier)\b", t):
        f = hoy - timedelta(days=2)
        return Rango(f, f, "anteayer")
    if re.search(r"\bayer\b", t):
        f = hoy - timedelta(days=1)
        return Rango(f, f, "ayer")
    if re.search(r"\b(hoy|hoy dia|esta noche|esta tarde)\b", t):
        return Rango(hoy, hoy, "hoy")
    m = re.search(r"\b(?:el )?(" + "|".join(DIAS_SEMANA) + r")(?: pasado)?\b", t)
    if m:   # el último <día> anterior o igual a hoy
        atras = (hoy.weekday() - DIAS_SEMANA[m.group(1)]) % 7
        f = hoy - timedelta(days=atras)
        return Rango(f, f, m.group(0))
    return None


# ------------------------------------------------------------------ productos
class _ProductoLike(Protocol):
    id: int
    nombre: str
    marca: str
    formato: str
    clase_modelo: str


def clave_marca(marca: str) -> str:
    """'Coca-Cola' -> 'cocacola', '7Up' -> '7up'."""
    return normalizar(marca).replace(" ", "")


# alias (normalizado) -> clave de marca
ALIAS_MARCAS: dict[str, str] = {
    "coca": "cocacola", "coca cola": "cocacola", "cocacola": "cocacola", "coke": "cocacola",
    "cocas": "cocacola",
    "fanta": "fanta", "fantas": "fanta",
    "sprite": "sprite", "esprite": "sprite", "sprites": "sprite",
    "pepsi": "pepsi", "pepsis": "pepsi",
    "7up": "7up", "7 up": "7up", "seven up": "7up", "sevenup": "7up", "siete up": "7up",
}
ALIAS_FORMATOS: dict[str, str] = {
    "lata": "lata", "latas": "lata", "latita": "lata", "latitas": "lata",
    "botella": "botella", "botellas": "botella", "botellita": "botella",
    "botellitas": "botella", "envase": "botella", "envases": "botella",
    "pet": "botella", "plastico": "botella",
}


@dataclass
class Productos:
    ids: list[int] = field(default_factory=list)
    nombres: list[str] = field(default_factory=list)
    menciones: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.ids)


def extraer_productos(texto: str, productos: Iterable[_ProductoLike]) -> Productos:
    """Productos de la BD mencionados en el texto.

    'coca' -> las dos Coca-Cola; 'coca lata' / 'latas de coca' -> solo la lata;
    'latas' sin marca -> todas las latas; 'coca_lata' (clase del modelo) también vale.
    """
    productos = list(productos)
    tokens = normalizar(texto).split()
    alias = dict(ALIAS_MARCAS)
    for p in productos:   # marcas nuevas cargadas en la BD también se reconocen
        alias.setdefault(normalizar(p.marca), clave_marca(p.marca))
    alias_tok = sorted(((a.split(), k) for a, k in alias.items()), key=lambda x: -len(x[0]))

    marcas: list[tuple[int, int, str, str]] = []      # (inicio, fin, clave, texto)
    i = 0
    while i < len(tokens):
        for partes, clave in alias_tok:
            if tokens[i:i + len(partes)] == partes:
                marcas.append((i, i + len(partes), clave, " ".join(partes)))
                i += len(partes)
                break
        else:
            i += 1
    formatos = [(i, ALIAS_FORMATOS[t], t) for i, t in enumerate(tokens) if t in ALIAS_FORMATOS]

    # Cada formato se asigna a la marca más cercana (≤ 3 palabras; empate: la anterior).
    pedidos: list[tuple[str | None, set[str], str]] = [(m[2], set(), m[3]) for m in marcas]
    formatos_sueltos: list[tuple[str, str]] = []
    for pos, fmt, txt in formatos:
        mejor, dist_mejor = None, 4
        for j, (ini, fin, _, _) in enumerate(marcas):
            dist = pos - fin + 1 if pos >= fin else ini - pos
            if dist < dist_mejor:
                mejor, dist_mejor = j, dist
        if mejor is None:
            formatos_sueltos.append((fmt, txt))
        else:
            clave, fmts, mencion = pedidos[mejor]
            fmts.add(fmt)
            pedidos[mejor] = (clave, fmts, f"{mencion} {txt}")
    for fmt, txt in formatos_sueltos:
        pedidos.append((None, {fmt}, txt))

    res = Productos()
    for clave, fmts, mencion in pedidos:
        res.menciones.append(mencion)
        for p in productos:
            if clave is not None and clave_marca(p.marca) != clave:
                continue
            if fmts and p.formato not in fmts:
                continue
            if p.id not in res.ids:
                res.ids.append(p.id)
                res.nombres.append(p.nombre)
    return res


# ------------------------------------------------------------------ intención
_RE_DIFERENCIA = re.compile(r"\b(falt\w*|sobr\w*|diferencia\w*|perd\w*|perdi\w*|merma\w*|"
                            r"descuadr\w*|desfase\w*|rob\w*|desapareci\w*|extravi\w*)")
_RE_STOCK = re.compile(r"\b(stock|cuant[oa]s?|hay|queda\w*|inventario|existencia\w*|"
                       r"disponible\w*|tenemos|tengo|saldo)\b")
_RE_CIERRE = re.compile(r"\b(cierres?|escaneos?|conteos?|arqueos?|conciliacion\w*)\b")
_RE_ID_CIERRE = re.compile(
    r"\b(?:cierre|escaneo|conteo|arqueo)\s+(?:(?:n|nro|num|numero|no)\s*)?(\d{1,6})\b(?!\s*/)"
    rf"(?!\s+de\s+(?:{_MES_RE}))")
_RE_ULTIMO = re.compile(r"\b(ultimo|ultima|reciente|anterior)\b")


def extraer_id_cierre(texto: str) -> int | None:
    m = _RE_ID_CIERRE.search(normalizar(texto))
    return int(m.group(1)) if m else None


def clasificar(texto: str, hay_productos: bool, id_cierre: int | None = None) -> str:
    """Intención por palabras clave, en este orden de prioridad:

    1. 'cierre N' -> detalle_cierre (o diferencia_puntual si además pregunta por un producto
       y por faltantes: '¿cuánta coca faltó en el cierre 3?').
    2. faltó/sobró/diferencia -> diferencia_puntual con producto, faltantes_rango sin él.
    3. stock/cuánto/hay/queda -> stock_producto.
    4. 'cierre' sin número -> detalle_cierre (el último o el de la fecha pedida).
    """
    t = normalizar(texto)
    dif = bool(_RE_DIFERENCIA.search(t))
    if id_cierre is not None:
        return "diferencia_puntual" if (dif and hay_productos) else "detalle_cierre"
    if dif:
        if hay_productos:
            return "diferencia_puntual"
        if _RE_CIERRE.search(t) and _RE_ULTIMO.search(t):   # "¿qué faltó en el último cierre?"
            return "detalle_cierre"
        return "faltantes_rango"
    if _RE_STOCK.search(t):
        return "stock_producto"
    if _RE_CIERRE.search(t):
        return "detalle_cierre"
    if hay_productos:       # "¿y la fanta lata?" -> lo más útil es el stock
        return "stock_producto"
    return "desconocida"
