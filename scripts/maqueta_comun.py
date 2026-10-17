"""Piezas compartidas por maqueta_plan.py y maqueta_ingesta.py.

Modelo físico de la maqueta (ver maqueta/GUIA.md):
- UNA sola fila de envases por nivel (profundidad 10-15 cm).
- 3 niveles, numerados de ABAJO hacia ARRIBA:
    nivel 1 (≈35 cm de alto) botellas grandes (1.5-3 L)
    nivel 2 (≈25 cm)         botellas chicas (500-600 ml)
    nivel 3 (≈15 cm)         latas (350 ml)
- Capacidad de un nivel = cuántos envases entran en fila en el ancho útil.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import DIR_MAQUETA, cargar_clases, nombres_clases  # noqa: E402

CLASES: list[str] = nombres_clases()
FORMATO: dict[str, str] = {c["clase"]: c["formato"] for c in cargar_clases()["clases"]}
MARCA: dict[str, str] = {c["clase"]: c["marca"] for c in cargar_clases()["clases"]}
NOMBRE: dict[str, str] = {c["clase"]: c["nombre"] for c in cargar_clases()["clases"]}
BOTELLAS = [c for c in CLASES if FORMATO[c] == "botella"]
LATAS = [c for c in CLASES if FORMATO[c] == "lata"]

# Encabezado de la verdad de terreno. Las 15 primeras columnas son las originales
# (no se tocan); distancia y celular se agregan AL FINAL.
COLUMNAS_ESCENAS = ["sesion", "escena", "fecha", "luz", "fondo", *CLASES, "observacion",
                    "distancia", "celular"]
# El plan agrega columnas de ayuda para armar la escena (no son verdad de terreno).
COLUMNAS_PLAN = [*COLUMNAS_ESCENAS, "tipo", "orden", "nivel_1", "nivel_2", "nivel_3",
                 "variaciones", "total"]

ARCHIVO_ESCENAS = DIR_MAQUETA / "escenas.csv"
ARCHIVO_PLAN = DIR_MAQUETA / "escenas_plan.csv"
ARCHIVO_FOTOS = DIR_MAQUETA / "fotos.csv"

LUCES = ["fria_techo", "natural", "calida", "poca_luz", "flash"]
FONDOS = ["blanco", "madera", "negro"]
ANGULOS = ["frontal", "izq20", "der20"]


@dataclass(frozen=True)
class Nivel:
    numero: int
    nombre: str
    tamano: str          # "grande", "chica" o "lata"
    alto_cm: int
    diametro_cm: float

    @property
    def formato(self) -> str:
        return "lata" if self.tamano == "lata" else "botella"


@dataclass
class Maqueta:
    """Parámetros físicos: ancho del estante, envases que tiene el estudiante, diámetros."""
    ancho_cm: float = 70.0
    margen_cm: float = 1.0       # a cada lado
    hueco_cm: float = 0.5        # entre envases
    diam_grande: float = 10.5    # botella 2 L
    diam_chica: float = 7.0      # botella 500-600 ml
    diam_lata: float = 6.6       # lata 350 ml
    envases: dict[str, int] = field(default_factory=lambda: {c: 5 for c in CLASES})
    grandes: dict[str, int] = field(default_factory=lambda: {c: 2 for c in BOTELLAS})

    def niveles(self) -> list[Nivel]:
        return [Nivel(1, "botellas grandes", "grande", 35, self.diam_grande),
                Nivel(2, "botellas chicas", "chica", 25, self.diam_chica),
                Nivel(3, "latas", "lata", 15, self.diam_lata)]

    def capacidad(self, nivel: Nivel) -> int:
        util = self.ancho_cm - 2 * self.margen_cm
        return max(0, math.floor((util + self.hueco_cm) / (nivel.diametro_cm + self.hueco_cm)))

    def stock(self, tamano: str) -> dict[str, int]:
        """Envases disponibles por clase para un tamaño de nivel."""
        if tamano == "lata":
            return {c: self.envases.get(c, 0) for c in LATAS}
        if tamano == "grande":
            return {c: min(self.grandes.get(c, 0), self.envases.get(c, 0)) for c in BOTELLAS}
        return {c: max(0, self.envases.get(c, 0) - self.grandes.get(c, 0)) for c in BOTELLAS}

    def problemas_conteo(self, conteo: dict[str, int]) -> list[str]:
        """Conteos imposibles para esta maqueta (lista vacía = todo bien)."""
        errores = []
        for c in CLASES:
            n = conteo.get(c, 0)
            if n < 0:
                errores.append(f"{c}={n} es negativo")
            elif n > self.envases.get(c, 0):
                errores.append(f"{c}={n} supera los {self.envases.get(c, 0)} envases que hay")
        niv = {n.tamano: n for n in self.niveles()}
        latas = sum(max(0, conteo.get(c, 0)) for c in LATAS)
        if latas > self.capacidad(niv["lata"]):
            errores.append(f"{latas} latas no entran en el nivel 3 (capacidad {self.capacidad(niv['lata'])})")
        botellas = sum(max(0, conteo.get(c, 0)) for c in BOTELLAS)
        cap_b = self.capacidad(niv["grande"]) + self.capacidad(niv["chica"])
        if botellas > cap_b:
            errores.append(f"{botellas} botellas no entran en los niveles 1 y 2 (capacidad {cap_b})")
        return errores


def parsear_por_clase(texto: str | None, defecto: int, validas: list[str]) -> dict[str, int]:
    """'5' -> todas en 5; 'coca_lata=3,fanta_lata=6' -> resto en defecto; mezcla: '4,coca_lata=6'."""
    resultado = {c: defecto for c in validas}
    partes = [p.strip() for p in (texto or "").split(",") if p.strip()]
    for parte in partes:                       # primero el valor general...
        if "=" not in parte:
            resultado = {c: int(parte) for c in validas}
    for parte in partes:                       # ...después los particulares
        if "=" in parte:
            clave, valor = (s.strip() for s in parte.split("=", 1))
            if clave not in validas:
                raise ValueError(f"clase desconocida: {clave!r} (válidas: {', '.join(validas)})")
            resultado[clave] = int(valor)
    if any(v < 0 for v in resultado.values()):
        raise ValueError("las cantidades no pueden ser negativas")
    return resultado


def maqueta_desde_args(args) -> Maqueta:
    return Maqueta(ancho_cm=args.ancho,
                   envases=parsear_por_clase(args.envases, 5, CLASES),
                   grandes=parsear_por_clase(args.grandes, 2, BOTELLAS))


def agregar_args_maqueta(parser) -> None:
    g = parser.add_argument_group("maqueta física")
    g.add_argument("--ancho", type=float, default=70.0, help="ancho interior de cada nivel en cm (def. 70)")
    g.add_argument("--envases", default=None,
                   help="envases por clase: '5' para todas o 'coca_lata=4,pepsi_lata=6' (def. 5 c/u)")
    g.add_argument("--grandes", default=None,
                   help="de las botellas de cada marca, cuántas son grandes (2 L): '2' o "
                        "'coca_botella=3' (def. 2 c/u; el resto son chicas)")


def leer_csv(ruta: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not ruta.exists():
        return [], []
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        return list(lector.fieldnames or []), list(lector)


def escribir_csv(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({k: fila.get(k, "") for k in columnas})


def conteo_de_fila(fila: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    """Cantidades por clase de una fila de CSV + errores de formato (vacío o no entero)."""
    conteo, errores = {}, []
    for c in CLASES:
        valor = (fila.get(c) or "").strip()
        try:
            conteo[c] = int(valor)
        except ValueError:
            errores.append(f"{c}={valor!r} no es un entero")
            conteo[c] = 0
    return conteo, errores


def nombre_foto(sesion: int, escena: int, foto: int) -> str:
    return f"sesion{sesion:02d}_escena{escena:02d}_foto{foto}.jpg"


def dir_sesion(base: Path, sesion: int) -> Path:
    return base / f"sesion_{sesion:02d}"
