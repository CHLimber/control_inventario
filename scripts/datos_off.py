"""Descarga fotos de producto (frente del envase) de Open Food Facts para la biblioteca de los sintéticos.

Open Food Facts (https://world.openfoodfacts.org) es una base abierta de productos con foto frontal.
Las fotos están bajo CC BY-SA 3.0 (la base, ODbL): se usan solo como material de entrenamiento y NO se
redistribuyen. Cada descarga deja la atribución en ``indice.csv`` (código de barras, nombre, URL).

Solo se aceptan envases INDIVIDUALES de nuestras marcas y formatos, para que sirvan como objeto
recortado: se descartan los packs ("9 latas", "6 x 330 ml"), las variantes que no son la clásica
(diet, zero, cereza, limón, uva...) y los productos cuyo formato no queda claro.

Salida: DIR_DATOS/web/off/<clase>/off__<código>.jpg  +  DIR_DATOS/web/off/indice.csv
Luego:  python scripts/sinteticos_objetos.py --web <DATOS>/web/off

Uso:  python scripts/datos_off.py [--paginas 8] [--marcas fanta sprite pepsi coca-cola]
Cortesía con el servicio: un User-Agent propio, una consulta cada pocos segundos y pocas descargas en paralelo.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DIR_DATOS, nombres_clases  # noqa: E402

API = "https://world.openfoodfacts.org/cgi/search.pl"
CABECERA = {"User-Agent": "ControlInventarioUniversidad/1.0 (proyecto academico; fotos solo para entrenamiento)"}
MARCAS = {"fanta": "fanta", "sprite": "sprite", "pepsi": "pepsi", "coca-cola": "coca"}
CAMPOS = "code,product_name,brands,quantity,packaging,image_front_url,countries_tags"

LATA = re.compile(r"\b(lata|latas|can|cans|canette|canettes|dose|dosen|lattina|blik|tin)\b")
BOTELLA = re.compile(r"\b(botella|botellas|bottle|bottles|bouteille|bouteilles|flasche|flaschen|garrafa|pet|botellita)\b")
PACK = re.compile(r"\b\d{1,2}\s*(x|por)\s*\d|\b\d{1,2}\s*(latas|cans|botellas|bottles|bouteilles|dosen|canettes|unidades|units|u)\b|\bpack\b|\bcaja\b|\bbox\b")
VARIANTE = re.compile(r"\b(diet|zero|light|cero|sin azucar|sugar free|sans sucre|cherry|cereza|vanilla|vainilla|spiced|grape|uva|"
                      r"lemon|limon|lime|lima|peach|durazno|mix|berry|frutos|tropical|apple|manzana|strawberry|fresa|frutilla|"
                      r"pina|pineapple|mango|passion|maracuya|exotic|blood|zero sugar|free|max|stevia|coffee|cafe|vainilla)\b")


def sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def clasificar(producto: dict, marca: str) -> tuple[str, str] | None:
    """(clase, formato) si es un envase individual clásico de la marca; None si hay que descartarlo."""
    nombre = sin_acentos(producto.get("product_name") or "")
    texto = f"{nombre} {sin_acentos(producto.get('packaging') or '')} {sin_acentos(producto.get('quantity') or '')}"
    marcas = sin_acentos(producto.get("brands") or "")
    clave = marca.replace("-", " ")
    if clave not in nombre.replace("-", " ") or clave.split()[0] not in marcas.replace("-", " "):
        return None
    if PACK.search(texto) or VARIANTE.search(texto):
        return None
    lata, botella = bool(LATA.search(texto)), bool(BOTELLA.search(texto))
    if lata == botella:      # ambos o ninguno: formato dudoso
        return None
    formato = "lata" if lata else "botella"
    clase = f"{MARCAS[marca]}_{formato}"
    return (clase, formato) if clase in nombres_clases() else None


def buscar(marca: str, paginas: int, pausa: float) -> list[dict]:
    salida = []
    for pagina in range(1, paginas + 1):
        params = {"action": "process", "json": 1, "page_size": 100, "page": pagina, "fields": CAMPOS,
                  "tagtype_0": "brands", "tag_contains_0": "contains", "tag_0": marca,
                  "tagtype_1": "categories", "tag_contains_1": "contains", "tag_1": "beverages",
                  "sort_by": "unique_scans_n"}
        for intento in range(3):
            try:
                r = requests.get(API, params=params, headers=CABECERA, timeout=60)
                r.raise_for_status()
                productos = r.json().get("products", [])
                break
            except (requests.RequestException, ValueError) as e:
                print(f"  {marca} pág {pagina}: {type(e).__name__}; reintento")
                time.sleep(5 * (intento + 1))
        else:
            continue
        salida += productos
        print(f"  {marca}: página {pagina}, {len(productos)} productos")
        if len(productos) < 100:
            break
        time.sleep(pausa)
    return salida


def _bajar(args: tuple[str, str, Path]) -> str:
    url, _, destino = args
    if destino.exists():
        return "ya"
    for u in (url.replace(".400.jpg", ".full.jpg"), url):
        try:
            r = requests.get(u, headers=CABECERA, timeout=60)
            if r.status_code == 200 and len(r.content) > 5000:
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(r.content)
                return "ok"
        except requests.RequestException:
            continue
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paginas", type=int, default=8, help="páginas de 100 productos por marca")
    ap.add_argument("--marcas", nargs="*", default=list(MARCAS))
    ap.add_argument("--pausa", type=float, default=7.0, help="segundos entre consultas")
    ap.add_argument("--salida", type=Path, default=DIR_DATOS / "web" / "off")
    a = ap.parse_args()

    filas, vistos = [], set()
    for marca in a.marcas:
        print(f"[{marca}]")
        for p in buscar(marca, a.paginas, a.pausa):
            codigo, url = p.get("code"), p.get("image_front_url")
            if not codigo or not url or codigo in vistos:
                continue
            res = clasificar(p, marca)
            if res is None:
                continue
            vistos.add(codigo)
            clase, formato = res
            filas.append({"clase": clase, "formato": formato, "codigo": codigo, "nombre": p.get("product_name", ""),
                          "marca": p.get("brands", ""), "cantidad": p.get("quantity", ""),
                          "paises": ";".join(p.get("countries_tags", [])), "url": url,
                          "archivo": f"{clase}/off__{codigo}.jpg", "licencia": "CC BY-SA 3.0 Open Food Facts"})
    print(f"\n{len(filas)} productos aceptados; descargando fotos...")
    tareas = [(f["url"], f["codigo"], a.salida / f["archivo"]) for f in filas]
    with ThreadPoolExecutor(4) as ex:
        estados = list(ex.map(_bajar, tareas))
    filas = [f for f, e in zip(filas, estados) if e in ("ok", "ya")]
    a.salida.mkdir(parents=True, exist_ok=True)
    ruta = a.salida / "indice.csv"
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()) if filas else ["clase"])
        w.writeheader()
        w.writerows(filas)
    por_clase = {}
    for f in filas:
        por_clase[f["clase"]] = por_clase.get(f["clase"], 0) + 1
    print(f"{len(filas)} fotos en {a.salida}: {dict(sorted(por_clase.items()))}")


if __name__ == "__main__":
    main()
