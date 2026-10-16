"""Genera docs/datos.md (reporte de calidad y conteos) y las grillas de muestras en docs/datos/.

Por dataset: licencia, versión, imágenes, fotos originales, copias aumentadas, duplicados por
pHash, distribución de clases con su destino en el remapeo y una grilla de muestras con las
cajas dibujadas (verde = una de nuestras clases, amarillo = solo_detector, rojo = descartar).
Al final: conteos por clase y split de detector_1clase, multiclase y recortes, con avisos.

Uso:  python scripts/datos_reporte.py [--muestras 12] [--sin-grillas]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

from src.config import nombres_clases  # noqa: E402
from src.datos.fuentes import FUENTES, NOTAS_REPORTE  # noqa: E402
from src.datos.fusion import conteos  # noqa: E402
from src.datos.indice import cargar_indice  # noqa: E402
from src.datos.remapeo import DESCARTAR, SOLO_DETECTOR, cargar_remapeo, destino  # noqa: E402
from src.datos.rutas import (  # noqa: E402
    DIR_DOCS, DIR_DOCS_DATOS, DIR_FUSION, DIR_RAW, DIR_RECORTES, RUTA_INDICE, RUTA_REMAPEO, nombres_por_alias,
)
from src.datos.split import verificar_fugas  # noqa: E402
from src.datos.util import SPLITS, a_pixeles, leer_etiquetas_yolo, ruta_etiqueta  # noqa: E402

UMBRAL_POCOS = 300
COLOR = {"clase": (40, 200, 70), SOLO_DETECTOR: (240, 200, 0), DESCARTAR: (230, 40, 40)}


def tabla_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for fila in df.itertuples(index=False):
        lineas.append("| " + " | ".join(str(v) for v in fila) + " |")
    return "\n".join(lineas)


def grilla(df_alias: pd.DataFrame, alias: str, nombres: list[str], remapeo: dict, n: int, destino_png: Path) -> None:
    """Grilla de n imágenes (una por foto original, elegidas de forma determinista) con cajas."""
    d = df_alias[df_alias["n_cajas"] > 0].drop_duplicates("grupo")
    d = d.assign(_o=d["ruta"].map(lambda r: hashlib.sha1(r.encode()).hexdigest())).sort_values("_o").head(n)
    lado, cols = 320, 4
    filas = max(1, -(-len(d) // cols))
    lienzo = Image.new("RGB", (cols * lado, filas * lado), (255, 255, 255))
    for k, r in enumerate(d.itertuples()):
        with Image.open(DIR_RAW / r.ruta) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        dib = ImageDraw.Draw(img)
        grosor = max(2, round(max(img.size) / 250))
        for cid, xc, yc, w, h in leer_etiquetas_yolo(ruta_etiqueta(DIR_RAW / r.ruta)):
            nombre = nombres[cid] if 0 <= cid < len(nombres) else str(cid)
            dest = destino(remapeo, alias, nombre)
            color = COLOR.get(dest, COLOR["clase"])
            x1, y1, x2, y2 = a_pixeles(xc, yc, w, h, img.width, img.height)
            dib.rectangle([x1, y1, x2, y2], outline=color, width=grosor)
            etiqueta = dest if dest not in (SOLO_DETECTOR, DESCARTAR) else nombre
            dib.text((x1 + 2, max(0, y1 - 11)), etiqueta[:22], fill=color)
        img.thumbnail((lado - 6, lado - 6))
        lienzo.paste(img, ((k % cols) * lado + 3, (k // cols) * lado + 3))
    destino_png.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destino_png, "JPEG", quality=85)


def seccion_dataset(f: dict, info: dict, df: pd.DataFrame, nombres: list[str], remapeo: dict,
                    df_todo: pd.DataFrame, con_grilla: bool, n_muestras: int) -> list[str]:
    alias = f["alias"]
    md = [f"### `{alias}` — {f['workspace']}/{f['project']}", ""]
    if df.empty:
        md += [f"**No disponible**: {info.get('error', 'no descargado')}.", ""]
        if info.get("licencia"):
            md += [f"Licencia declarada: {info['licencia']}."]
        if f.get("nota"):
            md += ["", f"Nota: {f['nota']}"]
        return md + [""]
    pre = (next((v for v in info.get("versiones", []) if v["version"] == info.get("version_usada")), {}) or {})
    aumentos = ", ".join(sorted(pre.get("aumentos", {}).keys())) or "ninguno"
    preproc = ", ".join(f"{k}" for k in pre.get("preprocesado", {}).keys()) or "ninguno"

    n = len(df)
    fotos = df["foto"].nunique() if "foto" in df.columns else df["id_original"].nunique()
    grupos = df["grupo"].nunique()
    sufijo_rf = int(df["copia_aumentada"].sum())
    copias = n - fotos
    # Duplicados por pHash: imágenes cuyo pHash coincide con el de OTRA foto original.
    ph = df_todo.groupby("phash")["id_original"].nunique()
    dup_internos = int(df["phash"].map(ph).gt(1).sum())
    otros = df_todo[df_todo["alias"] != alias]
    dup_externos = int(df["phash"].isin(set(otros["phash"])).sum())
    sin_cajas = int((df["n_cajas"] == 0).sum())
    cajas = Counter()
    for s in df["clases_ids"].astype(str):
        cajas.update(int(x) for x in s.split(";") if x != "")
    md += [
        f"- **Licencia**: {info.get('licencia', '¿?')} · **versión** v{info.get('version_usada')} "
        f"(de {len(info.get('versiones', []))}) · uso: `{f['uso']}` · tipo: `{f['tipo']}`",
        f"- Preprocesado de Roboflow: {preproc}; aumentos offline: {aumentos}",
        f"- Imágenes: **{n}** · fotos originales: **{fotos}** · copias aumentadas: {copias} "
        f"({sufijo_rf} con sufijo `.rf.<hash>`) · grupos anti-fuga (fotos + duplicados/ráfagas): {grupos}",
        f"- Duplicados por pHash con otra foto: {dup_internos} · con otros datasets: {dup_externos} · "
        f"imágenes sin cajas: {sin_cajas}",
        f"- Tamaño típico: {int(df['ancho'].median())}×{int(df['alto'].median())} px · "
        f"cajas: {sum(cajas.values())} ({sum(cajas.values()) / max(1, n):.1f} por imagen)",
        "",
    ]
    filas = []
    for cid, nombre in enumerate(nombres):
        filas.append({"id": cid, "clase original": nombre, "cajas": cajas.get(cid, 0),
                      "destino": destino(remapeo, alias, nombre)})
    for cid in sorted(set(cajas) - set(range(len(nombres)))):
        filas.append({"id": cid, "clase original": "(id sin nombre)", "cajas": cajas[cid], "destino": DESCARTAR})
    tabla = pd.DataFrame(filas).sort_values("cajas", ascending=False)
    md += [tabla_md(tabla), ""]
    basura = tabla[tabla["destino"].isin([DESCARTAR, SOLO_DETECTOR])]
    if not basura.empty:
        md += [f"Clases que no aportan marca: {int(basura['cajas'].sum())} cajas "
               f"({100 * basura['cajas'].sum() / max(1, tabla['cajas'].sum()):.0f} %).", ""]
    if f.get("nota"):
        md += [f"**Veredicto**: {f['nota']}", ""]
    if con_grilla and not df.empty:
        ruta = DIR_DOCS_DATOS / f"muestras_{alias}.jpg"
        grilla(df, alias, nombres, remapeo, n_muestras, ruta)
        md += [f"![muestras {alias}](datos/{ruta.name})", ""]
    return md


def seccion_fusion(modo: str, nombres_salida: list[str]) -> tuple[list[str], list[str]]:
    ruta = DIR_FUSION / modo / "_manifiesto.csv"
    if not ruta.exists():
        return [f"_{modo}: todavía no generado._", ""], []
    man = pd.read_csv(ruta, dtype={"clases_salida": str, "phash": str}, keep_default_na=False)
    t = conteos(man, nombres_salida)
    piv = t.pivot_table(index="clase", columns="split", values=["imagenes", "cajas"], aggfunc="sum")
    filas, avisos = [], []
    for c in nombres_salida:
        fila = {"clase": c}
        for s in SPLITS:
            fila[f"img {s}"] = int(piv.loc[c, ("imagenes", s)])
            fila[f"cajas {s}"] = int(piv.loc[c, ("cajas", s)])
        filas.append(fila)
        if fila["cajas train"] < UMBRAL_POCOS:
            avisos.append(f"{modo}: `{c}` tiene {fila['cajas train']} cajas en train (< {UMBRAL_POCOS})")
    tot = man.groupby("split").agg(imagenes=("ruta", "size"), cajas=("n_cajas_salida", "sum"),
                                   negativos=("n_cajas_salida", lambda s: int((s == 0).sum())))
    tot = tot.reindex(list(SPLITS)).reset_index()
    fugas = verificar_fugas(man)
    md = [tabla_md(tot), "", tabla_md(pd.DataFrame(filas)), "",
          "Por dataset de origen (imágenes):", "",
          tabla_md(man.groupby(["alias", "split"]).size().unstack(fill_value=0)
                   .reindex(columns=list(SPLITS), fill_value=0).reset_index()), "",
          "Verificación de fugas: " + ("**sin fugas** (ningún grupo ni pHash en dos splits)." if not fugas
                                      else "**FUGAS**: " + "; ".join(fugas)), ""]
    return md, avisos


def seccion_recortes(clases: list[str]) -> tuple[list[str], list[str]]:
    ruta = DIR_RECORTES / "indice_recortes.csv"
    if not ruta.exists():
        return ["_Recortes: todavía no generados._", ""], []
    ind = pd.read_csv(ruta, dtype={"phash": str, "motivo_descarte": str}, keep_default_na=False)
    ok = ind[ind["motivo_descarte"] == ""]
    piv = ok.groupby(["clase", "split"]).size().unstack(fill_value=0).reindex(index=clases, columns=list(SPLITS),
                                                                              fill_value=0)
    piv["total"] = piv.sum(axis=1)
    avisos = [f"recortes: `{c}` tiene {int(piv.loc[c, 'train'])} recortes en train (< {UMBRAL_POCOS})"
              for c in clases if piv.loc[c, "train"] < UMBRAL_POCOS]
    desc = ind[ind["motivo_descarte"] != ""]
    desc = desc.assign(motivo=desc["motivo_descarte"].str.replace(r"_\d+\.\d+$", "", regex=True))
    desc = desc.groupby(["clase", "motivo"]).size().unstack(fill_value=0)
    desc = desc.reindex(index=clases, fill_value=0)
    por_origen = ok.groupby(["clase", "origen"]).size().unstack(fill_value=0).reindex(index=clases, fill_value=0)
    fugas = verificar_fugas(ok.rename(columns={"imagen_original": "grupo"}))
    nit = pd.to_numeric(ind["nitidez"], errors="coerce").dropna()
    md = ["Recortes guardados (clase × split):", "", tabla_md(piv.reset_index()), "",
          "Por dataset de origen:", "", tabla_md(por_origen.reset_index()), "",
          "Descartes por motivo:", "", tabla_md(desc.reset_index()), "",
          f"Nitidez (varianza del Laplaciano a {128} px de alto): percentiles 5/25/50 = "
          f"{nit.quantile(.05):.0f} / {nit.quantile(.25):.0f} / {nit.quantile(.5):.0f}.", "",
          "Verificación de fugas: " + ("**sin fugas**." if not fugas else "**FUGAS**: " + "; ".join(fugas)), ""]
    if (DIR_DOCS_DATOS / "recortes_muestras.jpg").exists():
        md += ["![recortes](datos/recortes_muestras.jpg)", ""]
    return md, avisos


def seccion_color(clases: list[str]) -> list[str]:
    ruta = DIR_FUSION / "_control_color.csv"
    if not ruta.exists():
        return ["_Sin control de color (correr `scripts/datos_color.py`)._", ""]
    col = pd.read_csv(ruta, keep_default_na=False)
    col["rechazada"] = col["motivo"] != ""
    tasa = (col.pivot_table(index="clase", columns="alias", values="rechazada", aggfunc="mean")
            .reindex(clases).mul(100).round(0))
    tasa = tasa.apply(lambda s: s.map(lambda v: "-" if pd.isna(v) else f"{int(v)} %")).reset_index()
    md = ["`scripts/datos_color.py` mide, en la zona central de cada caja con marca, la fracción de píxeles "
          "rojos/naranjas/verdes/azules, oscuros (líquido de cola) y grises, y aplica un mínimo por clase "
          "(`src/datos/color.py`): coca_lata rojo ≥ 0,15; coca_botella rojo ≥ 0,05 y oscuro ≥ 0,10 (botella "
          "vacía); fanta naranja ≥ 0,10; sprite/7up verde ≥ 0,08 (sprite_botella oscuro ≤ 0,35); pepsi azul ≥ "
          "0,08; y descarta las cajas en escala de grises (gris > 0,97). La caja rechazada no entra a los "
          "recortes y su imagen sale del multiclase (queda en el detector). Nada se borra de `raw/`; el "
          "detalle está en `fusion/_control_color.csv`, `fusion/multiclase/_descartes_color.csv` y en la "
          "columna `motivo_descarte` de `recortes/indice_recortes.csv`.", "",
          f"Cajas rechazadas: {int(col['rechazada'].sum())} de {len(col)}. Porcentaje por clase y fuente:", "",
          tabla_md(tasa), "",
          "Qué saca (ver grillas): variantes de otro color rotuladas con la marca (Fanta frutilla/uva/"
          "manzana, Pepsi Diet, 7up Free), latas naranjas rotuladas coca_lata y botellas vacías de "
          "darians_bottle, las copias en grises de again_aug y los recortes tapados por los cuadros negros "
          "(cutout) de cold_drinks. Las latas de Coca-Cola con diseño geométrico rojo/turquesa de darians "
          "son una edición real de Coca-Cola: con rojo ≥ 0,15 se conserva la mayoría.", "",
          "Rechazadas (muestra):", "", "![rechazos de color](datos/color_rechazos.jpg)", "",
          "Aceptadas (muestra):", "", "![aceptadas](datos/color_aceptados.jpg)", ""]
    return md


def grilla_recortes(clases: list[str], por_clase: int = 8) -> None:
    ruta = DIR_RECORTES / "indice_recortes.csv"
    if not ruta.exists():
        return
    ind = pd.read_csv(ruta, keep_default_na=False)
    ok = ind[(ind["motivo_descarte"] == "") & (ind["split"] == "train")]
    lado = 96
    lienzo = Image.new("RGB", (por_clase * lado + 130, len(clases) * lado), (255, 255, 255))
    dib = ImageDraw.Draw(lienzo)
    for i, c in enumerate(clases):
        dib.text((4, i * lado + lado // 2 - 6), c, fill=(0, 0, 0))
        d = ok[ok["clase"] == c]
        d = d.assign(_o=d["ruta_recorte"].map(lambda r: hashlib.sha1(r.encode()).hexdigest())).sort_values("_o")
        for j, r in enumerate(d.head(por_clase).itertuples()):
            with Image.open(DIR_RECORTES / r.ruta_recorte) as im:
                im = im.convert("RGB")
                im.thumbnail((lado - 4, lado - 4))
                lienzo.paste(im, (130 + j * lado + 2, i * lado + 2))
    DIR_DOCS_DATOS.mkdir(parents=True, exist_ok=True)
    lienzo.save(DIR_DOCS_DATOS / "recortes_muestras.jpg", "JPEG", quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--muestras", type=int, default=12, help="imágenes por grilla")
    ap.add_argument("--sin-grillas", action="store_true")
    args = ap.parse_args()

    df = cargar_indice(RUTA_INDICE)
    remapeo = cargar_remapeo(RUTA_REMAPEO)
    clases = nombres_clases()
    nombres = nombres_por_alias()
    inspeccion = {}
    ruta_insp = DIR_RAW / "_inspeccion.json"
    if ruta_insp.exists():
        inspeccion = json.loads(ruta_insp.read_text(encoding="utf-8"))

    md = ["# Datos públicos: reporte de calidad y conteos", "",
          f"Generado por `scripts/datos_reporte.py` el {date.today().isoformat()}. "
          "Los datos viven en `DIR_DATOS` (`datos/raw`, `datos/fusion`, `datos/recortes`; no se versionan).", "",
          "Pipeline (cada paso es un script idempotente; el notebook `notebooks/02_datos.ipynb` los llama):", "",
          "1. `datos_descargar.py` → `raw/<alias>/` (última versión, formato YOLOv8; licencia en `_fuente.json`)",
          "2. `datos_remapeo.py` → borrador `config/remapeo_clases.yaml`",
          "3. `datos_indexar.py` → `fusion/_indice_imagenes.csv` (pHash, grupos anti-fuga, split)",
          "4. `datos_color.py` → `fusion/_control_color.csv` (control de color por clase)",
          "5. `datos_fusionar.py` → `fusion/detector_1clase/` y `fusion/multiclase/`",
          "6. `datos_recortes.py` → `recortes/<split>/<clase>/` + `indice_recortes.csv`",
          "7. `datos_reporte.py` → este documento", "",
          "## Criterios", "",
          "- **Grupo anti-fuga** = misma foto original: mismo proyecto de Roboflow y mismo nombre sin el sufijo "
          "`.rf.<hash>` (copias aumentadas y la misma foto en otra versión), unido con toda imagen de pHash "
          "idéntico (duplicados, incluso entre datasets). El split se sortea por grupo (80/10/10, hash "
          "determinista); un dataset con uso `test_externo` va entero a test.",
          "- Se usa 1 imagen por foto original (las copias aumentadas offline de Roboflow se descartan: YOLO y "
          "Keras aumentan en línea; `--copias-train N` permite usar más en train) y se quitan los duplicados "
          "exactos de pHash dentro de cada split. Las fotos distintas de un mismo grupo (ráfagas) se conservan.",
          "- Datasets usados en cada salida: `usar_en` de `src/datos/fuentes.py`. De lamar_grocery_color solo "
          "se usa la carpeta train/ (las etiquetas de valid/test están desplazadas).",
          "- **detector_1clase**: toda caja no descartada → `producto`; imágenes sin cajas útiles como negativos "
          "(≤ 10 % por split).",
          "- **multiclase**: solo datasets de marcas y cajas de las 9 clases; las otras bebidas quedan como fondo. "
          "Se excluyen las imágenes con cajas genéricas sin marca (`Cans`, `Bottle-Detection`...), que podrían "
          "esconder una de nuestras marcas sin etiquetar.",
          "- **recortes**: margen 5 %, se descartan cajas con lado < 32 px, borrosas (varianza del Laplaciano "
          "< 40 sobre el recorte a 128 px de alto) y duplicadas por pHash (si el duplicado está en otro split, "
          "se descarta el de val/test).", "",
          "## Datasets", ""]
    avisos = []
    tabla_lic = []
    for f in FUENTES:
        info = inspeccion.get(f["alias"], {})
        d = df[df["alias"] == f["alias"]] if not df.empty else df
        tabla_lic.append({"alias": f["alias"], "proyecto": f"{f['workspace']}/{f['project']}",
                          "versión": info.get("version_usada", "-"), "licencia": info.get("licencia", "-"),
                          "imágenes": len(d), "estado": "descargado" if len(d) else
                          f"no disponible ({info.get('error', 'sin descargar')})"})
    md += [tabla_md(pd.DataFrame(tabla_lic)), "",
           "Atribución (CC BY 4.0): cada dataset pertenece a su autor en Roboflow Universe "
           "(https://universe.roboflow.com/<workspace>/<proyecto>).", ""]
    for f in FUENTES:
        info = inspeccion.get(f["alias"], {})
        d = df[df["alias"] == f["alias"]] if not df.empty else df
        md += seccion_dataset(f, info, d, nombres.get(f["alias"], []), remapeo, df, not args.sin_grillas,
                              args.muestras)

    md += ["## Conteos finales", "", "### detector_1clase", ""]
    parte, av = seccion_fusion("detector_1clase", ["producto"])
    md += parte
    avisos += av
    md += ["### multiclase", ""]
    parte, av = seccion_fusion("multiclase", clases)
    md += parte
    avisos += av
    md += ["### Control de color (etiquetas equivocadas)", ""] + seccion_color(clases)
    md += ["### recortes (clasificador)", ""]
    if not args.sin_grillas:
        grilla_recortes(clases)
    parte, av = seccion_recortes(clases)
    md += parte
    avisos += av
    md += ["## Avisos", ""] + ([f"- {a}" for a in avisos] or ["- Ninguna clase por debajo de 300 ejemplos."]) + [""]
    md += [NOTAS_REPORTE]
    DIR_DOCS.mkdir(exist_ok=True)
    (DIR_DOCS / "datos.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte: {DIR_DOCS / 'datos.md'}")
    for a in avisos:
        print("AVISO:", a)


if __name__ == "__main__":
    main()
