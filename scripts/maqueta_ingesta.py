"""Ingesta de fotos de la maqueta: del celular a datos/maqueta/sesion_XX/ con nombres de escena.

Toma una carpeta con las fotos de UNA sesión, las ordena (fecha EXIF o nombre) y las agrupa de a
--fotos-por-escena (def. 3: frontal, izq20, der20). La foto i va a la escena
escena_inicial + i // 3 y se COPIA (nunca se mueve ni se pisa) como

    datos/maqueta/sesion_XX/sesionXX_escenaYY_fotoZ.jpg

Además:
- verifica que cada escena tenga su fila en escenas.csv; si falta y está en escenas_plan.csv,
  la copia con --copiar-plan (o pregunta, si se corre en una terminal);
- avisa de fotos sobrantes (grupo incompleto al final), escenas sin fila, escenas anotadas
  sin fotos, conteos imposibles (negativos, más envases de los que hay, más de lo que entra
  en el estante) y grupos sospechosos (fotos de una misma escena muy separadas en el tiempo);
- registra cada foto en datos/maqueta/fotos.csv (archivo, escena, ángulo, origen, fecha, celular).

Con --simular solo muestra lo que haría. Sale con código 1 si encontró problemas.

Ejemplo:
    python scripts/maqueta_ingesta.py "D:/DCIM/Camera/sesion1" --sesion 1 --simular
    python scripts/maqueta_ingesta.py "D:/DCIM/Camera/sesion1" --sesion 1 --copiar-plan
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maqueta_comun import (ANGULOS, ARCHIVO_ESCENAS, ARCHIVO_PLAN,  # noqa: E402
                           COLUMNAS_ESCENAS, DIR_MAQUETA, Maqueta, agregar_args_maqueta,
                           conteo_de_fila, dir_sesion, escribir_csv, leer_csv,
                           maqueta_desde_args, nombre_foto)

EXT_FOTO = {".jpg", ".jpeg"}
COLUMNAS_FOTOS = ["archivo", "sesion", "escena", "foto", "angulo", "origen", "fecha_foto", "celular"]
TAG_EXIF_IFD, TAG_FECHA_ORIGINAL, TAG_FECHA, TAG_MARCA, TAG_MODELO = 0x8769, 36867, 306, 271, 272


@dataclass
class Foto:
    ruta: Path
    fecha: datetime
    fuente_fecha: str      # "exif" o "archivo"
    celular: str = ""


def leer_foto(ruta: Path) -> Foto:
    """Fecha de toma (EXIF DateTimeOriginal > DateTime > fecha del archivo) y modelo del celular."""
    fecha, fuente, celular = None, "archivo", ""
    try:
        with Image.open(ruta) as img:
            exif = img.getexif()
            texto = exif.get_ifd(TAG_EXIF_IFD).get(TAG_FECHA_ORIGINAL) or exif.get(TAG_FECHA)
            if texto:
                fecha = datetime.strptime(str(texto).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S")
                fuente = "exif"
            marca = str(exif.get(TAG_MARCA, "")).strip("\x00 ")
            modelo = str(exif.get(TAG_MODELO, "")).strip("\x00 ")
            celular = modelo if modelo.lower().startswith(marca.lower()) else f"{marca} {modelo}".strip()
    except (OSError, ValueError):
        pass
    if fecha is None:
        fecha = datetime.fromtimestamp(ruta.stat().st_mtime)
    return Foto(ruta, fecha, fuente, celular)


@dataclass
class Informe:
    acciones: list[str]
    problemas: list[str]
    avisos: list[str]

    def imprimir(self, simular: bool) -> None:
        print(("SIMULACIÓN: no se escribe nada.\n" if simular else "") + "\n".join(self.acciones))
        for titulo, items in (("AVISOS", self.avisos), ("PROBLEMAS", self.problemas)):
            if items:
                print(f"\n{titulo} ({len(items)}):")
                for x in items:
                    print("  - " + x)
        print("\nResultado: " + ("sin problemas." if not self.problemas
                                 else f"{len(self.problemas)} problema(s); revisalos antes de seguir."))


def ingerir(origen: Path, sesion: int, *, fotos_por_escena: int = 3, escena_inicial: int = 1,
            orden: str = "exif", destino: Path = DIR_MAQUETA, escenas_csv: Path = ARCHIVO_ESCENAS,
            plan_csv: Path = ARCHIVO_PLAN, copiar_plan: bool | None = False, simular: bool = False,
            angulos: list[str] | None = None, maqueta: Maqueta | None = None,
            max_minutos: float = 5.0) -> Informe:
    """Hace (o simula) la ingesta y devuelve el informe. copiar_plan=None => preguntar."""
    angulos = angulos or ANGULOS
    maqueta = maqueta or Maqueta()
    inf = Informe([], [], [])
    if not origen.is_dir():
        inf.problemas.append(f"no existe la carpeta {origen}")
        return inf

    archivos = sorted(p for p in origen.iterdir() if p.is_file())
    fotos_raw = [p for p in archivos if p.suffix.lower() in EXT_FOTO]
    for p in archivos:
        if p.suffix.lower() not in EXT_FOTO:
            extra = " (configurá la cámara en JPG / 'Más compatible')" if p.suffix.lower() == ".heic" else ""
            inf.avisos.append(f"se ignora {p.name}: no es .jpg{extra}")
    fotos = [leer_foto(p) for p in fotos_raw]
    if orden == "exif":
        sin_exif = [f.ruta.name for f in fotos if f.fuente_fecha != "exif"]
        if sin_exif:
            inf.avisos.append(f"{len(sin_exif)} foto(s) sin fecha EXIF, se usa la del archivo: "
                              + ", ".join(sin_exif[:5]) + (" ..." if len(sin_exif) > 5 else ""))
        fotos.sort(key=lambda f: (f.fecha, f.ruta.name))
    else:
        fotos.sort(key=lambda f: f.ruta.name)
    if not fotos:
        inf.problemas.append(f"no hay fotos .jpg en {origen}")
        return inf

    completas = len(fotos) // fotos_por_escena * fotos_por_escena
    sobrantes = fotos[completas:]
    if sobrantes:
        inf.problemas.append(
            f"{len(sobrantes)} foto(s) sobrantes al final (no completan una escena de "
            f"{fotos_por_escena}); NO se copian: " + ", ".join(f.ruta.name for f in sobrantes)
            + ". ¿Sobra una foto repetida o falta una en alguna escena? Revisá el orden.")
    grupos: dict[int, list[Foto]] = {}
    for i, f in enumerate(fotos[:completas]):
        grupos.setdefault(escena_inicial + i // fotos_por_escena, []).append(f)
    for escena, grupo in grupos.items():
        minutos = (grupo[-1].fecha - grupo[0].fecha).total_seconds() / 60
        if minutos > max_minutos:
            inf.avisos.append(f"escena {escena}: sus fotos abarcan {minutos:.0f} min "
                              f"({grupo[0].ruta.name} … {grupo[-1].ruta.name}); ¿se corrió el orden?")

    # --- verdad de terreno -------------------------------------------------------------
    columnas, filas = leer_csv(escenas_csv)
    columnas = columnas or COLUMNAS_ESCENAS[:]
    columnas += [c for c in COLUMNAS_ESCENAS if c not in columnas]   # agrega distancia/celular al final
    _, plan = leer_csv(plan_csv)

    def clave(f: dict) -> tuple[int, int] | None:
        try:
            return int(f["sesion"]), int(f["escena"])
        except (KeyError, TypeError, ValueError):
            return None

    por_clave = {clave(f): f for f in filas if clave(f)}
    plan_por_clave = {clave(f): f for f in plan if clave(f)}
    faltan_en_plan = []
    for escena, grupo in grupos.items():
        if (sesion, escena) in por_clave:
            continue
        if (sesion, escena) in plan_por_clave:
            faltan_en_plan.append(escena)
        else:
            inf.problemas.append(f"escena {escena}: no tiene fila en {escenas_csv.name} ni en el plan; "
                                 "anotá a mano lo que había en el estante")
    if faltan_en_plan:
        if copiar_plan is None:
            copiar_plan = (not simular and sys.stdin.isatty() and input(
                f"{len(faltan_en_plan)} escena(s) sin fila en {escenas_csv.name} están en el plan "
                f"({_rango(faltan_en_plan)}). ¿Copiarlas del plan? [s/N] ").strip().lower() in ("s", "si", "sí"))
        if copiar_plan:
            for escena in faltan_en_plan:
                nueva = {c: plan_por_clave[(sesion, escena)].get(c, "") for c in columnas}
                grupo = grupos[escena]
                nueva["fecha"] = nueva.get("fecha") or grupo[0].fecha.strftime("%Y-%m-%d")
                nueva["celular"] = nueva.get("celular") or grupo[0].celular
                filas.append(nueva)
                por_clave[(sesion, escena)] = nueva
            inf.acciones.append(f"{'Copiaría' if simular else 'Copio'} {len(faltan_en_plan)} fila(s) "
                                f"del plan a {escenas_csv.name}: escenas {_rango(faltan_en_plan)}")
            inf.avisos.append("las filas copiadas del plan suponen que armaste EXACTAMENTE lo planeado; "
                              "si cambiaste algo, corregí escenas.csv")
        else:
            inf.problemas.append(f"escenas sin fila en {escenas_csv.name} (están en el plan): "
                                 f"{_rango(faltan_en_plan)}. Volvé a correr con --copiar-plan")

    anotadas = sorted(e for (s, e) in por_clave if s == sesion)
    sin_fotos = [e for e in anotadas if e not in grupos and not _ya_copiada(destino, sesion, e)]
    if sin_fotos:
        inf.avisos.append(f"escenas anotadas en {escenas_csv.name} sin fotos: {_rango(sin_fotos)}")
    for escena in anotadas:
        conteo, errores = conteo_de_fila(por_clave[(sesion, escena)])
        for e in errores + maqueta.problemas_conteo(conteo):
            inf.problemas.append(f"escena {escena}: conteo imposible: {e}")

    # --- copia de fotos ----------------------------------------------------------------
    carpeta = dir_sesion(destino, sesion)
    _, manifiesto = leer_csv(destino / "fotos.csv")
    en_manifiesto = {f["archivo"] for f in manifiesto}
    copiadas = omitidas = 0
    for escena, grupo in grupos.items():
        for z, foto in enumerate(grupo, 1):
            dest = carpeta / nombre_foto(sesion, escena, z)
            rel = dest.relative_to(destino).as_posix()
            if dest.exists():
                if filecmp.cmp(foto.ruta, dest, shallow=False):
                    omitidas += 1
                else:
                    inf.problemas.append(f"{rel} ya existe y es OTRA foto: no se pisa ({foto.ruta.name} "
                                         "queda sin copiar). Revisá --escena-inicial o borrá a mano.")
                    continue
            else:
                inf.acciones.append(f"{foto.ruta.name} -> {rel}")
                if not simular:
                    carpeta.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(foto.ruta, dest)
                copiadas += 1
            if rel not in en_manifiesto:
                manifiesto.append({"archivo": rel, "sesion": sesion, "escena": escena, "foto": z,
                                   "angulo": angulos[z - 1] if z <= len(angulos) else "",
                                   "origen": foto.ruta.name,
                                   "fecha_foto": foto.fecha.isoformat(timespec="seconds"),
                                   "celular": foto.celular})
                en_manifiesto.add(rel)
    if omitidas:
        inf.avisos.append(f"{omitidas} foto(s) ya estaban copiadas (idénticas); se omiten")
    inf.acciones.append(f"{'Copiaría' if simular else 'Copiadas'}: {copiadas} foto(s) en "
                        f"{len(grupos)} escena(s) ({_rango(sorted(grupos))}) -> {carpeta}")

    if not simular:
        filas.sort(key=lambda f: clave(f) or (10**6, 10**6))
        escribir_csv(escenas_csv, columnas, filas)
        manifiesto.sort(key=lambda f: (int(f["sesion"]), int(f["escena"]), int(f["foto"])))
        escribir_csv(destino / "fotos.csv", COLUMNAS_FOTOS, manifiesto)
    return inf


def _ya_copiada(destino: Path, sesion: int, escena: int) -> bool:
    return (dir_sesion(destino, sesion) / nombre_foto(sesion, escena, 1)).exists()


def _rango(escenas: list[int]) -> str:
    """[1,2,3,5,7,8] -> '1-3, 5, 7-8'."""
    tramos: list[list[int]] = []
    for e in sorted(escenas):
        if tramos and e == tramos[-1][1] + 1:
            tramos[-1][1] = e
        else:
            tramos.append([e, e])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in tramos)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("origen", type=Path, help="carpeta con las fotos de UNA sesión (copiadas del celular)")
    p.add_argument("--sesion", type=int, required=True)
    p.add_argument("--fotos-por-escena", type=int, default=3)
    p.add_argument("--escena-inicial", type=int, default=1,
                   help="número de la escena de la primera foto (para cargar una sesión en partes)")
    p.add_argument("--orden", choices=["exif", "nombre"], default="exif",
                   help="exif = por fecha de toma (def.); nombre = alfabético")
    p.add_argument("--angulos", default=",".join(ANGULOS), help="nombre de cada foto del grupo, en orden")
    p.add_argument("--destino", type=Path, default=DIR_MAQUETA)
    p.add_argument("--escenas-csv", type=Path, default=None, help="def. <destino>/escenas.csv")
    p.add_argument("--plan-csv", type=Path, default=None, help="def. <destino>/escenas_plan.csv")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--copiar-plan", action="store_true", default=None,
                   help="copiar del plan las filas que falten en escenas.csv sin preguntar")
    g.add_argument("--no-copiar-plan", dest="copiar_plan", action="store_false")
    p.add_argument("--max-minutos", type=float, default=5.0,
                   help="avisar si las fotos de una escena abarcan más minutos que esto")
    p.add_argument("--simular", action="store_true", help="solo mostrar lo que haría")
    agregar_args_maqueta(p)
    args = p.parse_args(argv)
    if args.fotos_por_escena < 1:
        p.error("--fotos-por-escena debe ser ≥ 1")

    inf = ingerir(args.origen, args.sesion, fotos_por_escena=args.fotos_por_escena,
                  escena_inicial=args.escena_inicial, orden=args.orden, destino=args.destino,
                  escenas_csv=args.escenas_csv or args.destino / "escenas.csv",
                  plan_csv=args.plan_csv or args.destino / "escenas_plan.csv",
                  copiar_plan=args.copiar_plan, simular=args.simular,
                  angulos=[a for a in args.angulos.split(",") if a],
                  maqueta=maqueta_desde_args(args), max_minutos=args.max_minutos)
    inf.imprimir(args.simular)
    return 1 if inf.problemas else 0


if __name__ == "__main__":
    sys.exit(main())
