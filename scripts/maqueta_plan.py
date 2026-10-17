"""Genera el plan de escenas de la maqueta: datos/maqueta/escenas_plan.csv (+ versión imprimible .md).

Cada escena dice QUÉ poner en cada nivel (de izquierda a derecha), con qué luz y fondo,
y qué variaciones hacer (envase girado, etiqueta tapada). Las cantidades por clase de la
escena son la verdad de terreno que después va a escenas.csv.

Garantías del plan (por sesión):
- respeta la capacidad física de cada nivel (una sola fila) y los envases que tenés;
- todas las clases aparecen y todas tienen escenas con CERO (se equilibra con contadores);
- hay escenas llenas, con un nivel medio vacío y con una marca entera en cero;
- cada tanto, una escena de ACERCAMIENTO a un nivel (no se mueve nada, solo se encuadra cerca);
- es reproducible: misma --semilla y mismos parámetros => mismo plan. Cada sesión usa su
  propia semilla derivada, así generar la sesión 2 más tarde da lo mismo que generarla junto a la 1.

Ejemplos (desde la raíz del repo):
    python scripts/maqueta_plan.py --sesiones 1 --escenas 26
    python scripts/maqueta_plan.py --sesiones 2,3 --escenas 30 --envases 5,pepsi_lata=4 --ancho 75
Si el plan ya tiene filas de esas sesiones, se reemplazan; las demás sesiones se conservan.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maqueta_comun import (ANGULOS, ARCHIVO_PLAN, CLASES, COLUMNAS_PLAN, FONDOS,  # noqa: E402
                           LUCES, MARCA, Maqueta, Nivel, agregar_args_maqueta, escribir_csv,
                           leer_csv, maqueta_desde_args)

PROPORCIONES = {"lleno": 0.2, "medio_vacio": 0.3, "marca_cero": 0.2}   # el resto: "normal"


def _elegir_min(rng: random.Random, opciones: list, contador: dict):
    """Elige la opción menos usada; desempata al azar."""
    minimo = min(contador[o] for o in opciones)
    return rng.choice([o for o in opciones if contador[o] == minimo])


def _llenar_nivel(rng: random.Random, maq: Maqueta, nivel: Nivel, fraccion: float,
                  ceros: set[str], apariciones: dict[str, int]) -> list[str]:
    """Devuelve la fila de envases (izquierda → derecha) de un nivel."""
    stock = maq.stock(nivel.tamano)
    elegibles = [c for c, n in stock.items() if n > 0 and c not in ceros]
    if not elegibles or fraccion <= 0:
        return []
    cap = maq.capacidad(nivel)
    objetivo = min(max(1, round(cap * fraccion)), cap, sum(stock[c] for c in elegibles))
    # primero uno de cada clase, priorizando las que menos aparecieron
    orden = sorted(elegibles, key=lambda c: (apariciones[c], rng.random()))
    asignado = {c: 0 for c in elegibles}
    for c in orden[:objetivo]:
        asignado[c] = 1
    while sum(asignado.values()) < objetivo:
        c = rng.choice([c for c in elegibles if asignado[c] < stock[c]])
        asignado[c] += 1
    return [c for c in elegibles for _ in range(asignado[c])]


def _ordenar(rng: random.Random, envases: list[str], modo: str) -> list[str]:
    if modo == "mezclado":
        envases = envases[:]
        rng.shuffle(envases)
        return envases
    clases = sorted(set(envases))
    rng.shuffle(clases)
    return [c for c in clases for _ in range(envases.count(c))]


def describir_fila(fila: list[str]) -> str:
    """['coca_lata','coca_lata','fanta_lata'] -> 'coca_lata x2 | fanta_lata'."""
    tramos: list[list] = []
    for c in fila:
        if tramos and tramos[-1][0] == c:
            tramos[-1][1] += 1
        else:
            tramos.append([c, 1])
    return " | ".join(c if n == 1 else f"{c} x{n}" for c, n in tramos)


def _variaciones(rng: random.Random, filas: dict[int, list[str]]) -> list[str]:
    envases = [(k, c) for k, fila in filas.items() for c in fila]
    if not envases:
        return []
    salida = []
    if rng.random() < 0.4:
        elegidos = rng.sample(envases, k=min(len(envases), rng.choice([1, 2])))
        for k, c in sorted(set(elegidos)):
            n = elegidos.count((k, c))
            salida.append(f"girar {n} {c} (nivel {k}) de costado/espalda")
    if rng.random() < 0.35:
        k, c = rng.choice(envases)
        salida.append(f"tapar media etiqueta de 1 {c} (nivel {k}) con un papel o la tira de precio")
    return salida


def _tipos(rng: random.Random, n_base: int) -> list[str]:
    tipos = []
    for tipo, p in PROPORCIONES.items():
        tipos += [tipo] * round(n_base * p)
    tipos = (tipos + ["normal"] * n_base)[:n_base]
    rng.shuffle(tipos)
    if "lleno" in tipos:                      # la escena 1 llena: es la más fácil de armar
        tipos.remove("lleno")
        tipos.insert(0, "lleno")
    return tipos


def generar_sesion(maq: Maqueta, sesion: int, n_escenas: int, semilla: int = 42,
                   luces: list[str] | None = None, fondos: list[str] | None = None,
                   acercamiento_cada: int = 6, celular: str = "") -> list[dict]:
    """Lista de filas (dict con COLUMNAS_PLAN) para una sesión."""
    rng = random.Random(semilla * 1000 + sesion)
    luces = luces or LUCES
    fondos = fondos or FONDOS
    niveles = maq.niveles()
    es_acerc = [bool(acercamiento_cada) and e > 1 and e % acercamiento_cada == 0
                for e in range(1, n_escenas + 1)]
    n_base = n_escenas - sum(es_acerc)
    tipos = _tipos(rng, n_base)
    # fondos en bloques contiguos (se cambia pocas veces); el orden rota entre sesiones
    rot = (sesion - 1) % len(fondos)
    orden_fondos = fondos[rot:] + fondos[:rot]
    marcas = sorted(set(MARCA.values()))
    apariciones = {c: 0 for c in CLASES}
    ceros_clase = {c: 0 for c in CLASES}
    ceros_marca = {m: 0 for m in marcas}
    medio_vacio = {n.numero: 0 for n in niveles}
    acerc_nivel = {n.numero: 0 for n in niveles}

    filas_csv: list[dict] = []
    ultima_base: dict | None = None
    i_base = 0
    for escena in range(1, n_escenas + 1):
        if es_acerc[escena - 1] and ultima_base is not None:
            base = ultima_base
            con_envases = [k for k in (1, 2, 3) if base["_filas"][k]]
            k = _elegir_min(rng, con_envases, acerc_nivel)
            acerc_nivel[k] += 1
            fila = base["_filas"][k]
            conteo = {c: fila.count(c) for c in CLASES}
            filas_csv.append({
                "sesion": sesion, "escena": escena, "fecha": "", "luz": base["luz"],
                "fondo": base["fondo"], **conteo,
                "observacion": f"acercamiento al nivel {k} de la escena {base['escena']}: NO mover nada",
                "distancia": f"nivel_{k}", "celular": celular, "tipo": "acercamiento",
                "orden": base["orden"], **{f"nivel_{j}": (describir_fila(fila) if j == k else "")
                                           for j in (1, 2, 3)},
                "variaciones": "; ".join(v for v in base["variaciones"].split("; ")
                                         if f"(nivel {k})" in v),
                "total": len(fila)})
            continue

        tipo = tipos[i_base]
        bloque = min(i_base * len(orden_fondos) // max(n_base, 1), len(orden_fondos) - 1)
        fondo = orden_fondos[bloque]
        luz = luces[(i_base + bloque + sesion) % len(luces)]
        i_base += 1

        ceros: set[str] = set()
        notas = []
        if tipo == "marca_cero":
            marca = _elegir_min(rng, marcas, ceros_marca)
            ceros |= {c for c in CLASES if MARCA[c] == marca}
            notas.append(f"sin {marca}")
        elif tipo != "lleno" and rng.random() < 0.4:
            c = _elegir_min(rng, CLASES, ceros_clase)
            ceros.add(c)
            notas.append(f"sin {c}")
        fracciones = {n.numero: 1.0 if tipo == "lleno" else rng.uniform(0.55, 1.0) for n in niveles}
        if tipo == "medio_vacio":
            k = _elegir_min(rng, [n.numero for n in niveles], medio_vacio)
            medio_vacio[k] += 1
            fracciones[k] = rng.uniform(0.2, 0.5)
            notas.insert(0, f"nivel {k} medio vacío")
        if tipo == "lleno":
            notas.insert(0, "lleno")

        modo = rng.choice(["agrupado", "mezclado"])
        filas = {n.numero: _ordenar(rng, _llenar_nivel(rng, maq, n, fracciones[n.numero], ceros,
                                                      apariciones), modo)
                 for n in niveles}
        conteo = {c: sum(f.count(c) for f in filas.values()) for c in CLASES}
        for c in CLASES:
            if conteo[c]:
                apariciones[c] += 1
            else:
                ceros_clase[c] += 1
        for m in marcas:
            if all(conteo[c] == 0 for c in CLASES if MARCA[c] == m):
                ceros_marca[m] += 1
        variaciones = _variaciones(rng, filas)
        if variaciones:
            notas.append("con variaciones")
        fila_csv = {
            "sesion": sesion, "escena": escena, "fecha": "", "luz": luz, "fondo": fondo, **conteo,
            "observacion": "; ".join(notas), "distancia": "completo", "celular": celular,
            "tipo": tipo, "orden": modo,
            **{f"nivel_{k}": describir_fila(filas[k]) for k in (1, 2, 3)},
            "variaciones": "; ".join(variaciones), "total": sum(conteo.values()), "_filas": filas}
        filas_csv.append(fila_csv)
        ultima_base = fila_csv

    errores = [f"escena {f['escena']}: {e}" for f in filas_csv
               for e in maq.problemas_conteo({c: f[c] for c in CLASES})]
    assert not errores, errores
    for f in filas_csv:
        f.pop("_filas", None)
    return filas_csv


def resumen(filas: list[dict]) -> str:
    lineas = [f"{'clase':16s} {'escenas':>8s} {'con cero':>9s} {'máx':>4s}"]
    for c in CLASES:
        n = [int(f[c]) for f in filas]
        lineas.append(f"{c:16s} {sum(x > 0 for x in n):8d} {sum(x == 0 for x in n):9d} {max(n):4d}")
    tipos: dict[str, int] = {}
    for f in filas:
        tipos[f["tipo"]] = tipos.get(f["tipo"], 0) + 1
    lineas.append("tipos: " + ", ".join(f"{k}={v}" for k, v in sorted(tipos.items())))
    return "\n".join(lineas)


def _fila_md(texto: str) -> str:
    """En lo imprimible se escribe la marca (el nivel ya dice el formato)."""
    if not texto:
        return "*(vacío)*"
    partes = []
    for tramo in texto.split(" | "):
        clase, _, n = tramo.partition(" x")
        partes.append(f"{MARCA[clase]}" + (f" ×{n}" if n else ""))
    return " · ".join(partes)


def imprimible(maq: Maqueta, sesion: int, filas: list[dict], semilla: int) -> str:
    niv = maq.niveles()
    out = [f"# Plan de la sesión {sesion} — maqueta ({len(filas)} escenas)", "",
           f"Generado con `python scripts/maqueta_plan.py --sesiones {sesion} --escenas {len(filas)} "
           f"--semilla {semilla} --ancho {maq.ancho_cm:g}` (mismos parámetros ⇒ mismo plan).", "",
           "**Capacidad por nivel (una sola fila):** "
           + " · ".join(f"nivel {n.numero} ({n.nombre}) = {maq.capacidad(n)}" for n in niv), "",
           "**Envases que se suponen:** "
           + ", ".join(f"{c} {maq.envases[c]}" for c in CLASES)
           + " (de las botellas, " + ", ".join(f"{c} {g} grandes" for c, g in maq.grandes.items()) + ").", "",
           "Niveles numerados de ABAJO hacia ARRIBA. Cada fila se lee de IZQUIERDA a DERECHA "
           "mirando el estante de frente. En los niveles 1 y 2 van botellas y en el 3 latas, "
           "así que solo se escribe la marca.", "",
           "Por escena: armar → revisar contra la tabla → anotar en `escenas.csv` (o tildar acá) → "
           f"3 fotos en este orden: {', '.join(ANGULOS)} (frontal, 20° a la izquierda, 20° a la derecha).",
           "", "## Antes de empezar", "",
           "- [ ] Celular con lente limpio, formato JPG (no HEIC), sin modo retrato ni filtros, fecha y hora correctas",
           "- [ ] Batería > 50 % y espacio libre", "- [ ] Envases limpios, sin agua derramada; tapas puestas",
           "- [ ] Fondos a mano: " + ", ".join(sorted({f['fondo'] for f in filas})),
           "- [ ] Luces a mano: " + ", ".join(sorted({f['luz'] for f in filas})),
           "- [ ] Carpeta/álbum vacío para esta sesión: SOLO las fotos de escenas, de a 3 "
           "(si una sale mal, repetila y BORRÁ la mala en el momento)", ""]
    for f in filas:
        e = int(f["escena"])
        out += [f"## Escena {e:02d} — {f['tipo']} · luz **{f['luz']}** · fondo **{f['fondo']}** · "
                f"{f['distancia']} · total {f['total']}", ""]
        if f["observacion"]:
            out += [f"*{f['observacion']}*", ""]
        out += ["| Nivel | Izquierda → derecha |", "|---|---|"]
        for n in reversed(niv):
            if f["tipo"] != "acercamiento" or f[f"nivel_{n.numero}"]:
                out.append(f"| {n.numero} · {n.nombre} | {_fila_md(f[f'nivel_{n.numero}'])} |")
        out.append("")
        if f["variaciones"]:
            out += [f"Variaciones: {f['variaciones']}", ""]
        cant = ", ".join(f"{c} {f[c]}" for c in CLASES if int(f[c]))
        cero = ", ".join(c for c in CLASES if not int(f[c]))
        out += [f"Cantidades: {cant}", "", f"En cero: {cero or '—'}", "",
                "☐ armada y revisada   ☐ anotada   "
                + "   ".join(f"☐ foto {i} {a}" for i, a in enumerate(ANGULOS, 1)), ""]
    out += ["## Al terminar", "", "- [ ] Pasar las fotos a la PC y correr "
            f"`python scripts/maqueta_ingesta.py <carpeta> --sesion {sesion} --simular`",
            "- [ ] Si todo cuadra, repetir sin `--simular` (y con `--copiar-plan` si no anotaste a mano)",
            "- [ ] Anotar en la observación cualquier escena que haya salido distinta al plan", "",
            "```", resumen(filas), "```", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sesiones", default="1", help="sesiones a generar, p. ej. '1' o '1,2,3' (def. 1)")
    p.add_argument("--escenas", type=int, default=26, help="escenas por sesión (def. 26)")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--luces", default=",".join(LUCES), help=f"def. {','.join(LUCES)}")
    p.add_argument("--fondos", default=",".join(FONDOS), help=f"def. {','.join(FONDOS)}")
    p.add_argument("--acercamiento-cada", type=int, default=6,
                   help="cada cuántas escenas va un acercamiento a un nivel (0 = nunca; def. 6)")
    p.add_argument("--celular", default="", help="modelo del celular (se copia a la columna celular)")
    p.add_argument("--salida", type=Path, default=ARCHIVO_PLAN, help="CSV del plan")
    p.add_argument("--dir-imprimible", type=Path, default=None,
                   help="carpeta del .md imprimible (def. la del CSV)")
    p.add_argument("--sin-imprimible", action="store_true")
    agregar_args_maqueta(p)
    args = p.parse_args(argv)

    maq = maqueta_desde_args(args)
    sesiones = [int(s) for s in args.sesiones.split(",") if s.strip()]
    if args.escenas < 1:
        p.error("--escenas debe ser ≥ 1")
    for n in maq.niveles():
        if maq.capacidad(n) < 1:
            p.error(f"con --ancho {args.ancho} no entra ni un envase en el nivel {n.numero}")
    print("Capacidad por nivel: " + ", ".join(f"nivel {n.numero} ({n.nombre}) {maq.capacidad(n)}"
                                               for n in maq.niveles()))
    _, previas = leer_csv(args.salida)
    filas = [f for f in previas if int(f["sesion"]) not in sesiones]
    for s in sesiones:
        nuevas = generar_sesion(maq, s, args.escenas, args.semilla,
                                [x for x in args.luces.split(",") if x],
                                [x for x in args.fondos.split(",") if x],
                                args.acercamiento_cada, args.celular)
        filas += nuevas
        print(f"\nSesión {s}: {len(nuevas)} escenas\n{resumen(nuevas)}")
        if not args.sin_imprimible:
            carpeta = args.dir_imprimible or args.salida.parent
            carpeta.mkdir(parents=True, exist_ok=True)
            md = carpeta / f"escenas_plan_sesion{s:02d}.md"
            md.write_text(imprimible(maq, s, nuevas, args.semilla), encoding="utf-8")
            print(f"Imprimible: {md}")
    filas.sort(key=lambda f: (int(f["sesion"]), int(f["escena"])))
    escribir_csv(args.salida, COLUMNAS_PLAN, filas)
    print(f"\nPlan: {args.salida} ({len(filas)} filas en total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
