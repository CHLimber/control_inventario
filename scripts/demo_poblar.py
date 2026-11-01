"""Puebla la BD con ~2 semanas de historia para una demostración (o un video) del sistema.

    python scripts/demo_poblar.py                 # 14 días: 1.ª semana variante A, después B
    python scripts/demo_poblar.py --dias 21 --variante-inicial ""   # todo con la variante B

Qué deja:
  * Los 8 productos, ventas y entradas de cada día, y un cierre por noche (salvo domingos, que el
    negocio cierra). La foto de cada cierre es un estante sintético que pasa por el modelo REAL
    (src/vision/pipeline.py); el cajero confirma con el conteo verdadero del estante, así que donde el
    modelo se equivocó queda la corrección y una observación. La primera mitad cuenta con la variante
    A y la segunda con la B: en el historial se ve que con B el cajero casi no corrige.
  * Una historia para el asistente: faltantes recurrentes de Pepsi lata, dos latas de Fanta abolladas
    retiradas y una entrada de botellas que no se registró completa (sobrante).
  * HOY sin cierre: datos/demo/estante_hoy.jpg es la foto para escanear en vivo. El stock esperado de
    hoy está armado para que ese escaneo muestre 1 Pepsi lata faltante y 1 Coca lata sobrante.
  * datos/demo/: otros estantes sintéticos para el laboratorio y el escáner, y verdad.json con el
    conteo exacto de cada foto.

ANTES de borrar copia la BD y las fotos de escaneos a datos/respaldos/<fecha_hora>/. Usa la BD de
URL_BD (.env) y las fotos de datos/escaneos: cerrar el servidor antes de correrlo, y correrlo el mismo
día de la demo (los movimientos de "hoy" quedan con la fecha del día en que se corre).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from contratos.modelos_bd import Movimiento  # noqa: E402
from src.api.pipeline import resolver_pipeline  # noqa: E402
from src.config import DIR_DATOS, DIR_ESCANEOS, nombres_clases, url_bd  # noqa: E402
from src.conciliacion import completar_cierre, confirmar_cierre, iniciar_cierre  # noqa: E402
from src.db import borrar_tablas, crear_fabrica_sesiones, crear_motor, crear_tablas  # noqa: E402
from src.db.seed import sembrar_productos  # noqa: E402
from src.sinteticos.al_vuelo import conteo_escena, generar_estante, guardar  # noqa: E402

DIR_DEMO = DIR_DATOS / "demo"
CAJEROS = ["maria", "carlos"]
HORAS_VENTA = [time(9, 40), time(12, 15), time(15, 30), time(18, 50)]
HORA_ENTRADA = time(8, 0)   # antes de la primera venta: el stock nunca queda negativo
HORA_CIERRE = time(21, 0)
# Hoy: real − esperado por clase (lo que debe mostrar el escaneo en vivo de estante_hoy.jpg)
DIFERENCIA_HOY = {"pepsi_lata": -1, "coca_lata": +1}


def respaldar(url: str) -> Path | None:
    destino = DIR_DATOS / "respaldos" / datetime.now().strftime("%Y%m%d_%H%M%S")
    hecho = False
    if url.startswith("sqlite:///"):
        bd = Path(url[len("sqlite:///"):])
        if bd.exists():
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bd, destino / bd.name)
            hecho = True
    if DIR_ESCANEOS.is_dir() and any(DIR_ESCANEOS.iterdir()):
        destino.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DIR_ESCANEOS), destino / "escaneos")
        hecho = True
    DIR_ESCANEOS.mkdir(parents=True, exist_ok=True)
    return destino if hecho else None


def dias_de_historia(n: int, hoy: date) -> list[date]:
    """Los últimos n días hábiles antes de hoy (sin domingos), del más viejo al más nuevo."""
    dias, d = [], hoy - timedelta(days=1)
    while len(dias) < n:
        if d.weekday() != 6:
            dias.append(d)
        d -= timedelta(days=1)
    return dias[::-1]


def diferencias_plantadas(reales: list[dict[str, int]], rng: random.Random) -> dict[tuple[int, str], tuple[int, str]]:
    """Lo que "pasa" en el negocio: (índice del día, clase) -> (real − esperado, observación).

    `reales`: conteo verdadero de cada noche. El sobrante va a la botella con más unidades de ese día
    (un sobrante necesita al menos una unidad en el estante).
    """
    n = len(reales)
    plan: dict[tuple[int, str], tuple[int, str]] = {}
    for i in range(1, n):
        if rng.random() < 0.45:
            plan[(i, "pepsi_lata")] = (-1, "Falta una Pepsi lata otra vez; se revisaron las cámaras.")
    plan[(n // 3, "fanta_lata")] = (-2, "2 latas de Fanta abolladas retiradas de la venta.")
    dia = n // 2
    botella = max((c for c in reales[dia] if c.endswith("_botella")), key=lambda c: reales[dia][c])
    nombre = botella.replace("_botella", "").capitalize()
    plan[(dia, botella)] = (+1, f"Sobra una {nombre} botella: la entrada del proveedor no se registró completa.")
    plan[(2 * n // 3, "coca_lata")] = (-1, "")
    return plan


def movimientos_del_dia(dia: date, anterior: dict[str, int], esperado: dict[str, int],
                        rng: random.Random, hasta: datetime | None = None) -> list[tuple[str, str, int, datetime, str]]:
    """Ventas y entradas que llevan el stock de `anterior` a `esperado` (entrada solo si hace falta)."""
    horas = [datetime.combine(dia, h) for h in HORAS_VENTA]
    hora_entrada = datetime.combine(dia, HORA_ENTRADA)
    if hasta is not None:   # hoy: todo antes de ahora (y no antes de las 00:00 de hoy)
        inicio = datetime.combine(dia, time(0, 0, 30))
        horas = [h for h in horas if h < hasta] or [max(inicio + timedelta(seconds=30), hasta - timedelta(minutes=20))]
        hora_entrada = max(inicio, min(hora_entrada, hasta - timedelta(minutes=30)))
    filas = []
    for clase in nombres_clases():
        ventas = [rng.choice([0, 1, 1, 2, 2, 3]) for _ in horas]
        entrada = esperado[clase] - anterior[clase] + sum(ventas)
        if entrada < 0:   # el stock baja más de lo sorteado: esa diferencia también se vendió
            ventas[rng.randrange(len(ventas))] -= entrada
            entrada = 0
        if entrada > 0:
            filas.append((clase, "entrada", entrada, hora_entrada, "almacen"))
        for h, v in zip(horas, ventas):
            if v > 0:
                filas.append((clase, "venta", v, h, rng.choice(CAJEROS)))
    return sorted(filas, key=lambda f: f[3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=14, help="días hábiles con cierre (antes de hoy)")
    ap.add_argument("--variante", choices=["A", "B"], default="B", help="modelo que cuenta en los cierres")
    ap.add_argument("--variante-inicial", choices=["A", "B", ""], default="A",
                    help="modelo de la primera mitad de los cierres ('' = siempre --variante)")
    ap.add_argument("--semilla", type=int, default=2026)
    a = ap.parse_args()

    url = url_bd()
    rng = random.Random(a.semilla)
    clases = nombres_clases()
    contar = {v: resolver_pipeline(None, v) for v in {a.variante, a.variante_inicial} if v}
    respaldo = respaldar(url)
    if respaldo:
        print(f"Respaldo de la BD y de las fotos anteriores: {respaldo}")

    # --- estantes: uno por cierre + el de hoy + algunos extra para el laboratorio
    hoy = date.today()
    ahora = datetime.now().replace(microsecond=0)
    dias = dias_de_historia(a.dias, hoy)
    if DIR_DEMO.exists():
        shutil.rmtree(DIR_DEMO)
    escenas = []
    for i, _ in enumerate(dias):
        esc = generar_estante(a.semilla * 100 + i, niveles=3 + (i % 2), orden=rng.choice(["ordenado", "normal", "normal"]),
                              lleno=True)
        escenas.append(esc)
    esc_hoy = generar_estante(a.semilla * 100 + 99, niveles=4, orden="ordenado", lleno=True)
    verdad = {"estante_hoy.jpg": conteo_escena(esc_hoy)}
    guardar(esc_hoy, DIR_DEMO / "estante_hoy.jpg")
    for nombre, niv, orden in [("estante_ordenado.jpg", 3, "ordenado"), ("estante_normal_5niveles.jpg", 5, "normal"),
                               ("estante_desordenado.jpg", 4, "desordenado")]:
        esc = generar_estante(a.semilla * 100 + 50 + len(verdad), niveles=niv, orden=orden, lleno=True)
        guardar(esc, DIR_DEMO / nombre)
        verdad[nombre] = conteo_escena(esc)
    (DIR_DEMO / "verdad.json").write_text(json.dumps(verdad, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- BD desde cero
    motor = crear_motor(url)
    borrar_tablas(motor)
    crear_tablas(motor)
    reales = [conteo_escena(e) for e in escenas]
    plan = diferencias_plantadas(reales, rng)
    with crear_fabrica_sesiones(motor)() as sesion:
        productos = sembrar_productos(sesion)
        # stock inicial del primer día: lo que hay en el estante esa noche + lo que se vende ese día
        stock = {c: reales[0][c] + rng.randint(2, 5) for c in clases}
        for c in clases:
            sesion.add(Movimiento(id_producto=productos[c].id, tipo="stock_inicial", cantidad=stock[c],
                                  fecha=datetime.combine(dias[0], time(7, 0)), usuario="admin"))
        for i, dia in enumerate(dias):
            real = reales[i]
            esperado = {c: real[c] - plan.get((i, c), (0, ""))[0] for c in clases}
            esperado = {c: max(0, n) for c, n in esperado.items()}
            for clase, tipo, cantidad, fecha, usuario in movimientos_del_dia(dia, stock, esperado, rng):
                sesion.add(Movimiento(id_producto=productos[clase].id, tipo=tipo, cantidad=cantidad,
                                      fecha=fecha, usuario=usuario))
            sesion.flush()

            cajero = CAJEROS[i % 2]
            variante = a.variante_inicial if a.variante_inicial and i < len(dias) // 2 else a.variante
            cierre = iniciar_cierre(sesion, usuario=cajero, variante=variante,
                                    fecha=datetime.combine(dia, HORA_CIERRE))
            carpeta = DIR_ESCANEOS / f"cierre_{cierre.id}"
            carpeta.mkdir(parents=True, exist_ok=True)
            foto = guardar(escenas[i], carpeta / "foto1.jpg")
            resultado = contar[variante]([foto])
            completar_cierre(sesion, cierre, resultado, DIR_ESCANEOS)

            notas = [plan[(i, c)][1] for c in clases if (i, c) in plan and plan[(i, c)][1]]
            corregidos = [f"{c} {resultado.conteo.get(c, 0)}→{real[c]}" for c in clases
                          if resultado.conteo.get(c, 0) != real[c]]
            if corregidos:
                notas.append("Recontado a mano: " + ", ".join(corregidos) + ".")
            ajustes = {productos[c].id: real[c] for c in clases}
            confirmar_cierre(sesion, cierre.id, ajustes, observacion=" ".join(notas) or None, usuario=cajero)
            sesion.commit()
            stock = dict(real)
            dif = sum(real[c] - esperado[c] for c in clases)
            print(f"  {dia} cierre {cierre.id} ({variante}): {sum(real.values())} envases, diferencia {dif:+d}"
                  f"{' · corregido: ' + ', '.join(corregidos) if corregidos else ''}")

        # --- hoy: movimientos hasta ahora, sin cierre (para escanear en vivo)
        real_hoy = verdad["estante_hoy.jpg"]
        esperado_hoy = {c: max(0, real_hoy[c] - DIFERENCIA_HOY.get(c, 0)) for c in clases}
        movs_hoy = movimientos_del_dia(hoy, stock, esperado_hoy, rng, hasta=ahora)
        for clase, tipo, cantidad, fecha, usuario in movs_hoy:
            sesion.add(Movimiento(id_producto=productos[clase].id, tipo=tipo, cantidad=cantidad,
                                  fecha=fecha, usuario=usuario))
        sesion.commit()

    print(f"\nBD: {url}")
    print(f"{len(dias)} cierres confirmados ({dias[0]} a {dias[-1]}); "
          f"{len(movs_hoy)} movimientos hoy, sin cierre.")
    print(f"Para escanear en vivo: {DIR_DEMO / 'estante_hoy.jpg'}")
    print("  esperado hoy:", {c: n for c, n in esperado_hoy.items() if n})
    print("  en la foto:  ", {c: n for c, n in real_hoy.items() if n})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
