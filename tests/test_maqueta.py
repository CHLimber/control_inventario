"""Tests de scripts/maqueta_plan.py y scripts/maqueta_ingesta.py (todo en tmp_path, nada en el repo)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import maqueta_ingesta as ing  # noqa: E402
import maqueta_plan as plan  # noqa: E402
from maqueta_comun import (BOTELLAS, CLASES, COLUMNAS_ESCENAS, LATAS, MARCA, Maqueta,  # noqa: E402
                           escribir_csv, leer_csv, parsear_por_clase)

COLUMNAS_ORIGINALES = ["sesion", "escena", "fecha", "luz", "fondo", *CLASES, "observacion"]


# ---------------------------------------------------------------- maqueta / parámetros
def test_capacidades_por_defecto():
    m = Maqueta()
    assert [m.capacidad(n) for n in m.niveles()] == [6, 9, 9]
    assert Maqueta(ancho_cm=60).capacidad(m.niveles()[2]) < 9


def test_parsear_por_clase():
    assert parsear_por_clase(None, 5, CLASES) == {c: 5 for c in CLASES}
    r = parsear_por_clase("4,coca_lata=6", 5, CLASES)
    assert r["coca_lata"] == 6 and r["fanta_lata"] == 4
    with pytest.raises(ValueError):
        parsear_por_clase("cerveza=3", 5, CLASES)


def test_problemas_conteo():
    m = Maqueta()
    assert m.problemas_conteo({c: 1 for c in CLASES}) == []
    assert any("negativo" in e for e in m.problemas_conteo({"coca_lata": -1}))
    assert any("supera" in e for e in m.problemas_conteo({"coca_lata": 6}))
    assert any("nivel 3" in e for e in m.problemas_conteo({c: 3 for c in LATAS}))   # 12 > 9


def test_columnas_escenas_conservan_las_originales():
    assert COLUMNAS_ESCENAS[:len(COLUMNAS_ORIGINALES)] == COLUMNAS_ORIGINALES


# ---------------------------------------------------------------- plan
@pytest.mark.parametrize("n", [20, 26, 30])
def test_plan_respeta_capacidad_y_cubre_clases(n):
    m = Maqueta()
    filas = plan.generar_sesion(m, 1, n, semilla=7)
    assert len(filas) == n and [f["escena"] for f in filas] == list(range(1, n + 1))
    for f in filas:
        conteo = {c: f[c] for c in CLASES}
        assert m.problemas_conteo(conteo) == []
        assert f["total"] == sum(conteo.values())
        # lo que dicen las filas de cada nivel coincide con las cantidades
        desc = " | ".join(f[f"nivel_{k}"] for k in (1, 2, 3) if f[f"nivel_{k}"])
        for c in CLASES:
            n_desc = sum(int(t.partition(" x")[2] or 1) for t in desc.split(" | ") if t.partition(" x")[0] == c)
            assert n_desc == f[c]
        caps = [m.capacidad(nv) for nv in m.niveles()]
        for k in (1, 2, 3):
            items = [t for t in f[f"nivel_{k}"].split(" | ") if t]
            assert sum(int(t.partition(" x")[2] or 1) for t in items) <= caps[k - 1]
    for c in CLASES:
        assert any(f[c] > 0 for f in filas), f"{c} nunca aparece"
        assert any(f[c] == 0 for f in filas), f"{c} nunca está en cero"
    tipos = {f["tipo"] for f in filas}
    assert {"lleno", "medio_vacio", "marca_cero", "acercamiento"} <= tipos
    assert any(all(f[c] == 0 for c in CLASES if MARCA[c] == "Fanta") or
               all(f[c] == 0 for c in CLASES if MARCA[c] == "Coca-Cola") for f in filas)


def test_plan_reproducible_y_distinto_por_sesion():
    m = Maqueta()
    assert plan.generar_sesion(m, 1, 24, semilla=3) == plan.generar_sesion(m, 1, 24, semilla=3)
    assert plan.generar_sesion(m, 1, 24, semilla=3) != plan.generar_sesion(m, 2, 24, semilla=3)


def test_plan_respeta_envases_limitados():
    m = Maqueta(envases={c: 2 for c in CLASES} | {"pepsi_lata": 0}, grandes={c: 1 for c in BOTELLAS})
    for f in plan.generar_sesion(m, 1, 25, semilla=1):
        assert f["pepsi_lata"] == 0
        assert all(f[c] <= 2 for c in CLASES)


def test_acercamiento_es_un_nivel_de_la_escena_anterior():
    filas = plan.generar_sesion(Maqueta(), 1, 12, semilla=5, acercamiento_cada=6)
    acerc = [f for f in filas if f["tipo"] == "acercamiento"]
    assert [f["escena"] for f in acerc] == [6, 12]
    for f in acerc:
        base = filas[f["escena"] - 2]
        k = int(f["distancia"].split("_")[1])
        assert f[f"nivel_{k}"] == base[f"nivel_{k}"]
        assert (f["luz"], f["fondo"]) == (base["luz"], base["fondo"])


def test_plan_cli_reemplaza_solo_su_sesion(tmp_path):
    salida = tmp_path / "escenas_plan.csv"
    assert plan.main(["--sesiones", "1,2", "--escenas", "10", "--salida", str(salida)]) == 0
    _, filas = leer_csv(salida)
    assert len(filas) == 20 and (tmp_path / "escenas_plan_sesion02.md").exists()
    antes_s1 = [f for f in filas if f["sesion"] == "1"]
    assert plan.main(["--sesiones", "2", "--escenas", "5", "--salida", str(salida), "--sin-imprimible"]) == 0
    _, filas = leer_csv(salida)
    assert [f for f in filas if f["sesion"] == "1"] == antes_s1
    assert sum(f["sesion"] == "2" for f in filas) == 5
    md = (tmp_path / "escenas_plan_sesion01.md").read_text(encoding="utf-8")
    assert "## Escena 01" in md and "## Escena 10" in md


# ---------------------------------------------------------------- ingesta
def crear_fotos(carpeta: Path, n: int, inicio=datetime(2026, 10, 18, 10, 0, 0),
                paso=timedelta(seconds=20), con_exif=True, nombres_al_reves=False) -> list[Path]:
    """Fotos JPG chicas con fecha EXIF; el nombre puede ir en orden inverso a la fecha."""
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for i in range(n):
        img = Image.new("RGB", (64, 48), (i * 9 % 256, 80, 160))
        exif = Image.Exif()
        if con_exif:
            exif[306] = (inicio + i * paso).strftime("%Y:%m:%d %H:%M:%S")
            exif[271], exif[272] = "Xiaomi", "Redmi Note 12"
        nombre = f"IMG_{(n - i if nombres_al_reves else i):04d}.jpg"
        ruta = carpeta / nombre
        img.save(ruta, exif=exif)
        rutas.append(ruta)
    return rutas


@pytest.fixture
def entorno(tmp_path):
    destino = tmp_path / "maqueta"
    destino.mkdir()
    escribir_csv(destino / "escenas.csv", COLUMNAS_ORIGINALES, [])
    filas_plan = plan.generar_sesion(Maqueta(), 1, 6, semilla=11)
    escribir_csv(destino / "escenas_plan.csv", list(filas_plan[0].keys()), filas_plan)
    return tmp_path, destino


def _ingerir(origen, destino, **kw):
    return ing.ingerir(origen, 1, destino=destino, escenas_csv=destino / "escenas.csv",
                       plan_csv=destino / "escenas_plan.csv", **kw)


def test_ingesta_simular_no_escribe(entorno):
    tmp, destino = entorno
    crear_fotos(tmp / "cel", 6)
    antes = (destino / "escenas.csv").read_text(encoding="utf-8")
    inf = _ingerir(tmp / "cel", destino, simular=True, copiar_plan=True)
    assert not (destino / "sesion_01").exists() and not (destino / "fotos.csv").exists()
    assert (destino / "escenas.csv").read_text(encoding="utf-8") == antes
    assert any("sesion01_escena02_foto3.jpg" in a for a in inf.acciones)
    assert inf.problemas == []


def test_ingesta_copia_ordena_por_exif_y_copia_plan(entorno):
    tmp, destino = entorno
    fuentes = crear_fotos(tmp / "cel", 6, nombres_al_reves=True)
    inf = _ingerir(tmp / "cel", destino, copiar_plan=True)
    assert inf.problemas == []
    carpeta = destino / "sesion_01"
    assert sorted(p.name for p in carpeta.iterdir()) == [
        f"sesion01_escena{e:02d}_foto{z}.jpg" for e in (1, 2) for z in (1, 2, 3)]
    # la primera por fecha es la de nombre más alto
    assert (carpeta / "sesion01_escena01_foto1.jpg").read_bytes() == fuentes[0].read_bytes()
    assert fuentes[0].exists()                                   # copia, no mueve
    columnas, filas = leer_csv(destino / "escenas.csv")
    assert columnas[:len(COLUMNAS_ORIGINALES)] == COLUMNAS_ORIGINALES and "celular" in columnas
    assert [(f["escena"], f["fecha"], f["celular"]) for f in filas] == [
        ("1", "2026-10-18", "Xiaomi Redmi Note 12"), ("2", "2026-10-18", "Xiaomi Redmi Note 12")]
    _, fotos = leer_csv(destino / "fotos.csv")
    assert [f["angulo"] for f in fotos[:3]] == ["frontal", "izq20", "der20"]


def test_ingesta_sin_copiar_plan_reporta_faltantes(entorno):
    tmp, destino = entorno
    crear_fotos(tmp / "cel", 3)
    inf = _ingerir(tmp / "cel", destino, copiar_plan=False)
    assert any("--copiar-plan" in p for p in inf.problemas)


def test_ingesta_sobrantes_y_escena_fuera_del_plan(entorno):
    tmp, destino = entorno
    crear_fotos(tmp / "cel", 23)          # 7 escenas completas (el plan tiene 6) + 2 sobrantes
    inf = _ingerir(tmp / "cel", destino, copiar_plan=True)
    assert any("2 foto(s) sobrantes" in p for p in inf.problemas)
    assert any(p.startswith("escena 7:") and "ni en el plan" in p for p in inf.problemas)
    assert not (destino / "sesion_01" / "sesion01_escena08_foto1.jpg").exists()


def test_ingesta_no_pisa_y_es_idempotente(entorno):
    tmp, destino = entorno
    crear_fotos(tmp / "cel", 3)
    assert _ingerir(tmp / "cel", destino, copiar_plan=True).problemas == []
    inf = _ingerir(tmp / "cel", destino, copiar_plan=True)         # misma carpeta otra vez
    assert inf.problemas == [] and any("ya estaban copiadas" in a for a in inf.avisos)
    _, fotos = leer_csv(destino / "fotos.csv")
    assert len(fotos) == 3
    _, filas = leer_csv(destino / "escenas.csv")
    assert len(filas) == 1
    otra = tmp / "otra"
    crear_fotos(otra, 3, inicio=datetime(2026, 10, 19, 9, 0, 0))
    for p in otra.iterdir():                                        # contenido distinto
        Image.new("RGB", (64, 48), (1, 2, 3)).save(p)
    original = (destino / "sesion_01" / "sesion01_escena01_foto1.jpg").read_bytes()
    inf = _ingerir(otra, destino, copiar_plan=True)
    assert any("no se pisa" in p for p in inf.problemas)
    assert (destino / "sesion_01" / "sesion01_escena01_foto1.jpg").read_bytes() == original


def test_ingesta_detecta_conteos_imposibles(entorno):
    tmp, destino = entorno
    fila = {c: 1 for c in CLASES} | {"sesion": 1, "escena": 1, "luz": "natural", "fondo": "blanco",
                                     "coca_lata": -1, "fanta_lata": "dos", "pepsi_botella": 9}
    escribir_csv(destino / "escenas.csv", COLUMNAS_ORIGINALES, [fila])
    crear_fotos(tmp / "cel", 3)
    inf = _ingerir(tmp / "cel", destino, simular=True)
    texto = "\n".join(inf.problemas)
    assert "coca_lata=-1 es negativo" in texto
    assert "no es un entero" in texto
    assert "pepsi_botella=9 supera" in texto


def test_ingesta_escena_inicial_orden_nombre_y_avisos(entorno):
    tmp, destino = entorno
    crear_fotos(tmp / "cel", 3, paso=timedelta(minutes=10))
    (tmp / "cel" / "IMG_9999.HEIC").write_bytes(b"x")
    inf = _ingerir(tmp / "cel", destino, simular=True, copiar_plan=True, escena_inicial=4,
                   orden="nombre")
    assert any("sesion01_escena04_foto1.jpg" in a for a in inf.acciones)
    assert any("HEIC" in a and "JPG" in a for a in inf.avisos)
    assert any("abarcan" in a for a in inf.avisos)


def test_ingesta_cli_simular(entorno, capsys):
    tmp, destino = entorno
    crear_fotos(tmp / "cel", 6, con_exif=False)
    codigo = ing.main([str(tmp / "cel"), "--sesion", "1", "--destino", str(destino), "--simular",
                       "--copiar-plan"])
    salida = capsys.readouterr().out
    assert codigo == 0 and "SIMULACIÓN" in salida and "sin fecha EXIF" in salida
    assert not (destino / "sesion_01").exists()
