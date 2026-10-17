"""Tests del generador de estantes sintéticos con objetos y fondos FALSOS (no usan datos reales ni rembg)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import cargar_clases, nombres_clases
from src.sinteticos import objetos as obj
from src.sinteticos.composicion import Parametros, componer
from src.sinteticos.lote import generar_lote

CLASES = cargar_clases()["clases"]
NOMBRES = nombres_clases()


def _png_envase(ruta: Path, formato: str, color: tuple[int, int, int]) -> None:
    w, h = (40, 130) if formato == "botella" else (50, 92)
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = color
    a[..., 3] = 255
    if formato == "botella":   # cuello angosto con fondo transparente a los costados
        a[: h // 5, : w // 3, 3] = 0
        a[: h // 5, 2 * w // 3:, 3] = 0
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, "RGBA").save(ruta)


@pytest.fixture(scope="module")
def biblioteca(tmp_path_factory) -> dict[str, list[Path]]:
    raiz = tmp_path_factory.mktemp("objetos")
    rng = np.random.default_rng(0)
    bib = {}
    for c in CLASES:
        rutas = []
        for k in range(3):
            r = raiz / c["clase"] / f"{k}.png"
            _png_envase(r, c["formato"], tuple(int(v) for v in rng.integers(0, 255, 3)))
            rutas.append(r)
        bib[c["clase"]] = rutas
    return bib


@pytest.fixture(scope="module")
def fondos(tmp_path_factory) -> list[Path]:
    raiz = tmp_path_factory.mktemp("fondos")
    r = raiz / "Background_falso.jpg"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(r)
    return [r]


def _leer_lineas(p: Path) -> list[list[str]]:
    return [l.split() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def lote(tmp_path_factory, biblioteca, fondos) -> Path:
    sal = tmp_path_factory.mktemp("lote") / "lote"
    p = Parametros(lado_largo=512, lado_corto=384)
    generar_lote(sal, 24, 7, biblioteca, CLASES, fondos, p, procesos=1)
    return sal


def test_estructura_y_etiquetas_validas(lote: Path):
    imgs = sorted((lote / "multiclase" / "images").glob("*.jpg"))
    assert len(imgs) == 24
    assert len(list((lote / "1clase" / "images").glob("*.jpg"))) == 24
    for d in ("multiclase", "1clase"):
        assert (lote / d / "data.yaml").exists()
    for im in imgs:
        W, H = Image.open(im).size
        mc = _leer_lineas(lote / "multiclase" / "labels" / f"{im.stem}.txt")
        uc = _leer_lineas(lote / "1clase" / "labels" / f"{im.stem}.txt")
        assert len(mc) == len(uc)
        for fila, fila1 in zip(mc, uc):
            assert len(fila) == 5 and fila1[0] == "0" and fila[1:] == fila1[1:]
            c = int(fila[0])
            assert 0 <= c < len(NOMBRES)
            cx, cy, w, h = map(float, fila[1:])
            assert w > 0 and h > 0
            assert -1e-6 <= cx - w / 2 and cx + w / 2 <= 1 + 1e-6
            assert -1e-6 <= cy - h / 2 and cy + h / 2 <= 1 + 1e-6
            assert w * W >= 2 and h * H >= 2


def test_conteos_coinciden_con_etiquetas(lote: Path):
    with (lote / "conteos.csv").open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    assert list(filas[0].keys()) == ["imagen", *NOMBRES, "total"]
    assert len(filas) == 24
    for fila in filas:
        lineas = _leer_lineas(lote / "multiclase" / "labels" / (Path(fila["imagen"]).stem + ".txt"))
        esperado = {c: 0 for c in NOMBRES}
        for l in lineas:
            esperado[NOMBRES[int(l[0])]] += 1
        assert {c: int(fila[c]) for c in NOMBRES} == esperado
        assert int(fila["total"]) == len(lineas)


def test_hay_ceros_y_variedad(lote: Path):
    with (lote / "conteos.csv").open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    # en algún lado hay clases en cero, y no todo está vacío
    assert any(int(f[c]) == 0 for f in filas for c in NOMBRES)
    assert sum(int(f["total"]) for f in filas) > 24
    assert len({int(f["total"]) for f in filas}) > 3


def test_estante_vacio_da_etiquetas_vacias(tmp_path, biblioteca, fondos):
    p = Parametros(lado_largo=384, lado_corto=288, p_estante_vacio=1.0)
    res = generar_lote(tmp_path / "v", 4, 0, biblioteca, CLASES, fondos, p, procesos=1)
    assert res["imagenes_vacias"] == 4
    for t in (tmp_path / "v" / "multiclase" / "labels").glob("*.txt"):
        assert t.read_text(encoding="utf-8").strip() == ""


def test_reproducible_con_semilla(tmp_path, biblioteca, fondos):
    p = Parametros(lado_largo=384, lado_corto=288)
    generar_lote(tmp_path / "a", 5, 3, biblioteca, CLASES, fondos, p, procesos=1)
    generar_lote(tmp_path / "b", 5, 3, biblioteca, CLASES, fondos, p, procesos=1)
    generar_lote(tmp_path / "c", 5, 4, biblioteca, CLASES, fondos, p, procesos=1)
    for f in sorted((tmp_path / "a" / "multiclase" / "images").glob("*.jpg")):
        g = tmp_path / "b" / "multiclase" / "images" / f.name
        assert f.read_bytes() == g.read_bytes()
    assert (tmp_path / "a" / "conteos.csv").read_text() == (tmp_path / "b" / "conteos.csv").read_text()
    la = [p.read_text() for p in sorted((tmp_path / "a" / "multiclase" / "labels").glob("*.txt"))]
    lc = [p.read_text() for p in sorted((tmp_path / "c" / "multiclase" / "labels").glob("*.txt"))]
    assert la != lc


def test_componer_formato_por_nivel(biblioteca):
    """Latas arriba y botellas abajo: en una foto completa las latas quedan más arriba que las botellas."""
    p = Parametros(lado_largo=640, lado_corto=480, p_acercamiento=0.0, p_estante_vacio=0.0,
                   p_nivel_vacio=0.0, p_latas_medio=0.0, p_perspectiva_lateral=0.0)
    formato = {c["id"]: c["formato"] for c in CLASES}
    esc = componer(np.random.default_rng(5), biblioteca, CLASES, [], p)
    latas = [c for c in esc.cajas if formato[c[0]] == "lata"]
    botellas = [c for c in esc.cajas if formato[c[0]] == "botella"]
    assert latas and botellas
    assert max(c[4] for c in latas) < min(c[4] for c in botellas)   # base de las latas más arriba
    H, W = esc.imagen.shape[:2]
    for _, x1, y1, x2, y2 in esc.cajas:
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H


def test_clase_sin_objetos_no_aparece(tmp_path, biblioteca, fondos):
    bib = dict(biblioteca)
    bib["fanta_lata"] = []
    res = generar_lote(tmp_path / "s", 6, 1, bib, CLASES, fondos, Parametros(lado_largo=384, lado_corto=288), procesos=1)
    assert res["objetos_por_clase"]["fanta_lata"] == 0
    assert "fanta_lata" in res["clases_sin_objetos"]


# ------------------------------------------------------------------ biblioteca de objetos

def test_candidatos_filtran_cortados_acostados_y_tapados(tmp_path):
    img, lab = tmp_path / "images", tmp_path / "labels"
    img.mkdir(); lab.mkdir()
    Image.new("RGB", (400, 400), "white").save(img / "a__foto1_jpg.rf.x.jpg")
    filas = [
        "0 0.25 0.5 0.1 0.4",      # botella parada OK (40x160)
        "0 0.5 0.85 0.5 0.25",     # botella acostada (200x100)
        "1 0.04 0.5 0.1 0.2",      # lata cortada por el borde izquierdo
        "1 0.80 0.5 0.15 0.25",    # lata OK (60x100)
        "1 0.85 0.5 0.15 0.25",    # lata tapada por la anterior
    ]
    (lab / "a__foto1_jpg.rf.x.txt").write_text("\n".join(filas))
    c = obj.candidatos_yolo(img, lab, CLASES)
    motivos = [x.motivo for x in c]
    assert motivos[0] == "" and motivos[1] == "no_parada" and motivos[2] == "cortada_por_borde"
    assert motivos[3] == "tapada" and motivos[4] == "tapada"
    assert obj.grupo_de("a__foto1_jpg.rf.x.jpg") == "a__foto1"


def test_seleccionar_uno_por_foto_original():
    cands = [obj.Candidato(Path(f"x__F{k % 2}_jpg.rf.{k}.jpg"), "coca_lata", "lata", (0, 0, 1, 2), f"x__F{k % 2}") for k in range(6)]
    sel = obj.seleccionar(cands, max_por_clase=10)
    assert len(sel) == 2


def test_limpiar_mascara_une_tapa_y_cuerpo():
    m = np.zeros((100, 40), np.uint8)
    m[5:15, 15:25] = 255    # tapa
    m[30:95, 5:35] = 255    # cuerpo (cuello transparente entre medio)
    alfa = obj.limpiar_mascara(m, (0, 0, 40, 100))
    filas = np.where((alfa > 127).any(axis=1))[0]
    assert filas[0] == 5 and filas[-1] == 94
    assert (alfa[15:30] > 127).any(axis=1).all()   # el cuello quedó relleno
    assert obj.calidad_mascara(alfa, (0, 0, 40, 100)) == ""


# ------------------------------------------------------------------ niveles y modos de orden

@pytest.mark.parametrize("niveles", [3, 4, 5])
def test_cantidad_de_niveles_configurable(biblioteca, niveles):
    """3, 4 o 5 niveles: las latas siguen arriba de las botellas y las cajas caben en la foto."""
    p = Parametros(lado_largo=640, lado_corto=480, n_niveles=niveles, p_acercamiento=0.0, p_estante_vacio=0.0,
                   p_nivel_vacio=0.0, p_latas_medio=0.0, p_perspectiva_lateral=0.0, p_desordenado=0.0)
    formato = {c["id"]: c["formato"] for c in CLASES}
    esc = componer(np.random.default_rng(5), biblioteca, CLASES, [], p)
    assert esc.meta["n_niveles"] == niveles
    latas = [c for c in esc.cajas if formato[c[0]] == "lata"]
    botellas = [c for c in esc.cajas if formato[c[0]] == "botella"]
    assert latas and botellas
    assert max(c[4] for c in latas) < min(c[4] for c in botellas)
    H, W = esc.imagen.shape[:2]
    for _, x1, y1, x2, y2 in esc.cajas:
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H


def test_mas_niveles_mas_envases(biblioteca):
    """Un estante de 5 niveles lleva en promedio más envases que uno de 3."""
    def medio(n):
        p = Parametros(lado_largo=640, lado_corto=480, n_niveles=n, p_acercamiento=0.0, p_estante_vacio=0.0,
                       p_nivel_vacio=0.0)
        return np.mean([len(componer(np.random.default_rng(s), biblioteca, CLASES, [], p).cajas) for s in range(8)])
    assert medio(5) > medio(3)


def test_niveles_sorteados_incluyen_los_tres_tamanos(biblioteca):
    p = Parametros(lado_largo=384, lado_corto=288)
    vistos = {componer(np.random.default_rng(s), biblioteca, CLASES, [], p).meta["n_niveles"] for s in range(40)}
    assert vistos == {3, 4, 5}


@pytest.mark.parametrize("modo,kw", [("ordenado", {"p_ordenado": 1.0, "p_desordenado": 0.0}),
                                      ("desordenado", {"p_desordenado": 1.0, "p_caido": 0.5}),
                                      ("normal", {"p_ordenado": 0.0, "p_desordenado": 0.0})])
def test_modos_de_orden_dan_cajas_validas(biblioteca, modo, kw):
    p = Parametros(lado_largo=512, lado_corto=384, p_acercamiento=0.0, p_estante_vacio=0.0, p_nivel_vacio=0.0, **kw)
    total = 0
    for s in range(8):
        esc = componer(np.random.default_rng(s), biblioteca, CLASES, [], p)
        assert esc.meta["orden"] == modo
        H, W = esc.imagen.shape[:2]
        for cid, x1, y1, x2, y2 in esc.cajas:
            assert 0 <= cid < len(NOMBRES) and 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        total += len(esc.cajas)
    assert total > 0


def test_orden_por_bloques_contiguos():
    from src.sinteticos.composicion import _clases_nivel
    disp = ["a", "b", "c", "d"]
    for s in range(10):
        salida = _clases_nivel(np.random.default_rng(s), disp, 14, Parametros(), ordenado=True)
        assert len(salida) == 14 and salida == sorted(salida, key=disp.index)


def test_desordenado_tiene_envases_acostados(biblioteca):
    """Con p_caido alto aparecen cajas más anchas que altas (envases acostados)."""
    p = Parametros(lado_largo=640, lado_corto=480, p_acercamiento=0.0, p_estante_vacio=0.0, p_nivel_vacio=0.0,
                   p_desordenado=1.0, p_caido=0.8, p_perspectiva_lateral=0.0)
    anchas = sum(1 for s in range(6)
                 for _, x1, y1, x2, y2 in componer(np.random.default_rng(s), biblioteca, CLASES, [], p).cajas
                 if (x2 - x1) > (y2 - y1))
    assert anchas > 0


# ------------------------------------------------------------------ fotos de producto (Open Food Facts)

def test_caja_producto_sobre_fondo_blanco():
    img = Image.new("RGB", (300, 500), "white")
    img.paste(Image.new("RGB", (80, 250), (200, 20, 30)), (110, 120))
    x1, y1, x2, y2 = obj.caja_producto(img)
    assert abs(x1 - 110) <= 6 and abs(x2 - 190) <= 6 and abs(y1 - 120) <= 6 and abs(y2 - 370) <= 6
    assert obj.caja_producto(Image.new("RGB", (300, 500), "white")) is None


def test_candidatos_web_y_biblioteca_sin_filtro_de_train(tmp_path):
    web = tmp_path / "off"
    (web / "coca_lata").mkdir(parents=True)
    (web / "coca_botella").mkdir(parents=True)
    for clase, tam in (("coca_lata", (90, 170)), ("coca_botella", (60, 40))):
        im = Image.new("RGB", (400, 600), "white")
        im.paste(Image.new("RGB", tam, (200, 20, 30)), (150, 200))
        im.save(web / clase / "off__123.jpg")
    (web / "indice.csv").write_text("clase,formato,codigo,archivo\ncoca_lata,lata,123,coca_lata/off__123.jpg\n"
                                    "coca_botella,botella,123,coca_botella/off__123.jpg\n", encoding="utf-8")
    c = {x.clase: x for x in obj.candidatos_web(web, CLASES)}
    assert c["coca_lata"].motivo == "" and c["coca_lata"].grupo == "off__123"
    assert c["coca_botella"].motivo == "chica"          # 40 px de alto: no sirve para recortar
    # la biblioteca no exige que las fotos web estén en la carpeta de train
    cache = tmp_path / "cache"
    (cache / "coca_lata").mkdir(parents=True)
    Image.new("RGBA", (30, 60), (200, 20, 30, 255)).save(cache / "coca_lata" / "a.png")
    obj.escribir_indice(cache, [{"clase": "coca_lata", "formato": "lata", "origen": "off__123.jpg|0,0,1,1",
                                 "archivo": "coca_lata/a.png"}])
    train = tmp_path / "train"
    train.mkdir()
    assert obj.cargar_biblioteca(cache, [train])["coca_lata"]
