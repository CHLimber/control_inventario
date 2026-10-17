"""Tests del pipeline de datos sobre un mini dataset YOLO sintético (tmp_path).

Cubren: ids originales, remapeo, fusión (detector y multiclase), recortes con sus filtros y
splits sin fugas. No usan la red ni los datos reales.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.datos import fusion, recortes
from src.datos.indice import indexar_dataset
from src.datos.remapeo import (
    DESCARTAR, SOLO_DETECTOR, cargar_remapeo, destino, es_generica, escribir_remapeo, sugerir, validar_remapeo,
)
from src.datos.split import asignar_grupos, asignar_splits, seleccionar, verificar_fugas
from src.datos.util import id_original, leer_etiquetas_yolo

CLASES = ["coca_botella", "coca_lata", "fanta_botella", "fanta_lata", "sprite_botella", "sprite_lata",
          "pepsi_botella", "pepsi_lata"]
NOMBRES_A = ["coke_can", "Cans", "beer_can", "person"]
NOMBRES_B = ["fanta bottle", "sprite_can"]
REMAPEO = {
    "ds_a": {"coke_can": "coca_lata", "Cans": SOLO_DETECTOR, "beer_can": SOLO_DETECTOR, "person": DESCARTAR},
    "ds_b": {"fanta bottle": "fanta_botella", "sprite_can": "sprite_lata"},
}


# ---------------------------------------------------------------- utilidades del mini dataset
def _imagen(semilla: int, lado: int = 256) -> Image.Image:
    """Imagen con textura única (pHash distinto por semilla)."""
    rng = np.random.default_rng(semilla)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((lado, lado), Image.NEAREST)
    ruido = rng.integers(0, 60, (lado, lado, 3), dtype=np.uint8)
    return Image.fromarray(np.clip(np.asarray(img).astype(int) + ruido, 0, 255).astype(np.uint8))


def _guardar(dir_ds: Path, split: str, nombre: str, img: Image.Image, lineas: list[str]) -> None:
    (dir_ds / split / "images").mkdir(parents=True, exist_ok=True)
    (dir_ds / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(dir_ds / split / "images" / nombre)
    (dir_ds / split / "labels" / (Path(nombre).stem + ".txt")).write_text("\n".join(lineas) + "\n")


def _yaml(dir_ds: Path, nombres: list[str]) -> None:
    dir_ds.mkdir(parents=True, exist_ok=True)
    (dir_ds / "data.yaml").write_text("nc: %d\nnames: %r\n" % (len(nombres), nombres), encoding="utf-8")


@pytest.fixture()
def mini(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    a, b = raw / "ds_a", raw / "ds_b"
    _yaml(a, NOMBRES_A)
    _yaml(b, NOMBRES_B)
    # ds_a: 20 fotos; la foto 0 tiene 2 copias aumentadas (.rf) en train.
    for i in range(20):
        img = _imagen(i)
        lineas = ["0 0.3 0.5 0.3 0.6",          # coke_can grande
                  "2 0.75 0.5 0.3 0.6",         # beer_can -> solo_detector
                  "3 0.5 0.1 0.05 0.05"]        # person -> descartar
        if i == 1:
            lineas.append("1 0.5 0.9 0.2 0.15")  # 'Cans' genérico -> fuera del multiclase
        if i == 0:
            _guardar(a, "train", "foto0_jpg.rf.aaa111.jpg", img, lineas)
            _guardar(a, "train", "foto0_jpg.rf.bbb222.jpg", img.transpose(Image.FLIP_LEFT_RIGHT), lineas)
            _guardar(a, "train", "foto0_jpg.rf.ccc333.jpg", img, lineas)  # copia idéntica
        else:
            _guardar(a, "train", f"foto{i}_jpg.rf.c{i:03d}.jpg", img, lineas)
    # ds_b: 10 fotos; la foto 0 es un DUPLICADO exacto de ds_a/foto5 (mismo pHash, otro dataset).
    for i in range(10):
        img = _imagen(5) if i == 0 else _imagen(100 + i)
        lineas = ["0 0.5 0.5 0.4 0.8", "1 0.2 0.2 0.05 0.05"]  # fanta grande + sprite chica (12 px)
        _guardar(b, "valid", f"b{i}.jpg", img, lineas)
    # ds_b: una foto con la caja de fanta UNIFORME (borrosa).
    lisa = _imagen(999)
    arr = np.asarray(lisa).copy()
    arr[8:248, 40:216] = 128  # más grande que la caja + margen
    _guardar(b, "valid", "lisa.jpg", Image.fromarray(arr), ["0 0.5 0.5 0.5 0.75"])
    indice = pd.concat([
        indexar_dataset(raw, "ds_a", "ws/proy_a", procesos=1),
        indexar_dataset(raw, "ds_b", "ws/proy_b", procesos=1),
    ], ignore_index=True)
    return {"raw": raw, "indice": indice, "tmp": tmp_path,
            "nombres": {"ds_a": NOMBRES_A, "ds_b": NOMBRES_B}}


# ---------------------------------------------------------------- ids y remapeo
def test_id_original_quita_sufijo_roboflow():
    assert id_original("IMG_001_jpg.rf.3f2a9c.jpg") == "IMG_001"
    assert id_original("foto.png") == "foto"
    assert id_original("a_jpg.rf.abc.jpg") == id_original("a_jpg.rf.def.jpg")


def test_sugerir_remapeo():
    assert sugerir("coke_can", CLASES) == "coca_lata"
    assert sugerir("Coca_Original_Bottle", CLASES) == "coca_botella"
    assert sugerir("Sprite20Oz", CLASES) == "sprite_botella"
    assert sugerir("Fanta-Can16oz", CLASES) == "fanta_lata"
    assert sugerir("7up_can", CLASES) == SOLO_DETECTOR     # 7up no es una de nuestras clases
    assert sugerir("pepsi_bottle", CLASES) == "pepsi_botella"
    assert sugerir("coca-cola", CLASES, "botella") == "coca_botella"
    assert sugerir("coca-cola", CLASES) == SOLO_DETECTOR          # sin formato
    assert sugerir("DietCokeCan16Oz", CLASES) == SOLO_DETECTOR    # variante
    assert sugerir("7up_bottle", CLASES) == SOLO_DETECTOR         # clase inexistente
    assert sugerir("beer_can", CLASES) == SOLO_DETECTOR
    assert sugerir("fantacoca-cola", CLASES) == SOLO_DETECTOR     # marca ambigua
    assert sugerir("DoritosNachoCheese", CLASES) == DESCARTAR
    assert sugerir("Cup", CLASES) == DESCARTAR
    assert es_generica("Cans") and es_generica("Bottle-Detection") and not es_generica("coke_can")


def test_remapeo_ida_y_vuelta_y_validacion(tmp_path):
    ruta = tmp_path / "remapeo.yaml"
    escribir_remapeo(ruta, REMAPEO, "# borrador")
    assert cargar_remapeo(ruta) == REMAPEO
    assert validar_remapeo(REMAPEO, CLASES) == []
    assert validar_remapeo({"x": {"a": "coca_cola"}}, CLASES)
    assert destino(REMAPEO, "ds_a", "no_existe") == DESCARTAR


def test_cajas_remapeadas(tmp_path):
    txt = tmp_path / "e.txt"
    txt.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n")
    det = fusion.cajas_remapeadas(txt, NOMBRES_A, REMAPEO, "ds_a", "detector_1clase", CLASES)
    multi = fusion.cajas_remapeadas(txt, NOMBRES_A, REMAPEO, "ds_a", "multiclase", CLASES)
    assert [c[0] for c in det] == [0, 0, 0]                      # person descartada
    assert [c[0] for c in multi] == [CLASES.index("coca_lata")]  # solo nuestras clases


# ---------------------------------------------------------------- grupos y splits
def test_grupos_unen_copias_y_duplicados(mini):
    df = asignar_grupos(mini["indice"])
    copias = df[df["id_original"] == "foto0"]
    assert len(copias) == 3 and copias["grupo"].nunique() == 1
    dup_a = df.loc[(df["alias"] == "ds_a") & (df["id_original"] == "foto5"), "grupo"].iloc[0]
    dup_b = df.loc[(df["alias"] == "ds_b") & (df["id_original"] == "b0"), "grupo"].iloc[0]
    assert dup_a == dup_b  # mismo pHash entre datasets -> mismo grupo


def test_id_repetido_se_subdivide_por_phash():
    # 30 imágenes llamadas "01": 3 fotos distintas x 10 copias casi iguales (1 bit de diferencia).
    bases = ["0f0f0f0f0f0f0f0f", "f0f0f0f0f0f0f0f0", "3c3c3c3c3c3c3c3c"]
    filas = []
    for b in bases:
        for k in range(10):
            ph = format(int(b, 16) ^ (1 << k), "016x")
            filas.append({"proyecto": "ws/p", "id_original": "01", "phash": ph, "alias": "x"})
    df = asignar_grupos(pd.DataFrame(filas), max_por_id=20, umbral_hamming=4)
    assert df["grupo"].nunique() == 3
    assert asignar_grupos(pd.DataFrame(filas), max_por_id=50)["grupo"].nunique() == 1


def test_split_sin_fugas_y_test_externo(mini):
    df = asignar_splits(asignar_grupos(mini["indice"]), {"ds_a": "entrenamiento", "ds_b": "test_externo"})
    assert verificar_fugas(df) == []
    assert set(df.loc[df["alias"] == "ds_b", "split"]) == {"test"}
    # el duplicado de ds_b arrastra a test a su gemelo de ds_a
    assert df.loc[(df["alias"] == "ds_a") & (df["id_original"] == "foto5"), "split"].iloc[0] == "test"
    # determinista
    df2 = asignar_splits(asignar_grupos(mini["indice"]), {"ds_a": "entrenamiento", "ds_b": "test_externo"})
    assert (df["split"].values == df2["split"].values).all()


def test_splits_estables_al_agregar_datos(mini):
    usos = {"ds_a": "entrenamiento", "ds_b": "entrenamiento"}
    solo_a = asignar_splits(asignar_grupos(mini["indice"][mini["indice"]["alias"] == "ds_a"]), usos, semilla=1)
    previos = dict(zip(solo_a["ruta"], solo_a["split"]))
    # Con otra semilla, sin 'previos' el sorteo cambiaría; con 'previos' lo ya asignado se mantiene.
    todo = asignar_splits(asignar_grupos(mini["indice"]), usos, semilla=7, previos=previos)
    a = todo[todo["alias"] == "ds_a"]
    assert (a["ruta"].map(previos) == a["split"]).all()
    assert verificar_fugas(todo) == []


def test_verificar_fugas_detecta_grupo_partido():
    df = pd.DataFrame({"grupo": ["g1", "g1", "g2"], "split": ["train", "test", "val"],
                       "phash": ["a1a1a1a1a1a1a1a1", "b2b2b2b2b2b2b2b2", "c3c3c3c3c3c3c3c3"]})
    assert verificar_fugas(df)


def test_seleccionar_una_copia_por_foto(mini):
    df = asignar_grupos(mini["indice"]).assign(split="train")
    sel = seleccionar(df, copias_train=1)
    assert sel.groupby(["alias", "grupo"]).size().max() == 1
    assert len(seleccionar(df, copias_train=0)) == len(df)


# ---------------------------------------------------------------- control de color
def test_control_de_color():
    from src.datos.color import medir, motivo_color

    img = Image.new("RGB", (200, 100), (255, 255, 255))
    img.paste((220, 20, 30), (0, 0, 100, 100))     # mitad izquierda roja (lata de Coca)
    img.paste((30, 160, 40), (100, 0, 200, 100))   # mitad derecha verde (lata de Sprite)
    rojo, verde = medir(img, (0, 0, 100, 100)), medir(img, (100, 0, 200, 100))
    assert rojo["rojo"] > 0.9 and verde["verde"] > 0.9
    assert motivo_color("coca_lata", rojo) == ""
    assert motivo_color("coca_lata", verde).startswith("color_rojo")   # verde rotulada Coca
    assert motivo_color("sprite_lata", verde) == ""
    gris = medir(img.convert("L").convert("RGB"), (0, 0, 100, 100))
    assert motivo_color("coca_lata", gris).startswith("color_gris")


def test_fusion_y_recortes_excluyen_color_incoherente(mini):
    df = asignar_splits(asignar_grupos(mini["indice"]), {"ds_a": "entrenamiento", "ds_b": "entrenamiento"})
    ruta = df.loc[df["alias"] == "ds_b", "ruta"].iloc[1]
    rechazos = {(ruta, 0): "color_naranja_0.00"}
    man = fusion.construir(df, mini["raw"], mini["tmp"] / "m", "multiclase", REMAPEO, mini["nombres"], CLASES,
                           {"ds_a", "ds_b"}, copias_train=0, max_negativos=0.0, rechazos_color=rechazos)
    assert ruta not in set(man["ruta"])
    assert (mini["tmp"] / "m" / "_descartes_color.csv").exists()
    ind = recortes.generar(df, mini["raw"], mini["tmp"] / "r", REMAPEO, mini["nombres"], CLASES,
                           rechazos_color=rechazos)
    fila = ind[(ind["imagen"] == ruta) & (ind["caja"] == 0)]
    assert fila["motivo_descarte"].iloc[0] == "color_naranja_0.00"


# ---------------------------------------------------------------- fusión
def test_fusion_detector_y_multiclase(mini):
    df = asignar_splits(asignar_grupos(mini["indice"]), {"ds_a": "entrenamiento", "ds_b": "entrenamiento"},
                        proporciones=(0.6, 0.2, 0.2))
    for modo in fusion.MODOS:
        salida = mini["tmp"] / "fusion" / modo
        man = fusion.construir(df, mini["raw"], salida, modo, REMAPEO, mini["nombres"], CLASES,
                               {"ds_a", "ds_b"}, copias_train=1, max_negativos=0.0)
        assert (salida / "data.yaml").exists()
        assert verificar_fugas(man) == []
        assert man.groupby(["alias", "grupo"]).size().max() == 1  # la foto aumentada aparece una vez
        etiquetas = list((salida / "labels").rglob("*.txt"))
        imagenes = list((salida / "images").rglob("*.jpg"))
        assert len(etiquetas) == len(imagenes) == len(man)
        ids = {c[0] for t in etiquetas for c in leer_etiquetas_yolo(t)}
        if modo == "detector_1clase":
            assert ids == {0}
        else:
            assert ids <= {CLASES.index("coca_lata"), CLASES.index("fanta_botella"), CLASES.index("sprite_lata")}
            assert not any("foto1_" in i.name for i in imagenes)  # tenía una caja genérica 'Cans'
        texto = (salida / "data.yaml").read_text(encoding="utf-8")
        assert ("producto" in texto) == (modo == "detector_1clase")


# ---------------------------------------------------------------- recortes
def test_recortes_filtros_y_sin_fugas(mini):
    df = asignar_splits(asignar_grupos(mini["indice"]), {"ds_a": "entrenamiento", "ds_b": "entrenamiento"},
                        proporciones=(0.6, 0.2, 0.2))
    df.loc[df["id_original"] == "foto0", "split"] = "train"  # el grupo entero, sigue sin fugas
    sel = seleccionar(df, copias_train=0)  # incluye las copias de foto0 -> la idéntica se descarta
    salida = mini["tmp"] / "recortes"
    ind = recortes.generar(sel, mini["raw"], salida, REMAPEO, mini["nombres"], CLASES,
                           margen=0.05, min_lado=32, min_nitidez=40)
    assert (salida / "indice_recortes.csv").exists()
    columnas = {"origen", "imagen", "imagen_original", "clase", "split", "motivo_descarte", "ruta_recorte"}
    assert columnas <= set(ind.columns)
    motivos = set(ind["motivo_descarte"])
    assert "chica" in motivos         # sprite de 12 px
    assert "borrosa" in motivos       # caja uniforme
    assert "duplicada" in motivos     # copia idéntica de foto0
    ok = ind[ind["motivo_descarte"] == ""]
    assert set(ok["clase"]) <= {"coca_lata", "fanta_botella"}
    for r in ok.itertuples():
        assert (salida / r.ruta_recorte).exists()
        assert r.ruta_recorte.startswith(f"{r.split}/{r.clase}/")
    assert verificar_fugas(ok.rename(columns={"imagen_original": "grupo"})) == []
    # el recorte mantiene margen pequeño: un poco más grande que la caja
    r = ok.iloc[0]
    with Image.open(salida / r.ruta_recorte) as im:
        assert im.width >= r.ancho and im.width <= r.ancho * 1.2 + 2


# ------------------------------------------------------------------ destino auto:<marca> (formato por la forma de la caja)
def test_resolver_auto_por_forma_de_caja():
    from src.datos.remapeo import es_auto, resolver_auto
    assert es_auto("auto:pepsi") and not es_auto("pepsi_lata")
    assert resolver_auto("auto:pepsi", 40, 74, CLASES) == "pepsi_lata"          # 1,85
    assert resolver_auto("auto:pepsi", 40, 130, CLASES) == "pepsi_botella"      # 3,25
    assert resolver_auto("auto:pepsi", 40, 100, CLASES) == SOLO_DETECTOR        # 2,5: forma dudosa
    assert resolver_auto("auto:pepsi", 40, 40, CLASES) == SOLO_DETECTOR         # cuadrada
    assert resolver_auto("auto:pepsi", 40, 300, CLASES) == SOLO_DETECTOR        # demasiado alta
    assert resolver_auto("auto:7up", 40, 74, CLASES) == SOLO_DETECTOR           # marca sin clase


def test_validar_remapeo_acepta_auto_solo_con_marca_existente():
    assert validar_remapeo({"x": {"a": "auto:coca", "b": "auto:pepsi"}}, CLASES) == []
    assert validar_remapeo({"x": {"a": "auto:marca_inexistente"}}, CLASES)


def _dataset_auto(tmp_path):
    img, lab = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    img.mkdir(parents=True)
    lab.mkdir(parents=True)
    Image.new("RGB", (400, 400), "white").save(img / "f.rf.abc.jpg")
    (lab / "f.rf.abc.txt").write_text("\n".join([
        "0 0.2 0.5 0.1 0.35",     # 40x140 px -> botella
        "0 0.5 0.5 0.12 0.22",    # 48x88 px  -> lata
        "0 0.8 0.5 0.10 0.25",    # 40x100 px -> forma dudosa
        "1 0.9 0.2 0.1 0.35",     # clase descartada
    ]))
    return lab / "f.rf.abc.txt"


def test_cajas_remapeadas_con_auto(tmp_path):
    from src.datos.fusion import cajas_remapeadas
    txt = _dataset_auto(tmp_path)
    remapeo = {"ds": {"Pepsi": "auto:pepsi", "Otra": DESCARTAR}}
    mc = cajas_remapeadas(txt, ["Pepsi", "Otra"], remapeo, "ds", "multiclase", CLASES)
    assert [c[0] for c in mc] == [CLASES.index("pepsi_botella"), CLASES.index("pepsi_lata")]   # la dudosa no entra
    uc = cajas_remapeadas(txt, ["Pepsi", "Otra"], remapeo, "ds", "detector_1clase", CLASES)
    assert len(uc) == 3 and all(c[0] == 0 for c in uc)                                          # al detector entran las 3


def test_recortes_con_tope_por_origen(tmp_path):
    """Con tope, de cada imagen salen a lo sumo `tope_por_imagen` recortes por clase."""
    import pandas as pd
    from src.datos.recortes import generar
    raw = tmp_path / "raw"
    img, lab = raw / "ds" / "train" / "images", raw / "ds" / "train" / "labels"
    img.mkdir(parents=True)
    lab.mkdir(parents=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)).save(img / "g.jpg")
    cajas = [f"0 {0.1 + 0.1 * k:.2f} 0.5 0.08 0.30" for k in range(8)]          # 8 botellas de pepsi (32x120 px)
    (lab / "g.txt").write_text("\n".join(cajas))
    sel = pd.DataFrame([{"alias": "ds", "ruta": "ds/train/images/g.jpg", "grupo": "g", "split": "train"}])
    kw = dict(remapeo={"ds": {"Pepsi": "auto:pepsi"}}, nombres_por_alias={"ds": ["Pepsi"]}, clases=CLASES,
              min_lado=8, min_nitidez=0.0)
    sin = generar(sel, raw, tmp_path / "s", **kw)
    con = generar(sel, raw, tmp_path / "c", topes={"ds": 100}, tope_por_imagen=2, **kw)
    assert (sin["motivo_descarte"] == "").sum() == 8
    assert (con["motivo_descarte"] == "").sum() == 2 and (con["motivo_descarte"] == "tope_por_origen").sum() == 6


def test_fuente_de_biblioteca_solo_train_con_clases_resueltas(tmp_path):
    """El árbol auxiliar lleva solo train; las cajas auto: reciben clase y las demás el id ID_OTRO."""
    from src.datos.biblioteca import ID_OTRO, construir_fuente_biblioteca
    raw = tmp_path / "raw"
    for split in ("train", "valid"):
        (raw / "ds" / split / "images").mkdir(parents=True)
        (raw / "ds" / split / "labels").mkdir(parents=True)
        Image.new("RGB", (400, 400), "white").save(raw / "ds" / split / "images" / f"{split}.jpg")
        (raw / "ds" / split / "labels" / f"{split}.txt").write_text(
            "0 0.2 0.5 0.1 0.35\n0 0.8 0.5 0.12 0.22\n1 0.5 0.5 0.1 0.3\n")
    df = pd.DataFrame([{"alias": "ds", "ruta": f"ds/{s}/images/{s}.jpg", "split": "train" if s == "train" else "val",
                        "n_cajas": 3} for s in ("train", "valid")])
    salida = tmp_path / "bib"
    conteo = construir_fuente_biblioteca(df, raw, salida, {"ds": {"Pepsi": "auto:pepsi", "Mountain Dew": "solo_detector"}},
                                         {"ds": ["Pepsi", "Mountain Dew"]}, CLASES, {"ds"})
    assert [p.name for p in (salida / "images" / "train").glob("*")] == ["ds__train.jpg"]
    assert not (salida / "images" / "val").exists()
    assert conteo["pepsi_botella"] == 1 and conteo["pepsi_lata"] == 1
    filas = [l.split() for l in (salida / "labels" / "train" / "ds__train.txt").read_text().splitlines()]
    assert sorted(int(f[0]) for f in filas) == sorted([CLASES.index("pepsi_botella"), CLASES.index("pepsi_lata"), ID_OTRO])


# ------------------------------------------------------------------ Open Food Facts: qué productos se aceptan
def test_off_acepta_solo_envases_individuales_clasicos():
    from scripts.datos_off import clasificar

    def p(nombre, marcas, cant="", emb=""):
        return {"product_name": nombre, "brands": marcas, "quantity": cant, "packaging": emb}

    assert clasificar(p("Fanta Naranja lata 330 ml", "Fanta, Coca-Cola", "330 ml"), "fanta") == ("fanta_lata", "lata")
    assert clasificar(p("Coca-Cola 2 L PET", "Coca-Cola", "2 l"), "coca-cola") == ("coca_botella", "botella")
    assert clasificar(p("Fanta Naranja 9 latas 330 ml", "Fanta"), "fanta") is None          # pack
    assert clasificar(p("Sprite Zero lata", "Sprite"), "sprite") is None                    # variante
    assert clasificar(p("Pepsi Max botella", "Pepsi"), "pepsi") is None                     # variante
    assert clasificar(p("Pepsi", "Pepsi", "330 ml"), "pepsi") is None                       # formato dudoso
    assert clasificar(p("Fanta lata botella", "Fanta"), "fanta") is None                    # ambos formatos
    assert clasificar(p("Sprite lata", "Otra marca"), "sprite") is None                     # la marca no coincide


def test_auto_con_un_solo_formato():
    from src.datos.remapeo import resolver_auto
    assert resolver_auto("auto:pepsi:botella", 40, 130, CLASES) == "pepsi_botella"
    assert resolver_auto("auto:pepsi:botella", 40, 74, CLASES) == SOLO_DETECTOR       # forma de lata: no se acepta
    assert resolver_auto("auto:pepsi:lata", 40, 74, CLASES) == "pepsi_lata"
    assert validar_remapeo({"x": {"a": "auto:coca:botella", "b": "auto:fanta:lata"}}, CLASES) == []
    assert validar_remapeo({"x": {"a": "auto:coca:envase"}}, CLASES)
