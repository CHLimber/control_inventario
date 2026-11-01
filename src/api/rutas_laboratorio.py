"""Laboratorio de modelos: probar detector y clasificador sobre una foto, sin tocar la BD.

    GET  /api/laboratorio/modelos     detectores (.pt) y clasificadores (.keras) encontrados
    POST /api/laboratorio/analizar    foto → cajas del detector + softmax del clasificador por caja
    POST /api/laboratorio/clasificar  foto de UN envase (un recorte) → softmax del clasificador
    POST /api/laboratorio/sintetico   estante sintético nuevo (src/sinteticos/al_vuelo.py) con su conteo exacto

A diferencia del pipeline real, acá se aceptan modelos a medio entrenar:
  * clasificadores con otras clases (p. ej. los de 9 clases): se usan con el clases.json que
    tienen al lado y la web avisa que no coinciden con config/clases.json;
  * detectores de prueba (runs/detector/*/weights/best.pt) y el YOLO de COCO sin entrenar
    (modelos/yolo26*.pt), filtrando sus clases bottle/cup.
El detector corre con un umbral bajo y devuelve TODAS las cajas con sus probabilidades: los
umbrales de detección y de clasificación se mueven en la web sin volver a llamar a la API.

Modelos recomendados (los que la web elige por defecto): la variante B, el YOLO multiclase, fue la
más fuerte. En estantes sintéticos nuevos contó exacto el 100 % de 45 imágenes, contra 27 % de la
variante A, cuyo clasificador nunca vio recortes sintéticos. En fotos reales no hay una comparación
justa todavía: cada variante se evaluó con un test distinto (ver docs/pendientes.md).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import DIR_MODELOS, RAIZ, nombres_clases
from src.vision.clasificador import leer_clases_modelo
from src.vision.clasificador_modelos import preprocesar_lote
from src.vision.detector import PESOS_1CLASE, PESOS_MULTICLASE, Detector
from src.vision.pipeline import recortar

# Fuera de /docs: es una herramienta de prueba, no parte del contrato (contratos/api_schemas.py).
router = APIRouter(prefix="/api/laboratorio", include_in_schema=False)

DIR_RUNS_DETECTOR = RAIZ / "runs" / "detector"
UMBRAL_DET_MINIMO = 0.05     # la web filtra encima de esto
LADO_MAXIMO = 2048           # fotos más grandes se achican antes de detectar
CLASES_COCO = {"bottle", "cup"}
CLASIFICADOR_PRINCIPAL = DIR_MODELOS / "clasificador" / "clasificador.keras"
VARIANTE_RECOMENDADA = "B"

_modelos_cargados: dict[tuple, object] = {}
_candado = threading.Lock()


# ------------------------------------------------------------------ catálogo de modelos
def _id(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.resolve().as_posix()


def _tipo_detector(ruta: Path) -> str:
    texto = _id(ruta).lower()
    if ruta.resolve().parent == DIR_MODELOS.resolve() and ruta.name.startswith("yolo"):
        return "coco"
    return "1clase" if "1clase" in texto else "multiclase"


_huellas: dict[tuple, str] = {}


def _huella(ruta: Path) -> str:
    """SHA-1 del archivo (en caché por ruta, tamaño y fecha): detecta copias del mismo modelo."""
    st = ruta.stat()
    clave = (str(ruta.resolve()), st.st_size, st.st_mtime)
    if clave not in _huellas:
        _huellas[clave] = hashlib.sha1(ruta.read_bytes()).hexdigest()
    return _huellas[clave]


def _fecha(ruta: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ruta.stat().st_mtime))


def listar_detectores() -> list[dict]:
    """De mejor a peor: el recomendado, los demás de modelos/detector (los que usa el sistema), otras
    corridas, el YOLO de COCO sin entrenar y las corridas de prueba. Las copias de un modelo de
    modelos/detector (p. ej. runs/detector/colab_*/weights/best.pt) no se repiten."""
    recomendado = _id(PESOS_MULTICLASE) if VARIANTE_RECOMENDADA == "B" else _id(PESOS_1CLASE)
    principales = sorted((DIR_MODELOS / "detector").glob("*.pt"), key=lambda r: (_id(r) != recomendado, r.name))
    vistos = {_huella(r) for r in principales}
    corridas = [r for r in sorted(DIR_RUNS_DETECTOR.glob("*/weights/best.pt")) if _huella(r) not in vistos]
    coco = sorted(DIR_MODELOS.glob("yolo*.pt"), reverse=True)           # yolo26s (más grande) antes que yolo26n
    filas = [(r, "principal") for r in principales]
    filas += [(r, "corrida") for r in corridas if "prueba" not in r.parent.parent.name]
    filas += [(r, "coco") for r in coco]
    filas += [(r, "prueba") for r in corridas if "prueba" in r.parent.parent.name]
    detectores = []
    for ruta, grupo in filas:
        id_ = _id(ruta)
        if id_ == recomendado:
            descripcion = f"variante {VARIANTE_RECOMENDADA} · el mejor"
        elif id_ == _id(PESOS_1CLASE):
            descripcion = "variante A · usar con el clasificador ★"
        elif id_ == _id(PESOS_MULTICLASE):
            descripcion = "variante B"
        else:
            descripcion = {"principal": "otro modelo", "corrida": "otra corrida",
                           "coco": "COCO sin entrenar · solo bottle y cup", "prueba": "corrida corta de prueba"}[grupo]
        detectores.append({"id": id_, "nombre": ruta.parent.parent.name if ruta.name == "best.pt" else ruta.stem,
                           "tipo": _tipo_detector(ruta), "prueba": grupo == "prueba",
                           "recomendado": id_ == recomendado, "descripcion": descripcion, "fecha": _fecha(ruta)})
    return detectores


def _metricas_clasificador(ruta: Path) -> dict:
    """<nombre>_metricas.json junto al modelo; el elegido (clasificador.keras) usa metricas.json."""
    candidatos = [ruta.with_name(f"{ruta.stem}_metricas.json")]
    if ruta.stem == "clasificador":
        candidatos.append(ruta.with_name("metricas.json"))
    for candidato in candidatos:
        if candidato.exists():
            try:
                return json.loads(candidato.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


def listar_clasificadores() -> list[dict]:
    """De mejor a peor: los de las 8 clases actuales por exactitud en test (el elegido primero) y después
    los de otras clases. Las copias exactas (clasificador.keras es una copia del modelo elegido de su
    corrida, p. ej. mobilenet.keras) aparecen una sola vez, con el nombre "clasificador"."""
    esperadas = nombres_clases()
    raiz = DIR_MODELOS / "clasificador"
    principal = CLASIFICADOR_PRINCIPAL.resolve() if CLASIFICADOR_PRINCIPAL.exists() else None
    rutas = sorted(raiz.rglob("*.keras"),
                   key=lambda r: (r.parent != raiz, r.parent.as_posix(), r.stem != "clasificador", r.stem))
    vistos: set[str] = set()
    clasificadores = []
    for ruta in rutas:
        try:
            clases = leer_clases_modelo(ruta)
        except Exception:  # sin clases.json al lado: no se puede interpretar su salida
            continue
        huella = _huella(ruta)
        if huella in vistos:
            continue
        vistos.add(huella)
        es_principal = ruta.resolve() == principal
        carpeta = ruta.parent.relative_to(DIR_MODELOS / "clasificador").as_posix()
        met = _metricas_clasificador(ruta)
        acc = (met.get("test") or {}).get("accuracy")
        partes = []
        if ruta.stem == "clasificador":
            partes.append(f"el elegido ({met['modelo']})" if met.get("modelo") else "el elegido")
        if acc is not None:
            partes.append(f"test {acc * 100:.1f} %".replace(".", ","))
        if clases != esperadas:
            partes.append(f"{len(clases)} clases · solo para comparar")
        clasificadores.append({
            "id": _id(ruta), "nombre": ruta.stem if carpeta == "." else f"{carpeta}/{ruta.stem}",
            "clases": clases, "coincide_config": clases == esperadas, "principal": es_principal,
            "exactitud_test": acc, "descripcion": " · ".join(partes), "fecha": _fecha(ruta)})
    clasificadores.sort(key=lambda c: (not c["coincide_config"], not c["principal"],
                                       -(c["exactitud_test"] or 0), c["nombre"]))
    return clasificadores


def _buscar(lista: list[dict], id_modelo: str, que: str) -> dict:
    for modelo in lista:
        if modelo["id"] == id_modelo:
            return modelo
    raise HTTPException(404, f"No se encontró el {que} {id_modelo!r} (recargá la lista de modelos).")


# ------------------------------------------------------------------ carga (en caché)
def _detector(id_modelo: str) -> Detector:
    ruta = RAIZ / id_modelo
    clave = ("det", str(ruta), ruta.stat().st_mtime)
    with _candado:
        if clave not in _modelos_cargados:
            _modelos_cargados[clave] = Detector(ruta, conf=UMBRAL_DET_MINIMO)
        return _modelos_cargados[clave]


def _clasificador(id_modelo: str):
    """(modelo keras, lado de entrada). Sin verificar las clases: el laboratorio acepta cualquiera."""
    ruta = RAIZ / id_modelo
    clave = ("cls", str(ruta), ruta.stat().st_mtime)
    with _candado:
        if clave not in _modelos_cargados:
            import keras

            modelo = keras.models.load_model(ruta, compile=False)
            _modelos_cargados[clave] = (modelo, int(modelo.inputs[0].shape[1]))
        return _modelos_cargados[clave]


def _probabilidades(id_modelo: str, recortes: list[np.ndarray]) -> np.ndarray:
    modelo, tamano = _clasificador(id_modelo)
    if not recortes:
        return np.zeros((0, int(modelo.outputs[0].shape[-1])), dtype=np.float32)
    x = preprocesar_lote(recortes, tamano).astype(np.float32)
    return np.concatenate([np.asarray(modelo(x[i:i + 64], training=False))
                           for i in range(0, len(x), 64)], axis=0)


# ------------------------------------------------------------------ utilidades
async def _leer_imagen(archivo: UploadFile) -> np.ndarray:
    datos = await archivo.read()
    if not datos:
        raise HTTPException(422, "La imagen está vacía.")
    try:
        with Image.open(io.BytesIO(datos)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")  # fotos del celular: respetar la rotación
            im.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
            return np.asarray(im)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, f"No se pudo leer {archivo.filename!r} como imagen.") from None


def _redondear(fila: np.ndarray) -> list[float]:
    return [round(float(p), 4) for p in fila]


# ------------------------------------------------------------------ rutas
def _variantes() -> dict:
    """Atajos de la web: qué detector y clasificador arman cada variante del sistema."""
    return {"A": {"detector": _id(PESOS_1CLASE), "clasificador": _id(CLASIFICADOR_PRINCIPAL),
                  "disponible": PESOS_1CLASE.exists() and CLASIFICADOR_PRINCIPAL.exists()},
            "B": {"detector": _id(PESOS_MULTICLASE), "clasificador": "",
                  "disponible": PESOS_MULTICLASE.exists()}}


@router.get("/modelos")
def modelos() -> dict:
    from src.sinteticos import al_vuelo

    return {"detectores": listar_detectores(), "clasificadores": listar_clasificadores(),
            "clases_config": nombres_clases(), "variantes": _variantes(),
            "variante_recomendada": VARIANTE_RECOMENDADA, "sinteticos": al_vuelo.DIR_OBJETOS.is_dir()}


@router.post("/analizar")
async def analizar(imagen: UploadFile = File(...), detector: str = Form(...),
                   clasificador: str = Form("")) -> dict:
    info_det = _buscar(listar_detectores(), detector, "detector")
    info_cls = _buscar(listar_clasificadores(), clasificador, "clasificador") if clasificador else None
    arr = await _leer_imagen(imagen)

    inicio = time.perf_counter()
    try:
        cajas = _detector(info_det["id"]).detectar(arr)
    except Exception as error:
        raise HTTPException(500, f"Falló el detector: {error}") from error
    if info_det["tipo"] == "coco":
        cajas = [c for c in cajas if c.clase in CLASES_COCO]
    t_det = (time.perf_counter() - inicio) * 1000

    probs = None
    t_cls = 0.0
    if info_cls is not None:
        inicio = time.perf_counter()
        try:
            probs = _probabilidades(info_cls["id"], [recortar(arr, c.x, c.y, c.w, c.h) for c in cajas])
        except Exception as error:
            raise HTTPException(500, f"Falló el clasificador: {error}") from error
        t_cls = (time.perf_counter() - inicio) * 1000

    return {
        "ancho": int(arr.shape[1]), "alto": int(arr.shape[0]),
        "detector": info_det, "clasificador": info_cls,
        "cajas": [{"x": round(c.x, 1), "y": round(c.y, 1), "w": round(c.w, 1), "h": round(c.h, 1),
                   "conf_det": round(c.conf_det, 4), "clase_det": c.clase,
                   "probs": _redondear(probs[i]) if probs is not None else None}
                  for i, c in enumerate(cajas)],
        "tiempos": {"deteccion_ms": round(t_det, 1), "clasificacion_ms": round(t_cls, 1)},
    }


@router.post("/clasificar")
async def clasificar(imagen: UploadFile = File(...), clasificador: str = Form(...)) -> dict:
    info_cls = _buscar(listar_clasificadores(), clasificador, "clasificador")
    arr = await _leer_imagen(imagen)
    inicio = time.perf_counter()
    try:
        probs = _probabilidades(info_cls["id"], [arr])[0]
    except Exception as error:
        raise HTTPException(500, f"Falló el clasificador: {error}") from error
    return {"clasificador": info_cls, "probs": _redondear(probs),
            "tiempo_ms": round((time.perf_counter() - inicio) * 1000, 1)}


@router.post("/sintetico")
def sintetico(semilla: int = Form(-1), niveles: int = Form(0), orden: str = Form("")) -> dict:
    """Estante sintético nuevo + su conteo exacto. semilla < 0 = al azar; niveles 0 y orden '' = al azar."""
    from src.sinteticos import al_vuelo

    if orden and orden not in al_vuelo.ORDENES:
        raise HTTPException(422, f"orden debe ser uno de {', '.join(al_vuelo.ORDENES)}")
    semilla = semilla if semilla >= 0 else random.randrange(1_000_000)
    inicio = time.perf_counter()
    try:
        esc = al_vuelo.generar_estante(semilla, niveles, orden or None)
    except al_vuelo.SinBiblioteca as error:
        raise HTTPException(503, str(error)) from None
    buf = io.BytesIO()
    Image.fromarray(esc.imagen).save(buf, format="JPEG", quality=92)
    clases = nombres_clases()
    return {
        "semilla": semilla, "ancho": int(esc.imagen.shape[1]), "alto": int(esc.imagen.shape[0]),
        "imagen": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
        "cajas": [{"x": round(x1, 1), "y": round(y1, 1), "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                   "clase": clases[int(cid)]} for cid, x1, y1, x2, y2 in esc.cajas],
        "conteo": al_vuelo.conteo_escena(esc),
        "meta": {k: esc.meta[k] for k in ("n_niveles", "orden", "luz", "vista") if k in esc.meta},
        "tiempo_ms": round((time.perf_counter() - inicio) * 1000, 1),
    }
