"""Laboratorio de modelos: probar detector y clasificador sobre una foto, sin tocar la BD.

    GET  /api/laboratorio/modelos     detectores (.pt) y clasificadores (.keras) encontrados
    POST /api/laboratorio/analizar    foto → cajas del detector + softmax del clasificador por caja
    POST /api/laboratorio/clasificar  foto de UN envase (un recorte) → softmax del clasificador

A diferencia del pipeline real, acá se aceptan modelos a medio entrenar:
  * clasificadores con otras clases (p. ej. los de 9 clases): se usan con el clases.json que
    tienen al lado y la web avisa que no coinciden con config/clases.json;
  * detectores de prueba (runs/detector/*/weights/best.pt) y el YOLO de COCO sin entrenar
    (modelos/yolo26*.pt), filtrando sus clases bottle/cup.
El detector corre con un umbral bajo y devuelve TODAS las cajas con sus probabilidades: los
umbrales de detección y de clasificación se mueven en la web sin volver a llamar a la API.
"""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import DIR_MODELOS, RAIZ, nombres_clases
from src.vision.clasificador import leer_clases_modelo
from src.vision.clasificador_modelos import preprocesar_lote
from src.vision.detector import Detector

# Fuera de /docs: es una herramienta de prueba, no parte del contrato (contratos/api_schemas.py).
router = APIRouter(prefix="/api/laboratorio", include_in_schema=False)

DIR_RUNS_DETECTOR = RAIZ / "runs" / "detector"
UMBRAL_DET_MINIMO = 0.05     # la web filtra encima de esto
MARGEN_RECORTE = 0.05        # el mismo margen con el que se generaron los recortes (src/datos/recortes.py)
LADO_MAXIMO = 2048           # fotos más grandes se achican antes de detectar
CLASES_COCO = {"bottle", "cup"}

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


def listar_detectores() -> list[dict]:
    rutas = sorted((DIR_MODELOS / "detector").glob("*.pt"))
    rutas += sorted(DIR_RUNS_DETECTOR.glob("*/weights/best.pt"))
    rutas += sorted(DIR_MODELOS.glob("yolo*.pt"))
    detectores = []
    for ruta in rutas:
        tipo = _tipo_detector(ruta)
        nombre = ruta.parent.parent.name if ruta.name == "best.pt" else ruta.stem
        detectores.append({"id": _id(ruta), "nombre": nombre, "tipo": tipo,
                           "fecha": time.strftime("%Y-%m-%d %H:%M", time.localtime(ruta.stat().st_mtime))})
    return detectores


def listar_clasificadores() -> list[dict]:
    esperadas = nombres_clases()
    clasificadores = []
    for ruta in sorted((DIR_MODELOS / "clasificador").rglob("*.keras")):
        try:
            clases = leer_clases_modelo(ruta)
        except Exception:  # sin clases.json al lado: no se puede interpretar su salida
            continue
        carpeta = ruta.parent.relative_to(DIR_MODELOS / "clasificador").as_posix()
        clasificadores.append({
            "id": _id(ruta), "nombre": ruta.stem if carpeta == "." else f"{carpeta}/{ruta.stem}",
            "clases": clases, "coincide_config": clases == esperadas,
            "fecha": time.strftime("%Y-%m-%d %H:%M", time.localtime(ruta.stat().st_mtime))})
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


def _recortar(imagen: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    mx, my = w * MARGEN_RECORTE, h * MARGEN_RECORTE
    x0, y0 = max(0, int(round(x - mx))), max(0, int(round(y - my)))
    x1, y1 = min(ancho, int(round(x + w + mx))), min(alto, int(round(y + h + my)))
    return imagen[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]


def _redondear(fila: np.ndarray) -> list[float]:
    return [round(float(p), 4) for p in fila]


# ------------------------------------------------------------------ rutas
@router.get("/modelos")
def modelos() -> dict:
    return {"detectores": listar_detectores(), "clasificadores": listar_clasificadores(),
            "clases_config": nombres_clases()}


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
            probs = _probabilidades(info_cls["id"], [_recortar(arr, c.x, c.y, c.w, c.h) for c in cajas])
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
