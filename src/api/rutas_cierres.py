"""Rutas del arqueo: escaneos, cierres, diferencias y asistente."""
from __future__ import annotations

import io
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from contratos.api_schemas import (MAX_IMAGENES_ESCANEO, CajaOut, CierreOut, CierreResumen,
                                   ConfirmarCierreIn, DiferenciaHistorial, EstadoCierre,
                                   FilaConciliacion, PreguntaIn, RespuestaAsistente, Totales)
from contratos.modelos_bd import Cierre, CierreDetalle, Producto
from src.api.dependencias import obtener_sesion, parsear_fecha
from src.conciliacion import (calcular_totales, completar_cierre, confirmar_cierre,
                              detalles_ordenados, estado_fila, iniciar_cierre, obtener_cierre)
from src.config import CLASE_DESCONOCIDO

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

PREFIJO_URL_ESCANEOS = "/escaneos"
_EXTENSIONES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif"}


# ---------------------------------------------------------------- serialización
def _carpeta_cierre(dir_escaneos: Path, id_cierre: int) -> Path:
    return Path(dir_escaneos) / f"cierre_{id_cierre}"


def _url(relativa: str) -> str:
    return f"{PREFIJO_URL_ESCANEOS}/{relativa.lstrip('/')}"


def resumen_cierre(cierre: Cierre) -> CierreResumen:
    return CierreResumen(id=cierre.id, fecha=cierre.fecha, usuario=cierre.usuario,
                         estado=cierre.estado, variante_modelo=cierre.variante_modelo,
                         totales=Totales(**calcular_totales(cierre.detalles)))


def cierre_a_salida(sesion: Session, cierre: Cierre, dir_escaneos: Path,
                    tiempo_ms: float = 0.0) -> CierreOut:
    clases = {id_: clase for id_, clase in sesion.execute(select(Producto.id, Producto.clase_modelo))}
    filas = [FilaConciliacion(
        id_producto=d.id_producto, clase_modelo=d.producto.clase_modelo, nombre=d.producto.nombre,
        esperado=d.esperado, detectado=d.detectado, ajustado=d.ajustado, diferencia=d.diferencia,
        estado=estado_fila(d.diferencia), confianza_media=d.confianza_media)
        for d in detalles_ordenados(cierre)]
    cajas = [CajaOut(x=c.x, y=c.y, w=c.w, h=c.h,
                     clase=clases.get(c.id_producto, CLASE_DESCONOCIDO),
                     conf_det=c.conf_detector, conf_cls=c.conf_clasificador,
                     indice_imagen=c.indice_imagen)
             for c in sorted(cierre.detecciones, key=lambda c: c.id or 0)]
    carpeta = _carpeta_cierre(dir_escaneos, cierre.id)
    originales = sorted(p for p in carpeta.glob("*")
                        if p.is_file() and "_anotada" not in p.stem) if carpeta.is_dir() else []
    return CierreOut(
        **resumen_cierre(cierre).model_dump(),
        observacion=cierre.observacion,
        imagen_anotada_url=_url(cierre.imagen_path) if cierre.imagen_path else None,
        imagenes_urls=[_url(f"{carpeta.name}/{p.name}") for p in originales],
        filas=filas, cajas=cajas,
        no_reconocidos=sum(1 for c in cierre.detecciones if c.id_producto is None),
        tiempo_ms=tiempo_ms)


# ---------------------------------------------------------------- escaneos
def _leer_imagen(archivo: UploadFile) -> tuple[bytes, str]:
    contenido = archivo.file.read()
    if not contenido:
        raise HTTPException(422, f"El archivo {archivo.filename!r} está vacío")
    try:
        with Image.open(io.BytesIO(contenido)) as im:
            formato = im.format or "JPEG"
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(422, f"{archivo.filename!r} no es una imagen válida "
                                 "(usar JPG o PNG; HEIC no está soportado)") from None
    return contenido, _EXTENSIONES.get(formato, f".{formato.lower()}")


@router.post("/escaneos", response_model=CierreOut, status_code=201, tags=["cierres"],
             summary="Sube 1 a 5 fotos del estante, cuenta y crea el cierre pendiente")
def crear_escaneo(request: Request,
                  imagenes: list[UploadFile] = File(..., description="1 a 5 fotos del estante"),
                  usuario: str = Form("cajero"),
                  sesion: Session = Depends(obtener_sesion)):
    if not 1 <= len(imagenes) <= MAX_IMAGENES_ESCANEO:
        raise HTTPException(422, f"Se esperan entre 1 y {MAX_IMAGENES_ESCANEO} imágenes "
                                 f"(llegaron {len(imagenes)})")
    leidas = [_leer_imagen(a) for a in imagenes]
    estado = request.app.state
    dir_escaneos: Path = estado.dir_escaneos

    cierre = iniciar_cierre(sesion, usuario=(usuario or "cajero").strip() or "cajero",
                            variante=estado.variante)
    carpeta = _carpeta_cierre(dir_escaneos, cierre.id)
    try:
        if carpeta.exists():      # resto de un intento fallido que reutilizó el id
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True)
        rutas = []
        for i, (contenido, extension) in enumerate(leidas, start=1):
            ruta = carpeta / f"foto{i}{extension}"
            ruta.write_bytes(contenido)
            rutas.append(ruta)
        try:
            resultado = estado.contar(rutas)
        except Exception as error:
            log.exception("Falló el pipeline de visión")
            raise HTTPException(500, f"Falló el conteo de visión: {error}") from error
        completar_cierre(sesion, cierre, resultado, dir_escaneos)
        sesion.commit()
    except BaseException:
        sesion.rollback()
        shutil.rmtree(carpeta, ignore_errors=True)
        raise
    return cierre_a_salida(sesion, cierre, dir_escaneos, tiempo_ms=float(resultado.tiempo_ms))


# ---------------------------------------------------------------- cierres
@router.get("/cierres", response_model=list[CierreResumen], tags=["cierres"])
def listar_cierres(desde: str | None = Query(None, description="YYYY-MM-DD o ISO"),
                   hasta: str | None = Query(None, description="YYYY-MM-DD (incluye ese día) o ISO"),
                   estado: EstadoCierre | None = None,
                   sesion: Session = Depends(obtener_sesion)):
    consulta = select(Cierre).order_by(Cierre.fecha.desc(), Cierre.id.desc())
    if (d := parsear_fecha(desde, parametro="desde")) is not None:
        consulta = consulta.where(Cierre.fecha >= d)
    if (h := parsear_fecha(hasta, fin_de_dia=True, parametro="hasta")) is not None:
        consulta = consulta.where(Cierre.fecha <= h)
    if estado is not None:
        consulta = consulta.where(Cierre.estado == estado)
    return [resumen_cierre(c) for c in sesion.scalars(consulta)]


@router.get("/cierres/{id}", response_model=CierreOut, tags=["cierres"])
def obtener_cierre_ruta(id: int, request: Request, sesion: Session = Depends(obtener_sesion)):
    return cierre_a_salida(sesion, obtener_cierre(sesion, id), request.app.state.dir_escaneos)


@router.post("/cierres/{id}/confirmar", response_model=CierreOut, tags=["cierres"],
             summary="Confirma el cierre y genera los movimientos de ajuste")
def confirmar_cierre_ruta(id: int, datos: ConfirmarCierreIn, request: Request,
                          sesion: Session = Depends(obtener_sesion)):
    ajustes = {a.id_producto: a.ajustado for a in datos.ajustes}
    cierre = confirmar_cierre(sesion, id, ajustes, observacion=datos.observacion,
                              usuario=datos.usuario or "cajero")
    sesion.commit()
    return cierre_a_salida(sesion, cierre, request.app.state.dir_escaneos)


# ---------------------------------------------------------------- diferencias
@router.get("/diferencias", response_model=list[DiferenciaHistorial], tags=["cierres"],
            summary="Historial por producto y cierre (incluye filas con diferencia 0)")
def listar_diferencias(id_producto: int | None = None,
                       desde: str | None = Query(None, description="YYYY-MM-DD o ISO"),
                       hasta: str | None = Query(None, description="YYYY-MM-DD (incluye ese día) o ISO"),
                       sesion: Session = Depends(obtener_sesion)):
    consulta = (select(CierreDetalle, Cierre.fecha, Producto.nombre)
                .join(Cierre, CierreDetalle.id_cierre == Cierre.id)
                .join(Producto, CierreDetalle.id_producto == Producto.id)
                .order_by(Cierre.fecha, Cierre.id, Producto.id))
    if id_producto is not None:
        consulta = consulta.where(CierreDetalle.id_producto == id_producto)
    if (d := parsear_fecha(desde, parametro="desde")) is not None:
        consulta = consulta.where(Cierre.fecha >= d)
    if (h := parsear_fecha(hasta, fin_de_dia=True, parametro="hasta")) is not None:
        consulta = consulta.where(Cierre.fecha <= h)
    return [DiferenciaHistorial(id_cierre=det.id_cierre, fecha=fecha, id_producto=det.id_producto,
                                nombre=nombre, esperado=det.esperado, detectado=det.detectado,
                                ajustado=det.ajustado, diferencia=det.diferencia)
            for det, fecha, nombre in sesion.execute(consulta)]


# ---------------------------------------------------------------- asistente (RAG)
@router.post("/asistente", response_model=RespuestaAsistente, tags=["asistente"])
def preguntar_asistente(datos: PreguntaIn, sesion: Session = Depends(obtener_sesion)):
    try:
        from src.rag import responder
    except ImportError as error:
        raise HTTPException(503, "El asistente RAG todavía no está disponible "
                                 f"(falta src.rag.responder: {error})") from None
    respuesta = responder(datos.pregunta, sesion)
    if isinstance(respuesta, RespuestaAsistente):
        return respuesta
    return RespuestaAsistente.model_validate(respuesta)
