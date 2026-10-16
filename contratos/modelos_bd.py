"""CONTRATO del esquema de base de datos (SQLAlchemy 2.0). Documentado en docs/esquema_bd.md.

Lo usan el agente BD-API (src/db, src/api, src/conciliacion) y el agente RAG (src/rag).
Cambiar este archivo solo con aprobación del orquestador.

Portabilidad: solo tipos estándar (los "enum" son String + CheckConstraint),
así se pasa de SQLite a MySQL cambiando únicamente URL_BD.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (Boolean, CheckConstraint, DateTime, Float, ForeignKey,
                        Integer, Numeric, String, Text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

TIPOS_MOVIMIENTO = ("stock_inicial", "entrada", "venta", "ajuste")
ESTADOS_CIERRE = ("pendiente", "confirmado")


def _en(columna: str, valores: tuple[str, ...]) -> str:
    return f"{columna} IN ({', '.join(repr(v) for v in valores)})"


class Base(DeclarativeBase):
    pass


class Producto(Base):
    __tablename__ = "producto"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str] = mapped_column(String(100))
    marca: Mapped[str] = mapped_column(String(50))
    formato: Mapped[str] = mapped_column(String(20))           # botella | lata
    precio: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    clase_modelo: Mapped[str] = mapped_column(String(50), unique=True)  # clave en clases.json
    activo: Mapped[bool] = mapped_column(Boolean, default=True)

    movimientos: Mapped[list[Movimiento]] = relationship(back_populates="producto")


class Movimiento(Base):
    """cantidad siempre > 0, salvo en 'ajuste', que lleva signo (+ sobrante, − faltante).

    stock_inicial: fija el stock base; el último stock_inicial anterior a la fecha
    pisa todo lo previo (ver docs/esquema_bd.md).
    """
    __tablename__ = "movimiento"
    __table_args__ = (
        CheckConstraint(_en("tipo", TIPOS_MOVIMIENTO), name="ck_movimiento_tipo"),
        CheckConstraint("tipo = 'ajuste' OR cantidad >= 0", name="ck_movimiento_cantidad"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id_producto: Mapped[int] = mapped_column(ForeignKey("producto.id"), index=True)
    tipo: Mapped[str] = mapped_column(String(20))
    cantidad: Mapped[int] = mapped_column(Integer)
    fecha: Mapped[datetime] = mapped_column(DateTime, index=True)
    usuario: Mapped[str] = mapped_column(String(50), default="cajero")
    # Si el ajuste lo generó la confirmación de un cierre, el id de ese cierre.
    id_cierre: Mapped[int | None] = mapped_column(ForeignKey("cierre.id"), nullable=True)

    producto: Mapped[Producto] = relationship(back_populates="movimientos")


class Cierre(Base):
    __tablename__ = "cierre"
    __table_args__ = (
        CheckConstraint(_en("estado", ESTADOS_CIERRE), name="ck_cierre_estado"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fecha: Mapped[datetime] = mapped_column(DateTime, index=True)
    usuario: Mapped[str] = mapped_column(String(50), default="cajero")
    # Imagen anotada principal, relativa a datos/escaneos/. Las originales están en la misma carpeta.
    imagen_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    estado: Mapped[str] = mapped_column(String(20), default="pendiente")
    variante_modelo: Mapped[str] = mapped_column(String(20))   # mock | A | B
    observacion: Mapped[str | None] = mapped_column(Text, nullable=True)

    detalles: Mapped[list[CierreDetalle]] = relationship(
        back_populates="cierre", cascade="all, delete-orphan")
    detecciones: Mapped[list[Deteccion]] = relationship(
        back_populates="cierre", cascade="all, delete-orphan")


class CierreDetalle(Base):
    """Una fila por producto activo en cada cierre.

    diferencia = (ajustado si está confirmado, si no detectado) − esperado.
    Negativa = faltante; positiva = sobrante.
    """
    __tablename__ = "cierre_detalle"

    id_cierre: Mapped[int] = mapped_column(ForeignKey("cierre.id"), primary_key=True)
    id_producto: Mapped[int] = mapped_column(ForeignKey("producto.id"), primary_key=True)
    esperado: Mapped[int] = mapped_column(Integer)
    detectado: Mapped[int] = mapped_column(Integer)
    ajustado: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diferencia: Mapped[int] = mapped_column(Integer)
    confianza_media: Mapped[float | None] = mapped_column(Float, nullable=True)

    cierre: Mapped[Cierre] = relationship(back_populates="detalles")
    producto: Mapped[Producto] = relationship()


class Deteccion(Base):
    """Cada caja del pipeline. id_producto NULL = 'desconocido'. Coordenadas en píxeles."""
    __tablename__ = "deteccion"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id_cierre: Mapped[int] = mapped_column(ForeignKey("cierre.id"), index=True)
    id_producto: Mapped[int | None] = mapped_column(ForeignKey("producto.id"), nullable=True)
    indice_imagen: Mapped[int] = mapped_column(Integer, default=0)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    conf_detector: Mapped[float] = mapped_column(Float)
    conf_clasificador: Mapped[float | None] = mapped_column(Float, nullable=True)

    cierre: Mapped[Cierre] = relationship(back_populates="detecciones")
