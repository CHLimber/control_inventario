"""Motor y sesiones de SQLAlchemy.

La URL sale de src.config.url_bd() (variable URL_BD). Para pasar a MySQL basta
cambiar la URL (p. ej. mysql+pymysql://usuario:clave@host/inventario); lo único
específico de SQLite son los connect_args de abajo.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from contratos.modelos_bd import Base
from src.config import url_bd


def crear_motor(url: str | None = None, eco: bool = False) -> Engine:
    """Crea el motor. Sin URL usa la de la configuración (URL_BD)."""
    url = url or url_bd()
    opciones: dict = {"echo": eco}
    es_sqlite = url.startswith("sqlite")
    if es_sqlite:
        # FastAPI atiende en varios hilos: SQLite debe aceptar conexiones de otros hilos.
        opciones["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            # En memoria: una sola conexión compartida; si no, cada conexión vería una BD vacía.
            opciones["poolclass"] = StaticPool
    else:
        opciones["pool_pre_ping"] = True
    motor = create_engine(url, **opciones)
    if es_sqlite:
        @event.listens_for(motor, "connect")
        def _activar_claves_foraneas(conexion, _registro):
            cursor = conexion.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return motor


def crear_fabrica_sesiones(motor: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: los objetos siguen legibles después del commit (para serializar).
    return sessionmaker(bind=motor, autoflush=False, expire_on_commit=False)


def crear_tablas(motor: Engine) -> None:
    """Crea las tablas del contrato si no existen (no migra ni borra nada)."""
    Base.metadata.create_all(motor)


def borrar_tablas(motor: Engine) -> None:
    Base.metadata.drop_all(motor)


@contextmanager
def sesion_transaccional(fabrica: sessionmaker[Session]) -> Iterator[Session]:
    """`with sesion_transaccional(f) as s:` hace commit al salir o rollback si hubo error."""
    sesion = fabrica()
    try:
        yield sesion
        sesion.commit()
    except Exception:
        sesion.rollback()
        raise
    finally:
        sesion.close()
