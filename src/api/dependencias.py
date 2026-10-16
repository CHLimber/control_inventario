"""Dependencias y utilidades compartidas por las rutas."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, time

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session


def obtener_sesion(request: Request) -> Iterator[Session]:
    """Una sesión por petición; las rutas hacen commit explícito."""
    sesion = request.app.state.fabrica_sesiones()
    try:
        yield sesion
    except Exception:
        sesion.rollback()
        raise
    finally:
        sesion.close()


def a_hora_local(valor: datetime) -> datetime:
    """La BD guarda fechas 'naive' en hora local; las que llegan con zona se convierten."""
    if valor.tzinfo is not None:
        return valor.astimezone().replace(tzinfo=None)
    return valor


def parsear_fecha(valor: str | None, *, fin_de_dia: bool = False,
                  parametro: str = "fecha") -> datetime | None:
    """'2026-10-15' (inicio o fin de ese día) o ISO completo '2026-10-15T21:30:00'. Si no, 422."""
    if valor is None or not valor.strip():
        return None
    valor = valor.strip()
    try:
        if len(valor) == 10:
            dia = date.fromisoformat(valor)
            return datetime.combine(dia, time.max if fin_de_dia else time.min)
        return a_hora_local(datetime.fromisoformat(valor))
    except ValueError:
        raise HTTPException(422, f"Parámetro '{parametro}' inválido: {valor!r} "
                                 "(usar YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS)") from None
