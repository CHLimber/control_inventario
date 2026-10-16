"""Asistente RAG del inventario: recuperación estructurada (SQL) + redacción con Claude.

Uso:  from src.rag import responder
      r = responder("¿qué faltó hoy?", sesion)   # -> contratos.api_schemas.RespuestaAsistente
Ver docs/rag.md.
"""
from .asistente import responder

__all__ = ["responder"]
