"""Prueba contra la API REAL de Claude (cuesta dinero). Solo corre si:
    ANTHROPIC_API_KEY está definida  Y  RAG_PRUEBA_REAL=1
Ejemplo (PowerShell):  $env:RAG_PRUEBA_REAL="1"; python -m pytest tests/test_rag_real.py
"""
import os

import pytest

from src import config
from src.rag import responder
from tests.test_rag_seed import HOY, crear_sesion_seed

pytestmark = pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY", "").strip() and os.getenv("RAG_PRUEBA_REAL") == "1"),
    reason="Prueba real desactivada: requiere ANTHROPIC_API_KEY y RAG_PRUEBA_REAL=1")


def test_respuesta_real_usa_las_cifras_del_contexto():
    s, _ = crear_sesion_seed()
    try:
        r = responder("¿Qué faltó hoy?", s, hoy=HOY)
    finally:
        s.close()
    print("\n", r.respuesta)
    assert r.modelo == config.MODELO_CLAUDE, "No respondió el LLM (¿clave inválida o error de red?)"
    assert not r.sin_datos and r.respuesta
    assert "3" in r.respuesta and "7" in r.respuesta.lower()   # 3 unidades de 7Up lata
