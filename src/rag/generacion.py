"""Generación de la respuesta: Claude redacta a partir del contexto JSON; si no se
puede (sin clave, error de red, rechazo), una plantilla determinista.

El LLM nunca calcula: todas las cifras vienen hechas en el contexto.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from src import config

log = logging.getLogger(__name__)

MAX_TOKENS = 400

SYSTEM_PROMPT = """Sos el asistente de inventario de una tienda de bebidas. Al cierre del día \
se fotografía el estante, un modelo de visión cuenta los productos y se compara con el stock \
esperado de la base de datos. Una diferencia negativa es un faltante y una positiva, un sobrante.

Reglas:
- Respondé SOLO con la información del contexto JSON que acompaña a la pregunta. No inventes \
productos, fechas ni cantidades, y no hagas cálculos nuevos: usá las cifras tal como vienen \
(los totales ya están calculados en "resumen").
- Si el contexto está vacío o no alcanza para responder, decí que no hay datos para eso.
- Respondé en español, breve (2 a 5 oraciones o una lista corta), con las cifras exactas y \
nombrando productos, fechas y número de cierre cuando corresponda.
- Texto plano, sin tablas ni encabezados."""


def armar_mensaje(pregunta: str, contexto: dict[str, Any]) -> str:
    return ("Pregunta del usuario:\n" + pregunta.strip() + "\n\n"
            "Contexto (JSON, única fuente de verdad):\n"
            + json.dumps(contexto, ensure_ascii=False, indent=1, default=str))


def _crear_cliente():
    """Cliente real de Anthropic, o None si no hay clave o falta el paquete."""
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("Paquete 'anthropic' no instalado: se usa la plantilla.")
        return None
    return anthropic.Anthropic(timeout=30.0, max_retries=1)


def _parametros_modelo(modelo: str) -> dict[str, Any]:
    """Respuesta corta de redacción: poco esfuerzo y, en Haiku, sin razonamiento extendido
    para que los ~400 tokens se usen en el texto. (En Opus 5.5 'disabled' da 400.)"""
    extra: dict[str, Any] = {"output_config": {"effort": "low"}}
    if modelo.startswith("claude-haiku-5"):
        extra["thinking"] = {"type": "disabled"}
    return extra


def generar_con_llm(pregunta: str, contexto: dict[str, Any], cliente=None) -> tuple[str, str] | None:
    """Devuelve (texto, modelo) o None si hay que caer a la plantilla."""
    if cliente is None:
        cliente = _crear_cliente()
        if cliente is None:
            return None
    modelo = config.MODELO_CLAUDE
    try:
        resp = cliente.messages.create(
            model=modelo,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": armar_mensaje(pregunta, contexto)}],
            **_parametros_modelo(modelo),
        )
    except Exception as e:   # red, 4xx/5xx, clave inválida: el asistente no debe caerse
        log.warning("Fallo la API de Claude (%s): se usa la plantilla.", type(e).__name__)
        return None
    if getattr(resp, "stop_reason", None) == "refusal":
        return None
    texto = "".join(getattr(b, "text", "") for b in getattr(resp, "content", [])
                    if getattr(b, "type", None) == "text").strip()
    return (texto, modelo) if texto else None


# ------------------------------------------------------------------ plantillas
def _u(n: int) -> str:
    return f"{n} unidad" if abs(n) == 1 else f"{n} unidades"


def _rango(r: dict) -> str:
    return f"el {r['desde']}" if r["desde"] == r["hasta"] else f"del {r['desde']} al {r['hasta']}"


def plantilla(intencion: str, contexto: dict[str, Any]) -> str:
    res, regs = contexto.get("resumen", {}), contexto.get("registros", [])
    if intencion == "stock_producto":
        partes = [f"{r['producto']}: {_u(r['stock'])}" for r in regs]
        return (f"Stock esperado al {res['fecha_corte'][:10]}: " + "; ".join(partes) + ".")
    if intencion == "faltantes_rango":
        cierres = ", ".join(f"#{c['id_cierre']}" for c in res["cierres_revisados"])
        if not regs:
            return (f"No hubo faltantes ni sobrantes {_rango(res)} "
                    f"(cierres revisados: {cierres}).")
        partes = []
        for a in res["por_producto"]:
            txt = []
            if a["faltante_total"]:
                txt.append(f"faltaron {_u(a['faltante_total'])}")
            if a["sobrante_total"]:
                txt.append(f"sobraron {_u(a['sobrante_total'])}")
            partes.append(f"{a['producto']}: " + " y ".join(txt))
        return (f"Diferencias {_rango(res)} (cierres {cierres}): " + "; ".join(partes)
                + f". Total: {_u(res['unidades_faltantes'])} faltantes y "
                  f"{_u(res['unidades_sobrantes'])} sobrantes.")
    if intencion == "diferencia_puntual":
        partes = []
        for r in regs:
            if r["diferencia"] < 0:
                que = f"faltaron {_u(-r['diferencia'])}"
            elif r["diferencia"] > 0:
                que = f"sobraron {_u(r['diferencia'])}"
            else:
                que = "sin diferencia"
            partes.append(f"cierre #{r['id_cierre']} ({r['fecha'][:10]}), {r['producto']}: "
                          f"esperado {r['esperado']}, contado {r['contado']}, {que}")
        return "; ".join(partes) + f". Diferencia neta: {res['diferencia_neta']:+d}."
    if intencion == "detalle_cierre":
        dif = [r for r in regs if r["diferencia"] != 0]
        cab = (f"Cierre #{res['id_cierre']} del {res['fecha'].replace('T', ' ')} "
               f"({res['estado']}): esperado {res['total_esperado']}, contado "
               f"{res['total_contado']}, diferencia {res['diferencia_total']:+d}.")
        if dif:
            cab += " Diferencias: " + "; ".join(
                f"{r['producto']} {r['diferencia']:+d} ({r['estado']})" for r in dif) + "."
        else:
            cab += " Todos los productos coincidieron."
        if res.get("no_reconocidos"):
            cab += f" Hubo {res['no_reconocidos']} objetos no reconocidos."
        if res.get("observacion"):
            cab += f" Observación: {res['observacion']}"
        return cab
    return SIN_DATOS


SIN_DATOS = "No hay datos en el inventario para responder esa pregunta."
NO_ENTENDI = ("No entendí la pregunta. Puedo responder sobre faltantes y sobrantes "
              "(\"¿qué faltó hoy?\"), el stock de un producto (\"¿cuánta Fanta lata hay?\") "
              "o el detalle de un cierre (\"¿qué pasó en el cierre 3?\").")
