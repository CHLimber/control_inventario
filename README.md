# Control de inventario por visión (IA + RAG + CNN)

Práctica 02 de Preparación de Decisiones — IA (UAGRM, 2-2026). Es el arqueo de cierre de caja, pero de mercadería: el celular fotografía el estante, el servidor (la PC) detecta, clasifica y cuenta las bebidas, compara el conteo con el stock de la BD, y un asistente RAG responde preguntas sobre las diferencias.

> Estado: **Fase 0** (estructura, entorno y contratos). El resto del README se completa en la Fase 4.

## Instalación

Desde Anaconda Prompt (nunca en `base`):

```bat
cd "C:\Users\limber\Documents\Semestres\2-2026\Preparacion Decisiones\IA\control_inventario"
conda env create -f environment.yml
conda activate inventariovision
copy .env.example .env          &REM y completar las claves
python scripts\verificar_entorno.py
python -m pytest
```

En VS Code, el kernel de los notebooks es `inventariovision`.

## Contratos (no se cambian sin acordarlo)

| Archivo | Qué fija |
|---|---|
| `config/clases.json` | las 9 clases + `desconocido`; el orden es el índice de salida de los modelos |
| `contratos/modelos_bd.py` + `docs/esquema_bd.md` | tablas, stock esperado y conciliación |
| `contratos/vision_types.py` | `Caja`, `ResultadoConteo`, `contar(imagenes) -> ResultadoConteo`, mediana por clase |
| `contratos/api_schemas.py` | endpoints y modelos Pydantic de la API |
| `src/vision/pipeline_mock.py` | pipeline falso con la forma real, para la API y la web |
| `src/config.py` | rutas y variables de entorno |

## Estructura

```
config/       clases.json (fuente de verdad), remapeo_clases.yaml
contratos/    api_schemas.py, modelos_bd.py, vision_types.py
docs/         guía de la práctica 01, esquema_bd.md, ...
notebooks/    01 viabilidad · 02 datos · 03 sintéticos · 04 detector · 05 clasificador · 06 evaluación
src/          db/ api/ conciliacion/ rag/ vision/
web/          escaner/ (celular) · panel/ (PC)
datos/        raw/ fusion/ recortes/ sinteticos/ escaneos/ maqueta/
maqueta/      guía de construcción, protocolo, checklist
modelos/  scripts/  tests/
```
