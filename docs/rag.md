# Asistente RAG del inventario

El asistente responde en español preguntas como *"¿qué faltó hoy?"*, *"¿cuánta Fanta lata hay?"* o *"¿qué pasó en el cierre 3?"* a partir de la base de datos del inventario. El código está en `src/rag/` y la API lo usa así:

```python
from src.rag import responder
r = responder(pregunta, sesion)        # -> contratos.api_schemas.RespuestaAsistente
```

`responder(pregunta, sesion, cliente=None, hoy=None)`: `sesion` es una sesión SQLAlchemy abierta (solo lee); `cliente` permite inyectar un cliente de Anthropic (los tests pasan uno falso); `hoy` fija la fecha de referencia (por defecto, `date.today()`).

## Cómo funciona: R-A-G en tres pasos

```
pregunta ──► 1. ENTENDER (reglas)         ──► 2. RECUPERAR (SQL)          ──► 3. GENERAR (Claude)
             intención + fechas + productos    consulta + cálculos en Python     redacta con el JSON
             extraccion.py                     consultas.py                      generacion.py
```

1. **Entender** (`extraccion.py`, sin LLM). Se normaliza el texto (minúsculas, sin acentos ni signos) y se aplican reglas:
   - **Intención**, por palabras clave y en orden de prioridad: "cierre N" → `detalle_cierre`; "faltó / sobró / diferencia" → `diferencia_puntual` si nombra un producto y `faltantes_rango` si no; "stock / cuánto / hay / queda" → `stock_producto`; "cierre" sin número → `detalle_cierre` (el último, o el de la fecha pedida); lo demás → `desconocida`.
   - **Fechas**: hoy, ayer, anteayer, esta semana, la semana pasada, últimos N días, este mes, el mes pasado, "en octubre", el lunes, "el 12 de octubre", "12/10", "12/10/2026", "del 1 al 10 de octubre", "entre el 28 de septiembre y el 3 de octubre", "del 1/10 al 8/10". Una fecha sin año que caería en el futuro se toma del año anterior.
   - **Productos**: marca con alias ("coca", "coca cola", "7 up", "seven up"…), formato ("lata", "latas", "botella"…) o clase del modelo ("fanta_lata"). Cada formato se asocia a la marca más cercana: "coca lata y fanta botella" son dos productos; "latas" solo son todas las latas; "coca" sola, las dos Coca-Cola.
2. **Recuperar** (`consultas.py`). Una consulta SQLAlchemy por intención. El resultado es un JSON acotado: `{"resumen": {...}, "registros": [...]}` con 50 filas como máximo. **Todas las cifras las calcula Python**: el stock con la regla de `docs/esquema_bd.md` (último `stock_inicial` + entradas − ventas ± ajustes), y los totales de faltantes y sobrantes por producto.
3. **Generar** (`generacion.py`). Se manda a Claude (`MODELO_CLAUDE`, por defecto `claude-haiku-5-5`, `max_tokens=400`, esfuerzo bajo) la pregunta y el contexto JSON. El *system prompt* le exige responder **solo** con ese contexto, en español, breve y con las cifras exactas, y decir que no hay datos si no alcanza. El LLM solo **redacta**.

Salvaguardas:
- Si la consulta no devuelve nada → `sin_datos=True`, respuesta fija y **no se llama al LLM** (no se gasta ni se arriesga una respuesta inventada).
- Si no hay `ANTHROPIC_API_KEY`, la API falla o el modelo rechaza la respuesta → **plantilla determinista** armada en Python con el mismo contexto (`modelo=None`). La demo funciona sin internet.
- `datos` de la respuesta devuelve los registros usados, para que el panel pueda mostrarlos y el usuario verifique la respuesta.

| Intención | Ejemplo | Qué consulta |
|---|---|---|
| `faltantes_rango` | ¿Qué faltó hoy? / esta semana | `cierre_detalle` con diferencia ≠ 0 en el rango (sin fecha: últimos 7 días) |
| `diferencia_puntual` | ¿Cuánta Coca lata faltó el 8 de octubre? | esperado vs contado de esos productos, cierre por cierre |
| `stock_producto` | ¿Cuánta Fanta lata hay? / ¿cuántas latas había el 4/10? | stock esperado al final de ese día |
| `detalle_cierre` | ¿Qué pasó en el cierre 3? / el último cierre | las 9 filas del cierre, totales, no reconocidos y observación |

## ¿Por qué recuperación estructurada (SQL) y no embeddings?

En un RAG "clásico" los documentos se cortan en fragmentos, cada fragmento se convierte en un vector (*embedding*) y se guarda en una base vectorial; ante una pregunta se buscan los fragmentos con vectores más parecidos y se le pasan al LLM. Eso funciona cuando la información es **texto libre** y lo que importa es el **parecido de significado**.

Acá los datos no son texto: son **filas con números y fechas** (movimientos, cierres, diferencias). Las preguntas tienen una forma muy concreta: *qué producto*, *qué fechas*, *qué dato*. Por eso conviene recuperar con SQL:

1. **Exactitud.** "¿Qué faltó el 12 de octubre?" necesita *exactamente* las filas de ese día. Una búsqueda por similitud devuelve "lo más parecido": el 12 y el 21 de octubre se parecen mucho como texto, y "Coca-Cola lata" y "Coca-Cola botella" también. En inventario, un vecino cercano es un error.
2. **Agregaciones.** El stock es una suma con reglas (último `stock_inicial`, ± ajustes). Un buscador vectorial no suma; tendría que hacerlo el LLM, y los LLM se equivocan haciendo cuentas largas. Con SQL + Python el número es correcto siempre y el LLM solo lo redacta.
3. **Datos vivos.** Cada venta cambia el stock. Con SQL la respuesta usa el estado actual de la BD; con embeddings habría que volver a indexar después de cada movimiento.
4. **Trazabilidad y costo.** Se sabe qué consulta se hizo y qué filas se usaron (`datos`), no hace falta un modelo de embeddings ni otra base de datos, y si no hay datos no se llama al LLM.

El precio de esta decisión es que la **comprensión de la pregunta es por reglas**: una pregunta formulada de una forma que las reglas no contemplan cae en `desconocida`. Para una tienda con un puñado de preguntas típicas es aceptable y fácil de ampliar (agregar una palabra clave o un alias). Una mejora natural sería que el LLM, con *tool use*, eligiera la consulta y sus parámetros en lugar de las reglas, pero manteniendo que las consultas y los cálculos los haga el código.

### En qué se parece y en qué se diferencia de un RAG vectorial

| | Este asistente (estructurado) | RAG con base vectorial |
|---|---|---|
| Fuente | tablas SQL (números, fechas) | documentos de texto en fragmentos |
| Cómo recupera | filtros exactos (fecha, producto, cierre) y agregaciones | similitud de embeddings (top-k) |
| Entender la pregunta | reglas: intención + parámetros | la pregunta se vectoriza tal cual |
| Quién calcula | Python | el LLM, si hace falta (riesgo de error) |
| Actualización | inmediata (consulta la BD en vivo) | reindexar al cambiar los documentos |
| Lo común | **R**ecuperar contexto acotado → **A**umentar el prompt con ese contexto → **G**enerar una respuesta que no se sale de él | ídem |

Lo esencial de un RAG se mantiene: el modelo no responde de memoria, sino a partir de información recuperada de nuestra fuente, y se le prohíbe salirse de ella.

### ¿Cuándo convendría la base vectorial?

Cuando la información está en **texto libre** y las preguntas buscan **significado**, no cifras exactas. En este mismo proyecto serían, por ejemplo:

- **Manuales y procedimientos**: "¿cómo tengo que fotografiar el estante?", "¿qué hago si el escáner no reconoce una botella?".
- **Políticas de la tienda**: "¿cuándo se considera un faltante como pérdida?", "¿quién autoriza un ajuste?".
- **Notas libres**: las `observacion` de los cierres si crecieran mucho ("¿hubo cierres con problemas de luz?", "¿qué cierres mencionan productos dañados?"). Ahí "luz baja", "poca iluminación" y "estaba oscuro" significan lo mismo con palabras distintas, que es justo lo que resuelven los embeddings.

También se pueden combinar (RAG **híbrido**): SQL para los números y búsqueda vectorial para los textos, y el LLM redacta con ambos contextos.

## Pruebas

```bat
python -m pytest tests/test_rag*.py
```

- `tests/test_rag_seed.py`: BD SQLite en memoria propia (8 productos de `config/clases.json`, movimientos del 1 al 15/10/2026 y 3 cierres confirmados con faltantes y sobrantes). El stock esperado se calcula ahí en paralelo, para comprobar la implementación.
- `tests/test_rag_extraccion.py`: fechas, productos, número de cierre e intención.
- `tests/test_rag_responder.py`: cada consulta contra el seed, `sin_datos`, plantilla sin clave y el cliente de Claude **mockeado** (se verifica que recibe el modelo, el *system prompt* y el contexto JSON). Estos tests vacían `ANTHROPIC_API_KEY`: nunca llaman a la API real.
- `tests/test_rag_real.py`: una llamada real, solo si hay `ANTHROPIC_API_KEY` **y** `RAG_PRUEBA_REAL=1` (cuesta dinero).
