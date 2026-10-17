# Estantes sintéticos (Fase 2)

Generador de imágenes de **estante de 3 a 5 niveles con una sola fila** (abajo botellas grandes, al medio
botellas chicas o latas, arriba latas; ver `maqueta/GUIA.md`) armadas por *copy-paste* de envases reales recortados
sin fondo. Cada imagen trae **etiquetas YOLO exactas** (1 clase `producto` y 9 clases) y su **conteo exacto**.

Notebook explicativo: `notebooks/03_sinteticos.ipynb`. Figuras de revisión: `docs/sinteticos/muestras.png`
(grilla con cajas) y `docs/sinteticos/distribucion.png`.

## Por qué sintéticos (y sus riesgos)

- **A favor:** conteos y cajas exactos gratis; controlamos la distribución (clases balanceadas, niveles
  medio vacíos, imágenes con **cero** de una clase o vacías del todo); imitan la geometría de la maqueta
  (una fila, 3 niveles, tira portaprecios) que casi no aparece en los datasets públicos.
- **Riesgos — brecha de dominio:** bordes pegados, iluminación y sombras incoherentes, envases fotografiados
  desde arriba (se ve la tapa de la lata), poca resolución de algunos recortes (se ven borrosos al agrandarlos),
  una sola vista por envase. El modelo puede aprender "artefactos de pegado". Por eso **val/test son
  SIEMPRE fotos reales** (fusión + maqueta) y los sintéticos solo se suman al *train*.

## Cómo se usa

```bat
set INVENTARIO_DATOS=<MAIN>\datos
:: 1) biblioteca de objetos sin fondo (una vez; incremental; ~15 min con 4 procesos en CPU)
python scripts/sinteticos_objetos.py --max-por-clase 400 --procesos 4
:: 2) lote
python scripts/sinteticos_generar.py --n 3000 --semilla 1 --salida lote_3000 --procesos 4
::    --figuras escribe docs/sinteticos/*.png ; --param clave=valor cambia un campo de Parametros
python -m pytest tests/test_sinteticos.py
```

Lotes generados (en `datos/sinteticos/`, ignorado por git):

| lote | semilla | imágenes | envases | vacías | tiempo (4 procesos) |
|---|---|---|---|---|---|
| `lote_500` | 0 | 500 | 5 709 | 34 | 42 s (12 img/s) |
| `lote_3000` | 1 | 3000 | 35 100 | 190 | 247 s (12.1 img/s) |

Por clase en `lote_3000`: entre 3 762 y 4 058 envases (balanceado ±4 %). 333 MB en disco.

### Estructura de un lote y rutas para Ultralytics

```
datos/sinteticos/<lote>/
  multiclase/images/*.jpg  multiclase/labels/*.txt  multiclase/data.yaml  multiclase/data_mixto.yaml
  1clase/images/*.jpg      1clase/labels/*.txt      1clase/data.yaml      1clase/data_mixto.yaml
  conteos.csv   imagen + 9 clases + total (verdad exacta; coincide con labels)
  meta.csv      modo (completo / nivel_k), vista (frontal/izq/der), luz, fondo, material, ... por imagen
  resumen.json  parámetros, semilla, tiempo, objetos por clase
```

Ultralytics busca las etiquetas cambiando `/images/` por `/labels/`, así que cada variante es un árbol
propio. Las imágenes de `1clase/images` son **enlaces duros** a las de `multiclase/images` (no duplican disco;
si el disco no lo permite se copian).

- **Entrenamiento recomendado:** `data_mixto.yaml` → `train` = [train real de la fusión, sintéticos],
  `val`/`test` = SOLO reales de la fusión.
  - detector 1 clase: `datos/sinteticos/lote_3000/1clase/data_mixto.yaml` (usa `fusion/detector_1clase`)
  - YOLO multiclase: `datos/sinteticos/lote_3000/multiclase/data_mixto.yaml` (usa `fusion/multiclase`)
- `data.yaml` = solo sintéticos con val = las mismas imágenes: sirve únicamente para probar que entrena
  (o para preentrenar y después ajustar con reales). Verificado con `check_det_dataset` de Ultralytics.
- Los `data*.yaml` tienen rutas absolutas: si se mueve la carpeta, regenerar el lote (o editar `path`).
- Sugerencia: empezar con proporción ~1:1 real:sintético y comparar en val real contra "solo real";
  si los sintéticos empeoran val, bajar su cantidad.

## Decisiones

**Fuente de objetos: SOLO `fusion/multiclase/images|labels/train`** (si se usara val/test, los sintéticos
filtrarían evaluación al entrenamiento). `cargar_biblioteca(..., dir_train)` además descarta objetos cuya
foto de origen ya no esté en train (por si la fusión se regenera y cambian los splits). No se usó
`recortes/` porque son las mismas cajas sin contexto (rembg funciona mejor con margen y así se pueden aplicar
los filtros de "cortado por el borde" y "tapado").

**Filtro de cajas** (`src/sinteticos/objetos.py`): envase completo (no toca el borde de la foto),
**parado** (alto/ancho 2.0–4.6 en botellas, 1.35–2.5 en latas: descarta los muchos envases acostados o
inclinados de darians), alto mínimo 90 px (botella) / 60 px (lata), tapado por otras cajas ≤ 12 %, y como
mucho **uno por foto original** (las copias aumentadas de Roboflow comparten foto).

**Quitar fondo:** `rembg` con **u2net** (medido en CPU: u2netp 0.26 s/recorte, u2net 0.55 s, silueta 0.79 s).
u2netp corta peor bordes y tapas; como es una sola vez y queda en cache, se eligió u2net. Con 4 procesos
y la CPU compartida con otros agentes rindió 1–2.2 recortes/s (950 recortes en ~15 min).
Limpieza de la máscara: se recorta a la caja, se conservan las piezas ≥ 3 % de la mayor que no bajan de ella
(la **tapa** suele quedar separada porque el cuello PET es transparente), se hace la silueta convexa por
filas interpolando el cuello, y se **endereza** (eje principal vertical si está inclinado ≤ 20°). Se descarta
si la máscara cubre < 40 % o > 97 % de la caja, o no ocupa ≥ 85 % del alto / 70 % del ancho.

**Control de color por clase** (revisado a ojo en la biblioteca): darians tiene **etiquetas equivocadas**
(latas con dibujos naranjas/verdes rotuladas `coca_lata`, botellas **vacías** o de Sprite como
`coca_botella`, una Coca como `sprite_botella`, una Coca como `fanta_lata`). Reglas (fracción de píxeles):
coca_lata rojo ≥ 0.35; coca_botella rojo ≥ 0.10 y oscuro ≥ 0.15; fanta_lata naranja ≥ 0.35;
fanta_botella naranja ≥ 0.25; sprite_lata verde ≥ 0.25; sprite_botella verde ≥ 0.15 y oscuro ≤ 0.30;
pepsi azul ≥ 0.15; 7up_lata verde ≥ 0.30. Descartó 44 objetos.

Biblioteca final (`datos/sinteticos/_objetos/`, `indice.csv` + `rechazos.csv` con el motivo):

| coca_bot | coca_lata | fanta_bot | fanta_lata | sprite_bot | sprite_lata | pepsi_bot | pepsi_lata | 7up_lata |
|---|---|---|---|---|---|---|---|---|
| 80 | 114 | 94 | **8** | 117 | 52 | 72 | 52 | 69 |

**Geometría** (`src/sinteticos/composicion.py`, en cm → px): ancho interior 60–80 cm; niveles ≈ 35/25/15 cm
(±); tablas 1–1.8 cm; tira portaprecios 1.8–2.6 cm pegada al canto (con precios "Bs N"), que tapa la base de
los envases como en la tienda. Alto/diámetro reales por nivel: botella grande 29–33 × 9.5–11 cm, chica
20–24 × 6.3–7.5, lata 11.8–12.6 × 6.4–6.8. El recorte conserva su forma si difiere ≤ 20 % de la real (corrige
las imágenes estiradas por Roboflow). Las botellas son la misma clase en los dos niveles; solo cambia la
escala. Envases apoyados en el piso del nivel, en fila, con huecos de Dirichlet (desde parejo hasta
"medio vacío").

**Variación:** cantidad por nivel ~ Beta × capacidad; nivel vacío 12 %; estante vacío 4 % (más los que
quedan vacíos por azar: ~6 % de imágenes vacías); latas en el nivel del medio 35 % (también balancea: hay
4 clases de botella y 5 de lata); productos agrupados (facings) 60 % o mezclados; solapamiento leve con el
vecino 25 % (≤ 12 % del ancho); rotación ±5°; brillo/contraste/saturación/tono ±3 por objeto; borde
*feather* (erosión 1 px + blur σ 0.6–1.2 del alfa); sombra suave 70 %; sombreado interior bajo cada tabla;
fondo cartulina blanca/madera/negro/gris o foto real (`Background_*` de train); pared alrededor (textura o
foto desenfocada); perspectiva leve (frontal o de costado ~20° como `izq20/der20` del protocolo);
luces `fria/natural/calida/poca_luz/flash` (como la columna `luz` de escenas.csv) + gradiente; blur, ruido y
JPEG 70–95. 20 % de las fotos son **acercamientos a un nivel** (60–100 % del ancho; aspecto panorámico).

**Cajas y regla de oclusión:** se pinta un *mapa de ids* igual que la imagen (envases en orden z, después
tablas/tira/laterales en -1) y se le aplica la misma perspectiva con vecino más cercano; la caja es el bbox
de lo **visible**. Un envase con **< 40 % de su silueta visible** (tapado > 60 %) se **quita de la escena**
antes de dibujar (no queda pegado sin etiqueta). En los acercamientos, un envase que quedó mayormente fuera
del cuadro no se etiqueta (`parciales_sin_etiqueta` en meta.csv; con ≥ 40 % visible sí, con la caja recortada).

**Reproducibilidad:** la imagen *i* usa `default_rng([semilla, i])`: mismo resultado con 1 o 4 procesos.

## Niveles y modos de orden (3.ª ronda, 2026-10-17)

- **Niveles:** `Parametros.n_niveles` = 3, 4 o 5 (0 = sortear con `p_niveles`, por defecto 25 / 40 / 35 %). Abajo
  siempre botellas grandes y arriba siempre latas; en el medio, desde arriba, cada nivel lleva latas con
  probabilidad `p_latas_medio` hasta el primero que no, y debajo de ese hay botellas chicas (a veces grandes).
  Así las latas nunca quedan debajo de una botella. Alturas: grande 35 cm, chica 25 cm, lata 15 cm (arriba)
  o 16-25 cm (en el medio).
- **Modo de orden** (uno por estante, columna `orden` de `meta.csv`; `n_niveles` también queda en `meta.csv`):
  - `ordenado` (`p_ordenado` 25 %): bloques contiguos de la misma clase, huecos parejos y **sin solapes**.
  - `desordenado` (`p_desordenado` 25 %): envases inclinados hasta ±22° (`max_rotacion_desorden`), algunos
    **acostados** (`p_caido` 12 %), huecos irregulares, más mezcla y solapes de hasta el 30 %; la regla del 40 %
    visible sigue quitando los que quedan casi tapados.
  - `normal` (el resto): el comportamiento de la primera versión.
- Las cajas siguen saliendo del mapa de ids, así que el desorden no las vuelve inexactas (cubierto en
  `tests/test_sinteticos.py`).

### Biblioteca ampliada con fuentes de góndola

Las fuentes que no entran al multiclase pero tienen envases completos de frente (`cold_drinks_2`,
`hccb_overall`; bandera `"biblioteca": True` en `src/datos/fuentes.py`) alimentan la biblioteca así:

```bat
python scripts/datos_biblioteca.py          :: arma datos/fusion/_biblioteca (solo train, clases resueltas por auto:)
python scripts/sinteticos_objetos.py --fuente datos/fusion/_biblioteca --max-por-clase 400 --procesos 4
```

`sinteticos_generar.py` lee automáticamente las dos carpetas de train (multiclase y `_biblioteca`) para
descartar objetos cuya foto ya no esté en train.

**7Up:** la clase `7up_lata` se quitó del proyecto el 2026-10-17. Sus recortes ya no se usan y la
biblioteca/lotes anteriores (con 9 clases) se deben regenerar.

## Revisión a ojo (qué mirar en la grilla)

`docs/sinteticos/muestras.png`: cajas pegadas a cada envase (sin incluir la parte tapada por la tira),
latas arriba y botellas abajo, envases apoyados (no flotan), bordes sin halo, imágenes con niveles vacíos.
Iteraciones hechas tras mirar: neutros que salían violetas (tinte por canal), reflejo de la imagen en los
bordes al aplicar perspectiva (ahora el lienzo tiene margen real), acercamientos demasiado chicos (ahora
recorren 60–100 % del ancho), tapas perdidas y "palitos" debajo del envase en las máscaras, objetos mal
rotulados (control de color).

Defectos conocidos que quedan: recortes chicos agrandados se ven borrosos en los acercamientos; latas vistas
desde arriba; fondos-foto de darians (pisos con puntos) poco realistas como fondo del estante; algunas
máscaras con restos del fondo original.

## Tests

`tests/test_sinteticos.py` (objetos y fondos falsos en `tmp_path`, sin rembg): etiquetas válidas y dentro de
la imagen, 1clase = multiclase con clase 0, `conteos.csv` = etiquetas, ceros presentes, estante vacío →
etiquetas vacías, reproducible con semilla (bytes idénticos) y distinta con otra semilla, latas por encima
de botellas, clase sin objetos no aparece, filtros de cajas, uno por foto original, unión tapa-cuerpo.

## Pendientes

- **fanta_lata tiene solo 8 objetos** (15 cajas útiles en train): los sintéticos la balancean, pero repiten
  mucho los mismos 8 envases (riesgo de memorizar). Hacen falta más fotos reales de Fanta lata.
- Calidad de etiquetas de darians (ver control de color): conviene revisarla en la fusión (agente Datos).
- Cuando existan fotos de la maqueta vacía en `datos/maqueta/fondos/`, se usan solas como fondo.
- Distractores (otros productos sin etiqueta multiclase) no incluidos.
- Medir el aporte real: entrenar con y sin sintéticos y comparar en val/test reales y en la maqueta.
