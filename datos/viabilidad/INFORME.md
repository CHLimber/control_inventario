# Viabilidad del detector (Detector-0): YOLO preentrenado en COCO, sin entrenar

Fecha: 2026-10-15. Notebook: `notebooks/01_detector_viabilidad.ipynb`. Lógica: `scripts/viabilidad_detector.py`
(descarga de datos: `scripts/viabilidad_descargar.py`). Todos los números salen de los CSV de
`datos/viabilidad/resultados/`. Se obtuvieron ejecutando el código; ninguno está estimado a mano.

**Pregunta:** ¿cuánto rinde `yolo26n.pt` (COCO), tal como viene, para contar botellas y latas en una foto?

## 1. Datos usados

Se usaron datasets públicos de Roboflow Universe, descargados en formato YOLOv8 a `datos/viabilidad/salidas/raw/`
(ignorado por git). Solo se tomaron los splits *test* y *valid*, fotos con **3 o más bebidas** y **una sola copia por
foto original** (las copias `<foto>.rf.<hash>.jpg` se agrupan por nombre). Se eligieron como máximo 45 fotos por
dataset con semilla 0.

| Dataset | Escena | Fotos | Bebidas reales | Botellas | Latas | Bebidas/foto (mín–máx) |
|---|---|---:|---:|---:|---:|---|
| `rosd/final_hr` v1 | mesa, vista frontal-oblicua, 640×640, botellas y latas mezcladas, algunas acostadas | 45 | 364 | 110 | 254 | 8,1 (3–13) |
| `roboflow-100/soda-bottles` v4 | interior de heladera, cámara cenital con ojo de pez, 1920×1080, botellas PET apiladas | 45 | 715 | 715 | 0 | 15,9 (5–32) |
| **Total** | | **90** | **1079** | **825** | **254** | 12,0 |

- **Conteo real:** número de cajas de bebida en las etiquetas YOLO. En `rosd` las 16 clases son `*_can` o `*_bottle`
  y se mapean a lata o botella por nombre. En `soda-bottles` las clases son `coca-cola`, `fanta` y `sprite`; se
  revisaron a ojo y son todas botellas. La lista completa está en `resultados/clases_datasets.csv`.
- `darians-workspace-4fake/bottle-detection-q9eas` se descargó, pero no aportó fotos: casi todas tienen 1 objeto y
  ninguna tiene 3 bebidas o más. `ai-k6ynk/bottle-can-cup-dataset-pvesi` no tiene versiones exportables.
- **Limitación:** no hay un estante de comercio con vista frontal. `rosd` es lo más parecido a la foto de un estante;
  `soda-bottles` es un caso difícil (vista cenital, botellas acostadas y tapadas entre sí).

## 2. Método

- El modelo se corrió una sola vez por imagen (`imgsz=640`, CPU, `conf≥0.10`, las 80 clases) y se guardaron las
  detecciones crudas. Después se filtraron por **umbral** (0.25, 0.40, 0.50) y por **variante** de clases COCO:
  `bottle` (39) o `bottle+cup` (39+41).
- **Conteo:** error = detectadas − reales en cada imagen. Se reporta **MAE**, **MAE relativo** (|error|/reales),
  **% de imágenes con conteo exacto** y **sesgo** (error medio; negativo = subconteo).
- **Detección:** emparejamiento voraz por confianza con **IoU ≥ 0.5** contra todas las cajas de bebida (sin importar
  la clase). Con eso se calculan P, R y F1, más el recall separado para botellas y para latas.
- **Tiempo:** 60 imágenes ya cargadas en memoria, 640 px, 5 de calentamiento. Se mide el tiempo de reloj de
  `predict` completo (pre, inferencia y post), con `torch.set_num_threads` en 1 y en 8.

## 3. Resultados (90 fotos, 1079 bebidas)

### 3.1 Tabla principal, total

| Modelo | Umbral | Variante | MAE | MAE rel. | % exacto | Sesgo | % subconteo | P | R | F1 | R botellas | R latas |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | 0.25 | bottle | **9.54** | 68.9 % | **12.2 %** | −9.54 | 87.8 % | 0.991 | **0.202** | 0.336 | 0.118 | 0.476 |
| yolo26n | 0.25 | bottle+cup | 9.52 | 68.6 % | 12.2 % | −9.52 | 87.8 % | 0.986 | 0.203 | 0.337 | 0.118 | 0.480 |
| yolo26n | 0.40 | bottle | 10.03 | 75.5 % | 6.7 % | −10.03 | 93.3 % | 0.994 | 0.162 | 0.279 | 0.104 | 0.350 |
| yolo26n | 0.40 | bottle+cup | 10.03 | 75.5 % | 6.7 % | −10.03 | 93.3 % | 0.994 | 0.162 | 0.279 | 0.104 | 0.350 |
| yolo26n | 0.50 | bottle | 10.34 | 79.6 % | 3.3 % | −10.34 | 96.7 % | 0.993 | 0.136 | 0.240 | 0.101 | 0.252 |
| yolo26n | 0.50 | bottle+cup | 10.34 | 79.6 % | 3.3 % | −10.34 | 96.7 % | 0.993 | 0.136 | 0.240 | 0.101 | 0.252 |
| yolo26s | 0.25 | bottle | 7.98 | 58.6 % | 5.6 % | −7.98 | 94.4 % | 0.958 | 0.321 | 0.481 | 0.274 | 0.472 |
| yolo26s | 0.25 | bottle+cup | **7.71** | 56.6 % | 6.7 % | −7.71 | 93.3 % | 0.943 | **0.336** | 0.496 | 0.293 | 0.476 |
| yolo26s | 0.40 | bottle | 8.86 | 66.8 % | 3.3 % | −8.86 | 96.7 % | 0.975 | 0.255 | 0.404 | 0.230 | 0.335 |
| yolo26s | 0.40 | bottle+cup | 8.82 | 66.6 % | 3.3 % | −8.82 | 96.7 % | 0.975 | 0.258 | 0.408 | 0.234 | 0.335 |
| yolo26s | 0.50 | bottle | 9.56 | 74.0 % | 3.3 % | −9.56 | 96.7 % | 0.977 | 0.198 | 0.330 | 0.196 | 0.205 |
| yolo26s | 0.50 | bottle+cup | 9.53 | 73.9 % | 3.3 % | −9.53 | 96.7 % | 0.977 | 0.200 | 0.332 | 0.199 | 0.205 |

En **ninguna** configuración hubo sobreconteo: el % de imágenes con sobreconteo es 0 en las 12 filas. El modelo es
muy preciso (P ≈ 0.94–0.99: lo que marca casi siempre es una bebida), pero ve pocas.

### 3.2 Por dataset (umbral 0.25)

| Modelo | Variante | Dataset | MAE | MAE rel. | % exacto | P | R | R botellas | R latas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | bottle | rosd (mesa) | **3.38** | 38.9 % | **24.4 %** | 0.991 | **0.577** | 0.809 | 0.476 |
| yolo26n | bottle | soda-bottles (heladera) | 15.71 | 98.9 % | 0.0 % | 1.000 | 0.011 | 0.011 | – |
| yolo26s | bottle+cup | rosd (mesa) | 3.02 | 39.8 % | 11.1 % | 0.991 | 0.621 | 0.955 | 0.476 |
| yolo26s | bottle+cup | soda-bottles (heladera) | 12.40 | 73.4 % | 2.2 % | 0.873 | 0.192 | 0.192 | – |

En la heladera vista desde arriba `yolo26n` prácticamente no detecta nada: 8 detecciones para 715 botellas.

### 3.3 Tiempo de inferencia en esta PC (CPU AMD Ryzen 7 7735HS, sin GPU; torch 2.14.1+cpu; ultralytics 8.4.178)

| Modelo | Hilos torch | ms/img promedio | p50 | p95 | img/s |
|---|---:|---:|---:|---:|---:|
| yolo26n | 1 | 145.9 | 133.6 | 212.3 | 6.9 |
| yolo26n | **8** | **91.7** | 77.4 | **148.5** | 10.9 |
| yolo26s | 1 | 398.5 | 353.9 | 593.3 | 2.5 |
| yolo26s | 8 | 197.2 | 172.7 | 288.8 | 5.1 |

Hay que fijar los hilos con cuidado. La primera llamada a `predict` ejecuta `select_device()`, que hace
`torch.set_num_threads(min(8, cpu−1))` y pisa cualquier valor anterior. Por eso los hilos se fijan **después** de esa
primera llamada, y el script verifica que no cambien. Las mediciones se hicieron con otros procesos del proyecto
corriendo en paralelo, así que hay que tomarlas como orientativas. Para producción conviene usar 8 hilos: con
`yolo26n` cuesta menos de 0,1 s por foto, más que suficiente para fotos sueltas del celular.

## 4. ¿Qué pasa con las latas?

Para cada caja real se buscó la detección con mayor IoU (≥ 0.5, conf ≥ 0.25) entre **las 80 clases COCO**
(`resultados/latas_clase_coco.csv`, `resultados/orientacion_clase_coco.csv`).

| Real (rosd + soda) | yolo26n: `bottle` | `cup` | otra clase | nada |
|---|---:|---:|---:|---:|
| Lata (254) | **48.0 %** | 0.4 % | 2.0 % (`cell phone`) | 49.6 % |
| Botella (825) | 11.8 % | 0 % | 1.9 % (`orange`, `snowboard`…) | 86.3 % |

- **Las latas salen como `bottle`, no como `cup`.** COCO no tiene la clase *lata*, y el modelo trata a la lata
  parada como una botella. `cup` casi nunca aparece (1 de 254), así que la variante `bottle+cup` no aporta
  nada medible con yolo26n. Con yolo26s suma algo de recall en la heladera, pero baja la precisión.
- **Depende de la orientación (rosd, yolo26n):** las latas **de pie** se detectan como `bottle` en un 64 % de los
  casos y las **acostadas** solo en un 3 %. Con las botellas pasa lo mismo: 96 % de pie contra 31 % acostadas.
  `yolo26s` tolera mejor las acostadas (latas 57 %, botellas 81 %).
- Los errores de clase son raros: alguna lata sale como `cell phone` y alguna botella naranja como `orange`.

## 5. Conclusión

**Sin entrenar, `yolo26n` (bottle, conf 0.25) cuenta con MAE = 9,5 bebidas por foto y acierta el conteo exacto
en el 12 % de las fotos. Su recall es 0,20 y su precisión 0,99. En el escenario parecido a un estante (mesa con
vista frontal) mejora a MAE 3,4 (39 % del total), recall 0,58 y 24 % de conteos exactos.**

- **No sirve para contar inventario.** Siempre cuenta de menos (sesgo negativo en el 88–97 % de las fotos), pierde
  la mitad de las latas y casi todo lo que está acostado o se ve desde arriba. Además no distingue marca ni formato.
- **Sí sirve como:** (a) un localizador de alta precisión para preetiquetar fotos de la maqueta (las cajas que
  propone casi siempre son correctas, pero hay que agregar a mano las que faltan); (b) una prueba de que la cadena
  completa de foto, detector y conteo corre en CPU en unos 90 ms por foto.
- **Sobre el umbral:** subirlo solo empeora el resultado. Si se usa un modelo COCO, conviene conf 0.25 o menos.
- **Se justifica entrenar** un detector propio (clase única `producto` o las 9 clases de `config/clases.json`) con
  fotos de estante que incluyan latas de pie y acostadas.

## 6. LÍNEA BASE para comparar el detector entrenado

Configuración de referencia: **`yolo26n.pt` COCO, clase `bottle`, conf = 0.25, imgsz = 640, CPU con 8 hilos**.

| Conjunto | Fotos / bebidas | MAE | MAE rel. | % exacto | Sesgo | P (IoU 0.5) | R (IoU 0.5) | R latas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Total (rosd + soda) | 90 / 1079 | **9.54** | 68.9 % | **12.2 %** | −9.54 | **0.991** | **0.202** | 0.476 |
| Solo rosd (mesa, más parecido al estante) | 45 / 364 | **3.38** | 38.9 % | **24.4 %** | −3.38 | 0.991 | **0.577** | 0.476 |
| Tiempo | 60 imgs | **91.7 ms/img** (p95 148.5) con 8 hilos · 145.9 ms/img (p95 212.3) con 1 hilo | | | | | | |

Techo sin entrenar (`yolo26s`, bottle+cup, 0.25): MAE total 7.71, R 0.336; en rosd, MAE 3.02 y R 0.621. Cuesta
197 ms/img con 8 hilos.

**Criterio sugerido:** el detector entrenado tiene que bajar el MAE de la maqueta claramente por debajo de 1 bebida
por foto, con más del 70 % de conteos exactos y recall de 0,9 o más, sin pasar de unos 150 ms/img en esta CPU.
Para comparar en igualdad de condiciones hay que evaluarlo también sobre esta misma muestra
(`resultados/seleccion.csv`, comando `evaluar`).

## 7. Archivos

- `resultados/seleccion.csv`: las 90 fotos (ruta relativa a `salidas/raw/`, conteos reales).
- `resultados/metricas.csv`, `metricas_por_dataset.csv`, `por_imagen.csv`: métricas.
- `resultados/latas_clase_coco.csv`, `orientacion_clase_coco.csv`: análisis de latas.
- `resultados/tiempos.csv`, `entorno_tiempos.json`: tiempos y entorno.
- `resultados/casos_anotados.csv`: los 12 casos dibujados en `salidas/anotadas/` (no versionado; verde = real,
  rojo = detectado): 4 exactos, 4 de la heladera con 0 detecciones y 4 de la mesa con latas acostadas omitidas.
- `graficos/*.png`: métricas según el umbral, real contra detectado, latas y tiempos.

Para reproducir:
```bat
python scripts\viabilidad_descargar.py --raiz .
python scripts\viabilidad_detector.py todo --raiz . --modelos yolo26n.pt yolo26s.pt
```
