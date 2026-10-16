# Detector YOLO (Fase 2)

Fine-tuning de `yolo26n.pt` (COCO) para detectar bebidas. Hay dos variantes con el mismo código:

| Variante | Dataset | Clases | Rol en el sistema | Pesos que se cargan |
|---|---|---|---|---|
| **A** `1clase` | `datos/fusion/detector_1clase` | `producto` | detector del sistema; la marca la pone la CNN Keras sobre el recorte | `modelos/detector/detector_1clase.pt` |
| **B** `multiclase` | `datos/fusion/multiclase` | las 9 de `config/clases.json` | línea base (detecta y clasifica en una sola pasada) | `modelos/detector/detector_multiclase.pt` |

> **Estado:** el código, las pruebas de 1 época y el notebook de Colab están listos. **Todavía no hay un
> entrenamiento completo**, así que no hay mAP ni error de conteo del detector entrenado. Los únicos números
> de este documento son tiempos medidos y la línea base COCO, que se volvió a medir con el código nuevo.

## Archivos

| Archivo | Qué hace |
|---|---|
| `src/vision/detector.py` | `Detector(pesos=None, conf=UMBRAL_DET, imgsz=640).detectar(imagen) -> list[Caja]` (RGB `np.ndarray` o ruta). Carga diferida y cacheada; error claro si faltan los pesos. Con pesos de 1 clase devuelve `clase="producto"`; con multiclase devuelve el nombre de la clase. `conf_cls=None` en los dos casos |
| `scripts/detector_entrenar.py` | entrenamiento (`--variante`, `--modelo`, `--epocas`, `--imgsz`, `--batch`, `--fraction`, `--max-por-origen`, `--con-sinteticos`, `--prueba`, `--device`, `--hilos`, `--workers`). Arma su propio `data.yaml` en `runs/detector/tmp/<corrida>/` y **no toca `datos/`**. Mide los tiempos (`tiempos.json`) y copia `best.pt` a `modelos/detector/detector_<variante>.pt` |
| `scripts/detector_evaluar.py` | `metricas` (mAP@0.5, mAP@0.5:0.95, P y R en val y test, más la matriz de confusión), `conteo` (MAE por imagen en test, en total y por dataset de origen), `seleccion` (las 90 fotos de la viabilidad), `fuga`, `curvas` y `todo` |
| `scripts/detector_empaquetar_colab.py` | arma el zip que se sube a Colab |
| `notebooks/04_detector.ipynb` | notebook para defender la práctica: teoría (detección vs. clasificación, métricas, transfer learning, aumento), datos, fuga, tiempos, curvas, métricas, matriz de confusión, conteo, TensorBoard y una demo del `Detector` |
| `notebooks/04_detector_colab.ipynb` | entrenamiento completo con GPU (las dos variantes), evaluación y descarga |
| `tests/test_detector.py` | forma de las `Caja`, usando `yolo26n.pt` COCO como reemplazo sobre `ultralytics/assets/bus.jpg`. Se salta si no están los pesos |

## Decisiones

- **Transfer learning:** se parte de los pesos COCO, se cambia la cabeza (80 clases → 1 o 9) y se reentrena
  toda la red. Es el estándar con pocos datos y converge en decenas de épocas.
- **Hiperparámetros** (`HIPER` en el script; son los mismos en Colab): 60 épocas en 1clase y 80 en multiclase,
  ambas con `patience=15`, `imgsz=640` y `optimizer=auto`. El batch es 16 en CPU y 32 en la T4.
- **Aumento:** `mosaic=1` (se apaga en las últimas 10 épocas), HSV, `scale=0.5`, `translate=0.1`, `degrees=0`,
  `mixup=0` y **`flipud=0` siempre**, porque en un estante nada está cabeza abajo.
- **`fliplr`:** **0.5 en 1clase** (una botella espejada sigue siendo "producto" y suma variedad) y **0 en
  multiclase** (el texto y el logo espejados no existen en la realidad; la marca se reconoce en parte por el
  texto y el modelo aprendería patrones falsos).
- **`single_cls=True` en 1clase:** junta en "producto" cualquier id de clase. Así los sintéticos pueden venir
  etiquetados con los ids multiclase y sirven igual para el detector de 1 clase.
- **Tope por dataset de origen (`--max-por-origen`, `MAX_POR_ORIGEN` en Colab = 4000 para 1clase):** de las
  19 560 imágenes de train de `detector_1clase`, 14 355 son de `darians_bottle` (416 px, casi siempre **una**
  botella sola, muy distinto de un estante). Con el tope, el train queda en 9205 imágenes (darians 4000,
  rosd 2552, soda 1776, lamar_grocery_color 877): la época dura la mitad y el dataset queda más equilibrado.
  Con `None` se usa todo.
- **Sintéticos:** `--con-sinteticos` agrega como carpeta extra de train la primera que exista de
  `datos/sinteticos/{detector_1clase|multiclase|1clase}/images/train` o `datos/sinteticos/images/train`.
  Las etiquetas van en la carpeta `labels/` hermana, con los ids de `config/clases.json`. El empaquetador
  también acepta `--con-sinteticos`. Hoy `datos/sinteticos/` está vacía.
- **TensorBoard:** los scripts y Colab hacen `settings.update({'tensorboard': True})`. Ojo: es un ajuste
  **global** de Ultralytics para el usuario de Windows y queda activado también en otros proyectos. Los eventos
  quedan en cada corrida: `tensorboard --logdir runs/detector`. En el notebook, la primera celda define
  `TENSORBOARD_BINARY` (el arreglo de la práctica 01).
- **Hilos en CPU:** Ultralytics fija 8 hilos en `select_device`. El script cambia
  `ultralytics.utils.torch_utils.NUM_THREADS` y vuelve a llamar a `torch.set_num_threads` en cada época
  (`--hilos 4`).

## Fuga de datos en la selección de viabilidad

`python scripts/detector_evaluar.py fuga` cruza `datos/viabilidad/resultados/seleccion.csv` con
`datos/fusion/_indice_imagenes.csv` (misma foto original o el mismo grupo de copias aumentadas):

| Dataset | Fotos en train del fusionado | en val | en test |
|---|---:|---:|---:|
| soda-bottles (45) | 36 | 6 | 3 |
| rosd final_hr (45) | 39 | 4 | 2 |

**Hay fuga: 75 de las 90 fotos están en train.** El MAE del detector entrenado sobre esas 90 fotos va a salir
optimista. `detector_evaluar.py seleccion` informa por separado "todas (90)" y "sin fuga" (15 fotos). La
comparación justa es sobre el **split test del fusionado**, corriendo también la línea base COCO sobre esas
mismas imágenes.

## Línea base COCO, medida de nuevo con `detector_evaluar.py`

Es `yolo26n.pt` sin entrenar, contando solo la clase `bottle`, con conf 0.25:

| Conjunto | Imágenes | Bebidas | MAE | % exactas | Sesgo | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selección de viabilidad, las 90 | 90 | 1079 | **9.54** | 12.2 | −9.54 | 0.991 | 0.202 |
| Selección, solo sin fuga | 15 | 223 | 13.07 | 6.7 | −13.07 | 1.00 | 0.121 |
| **Test del fusionado (detector_1clase)** | 2504 | 8263 | **3.03** | 16.8 | −2.99 | – | – |
| · darians_bottle | 1801 | 1666 | 0.88 | 11.9 | −0.86 | | |
| · lamar_grocery_color | 145 | 218 | 0.35 | 77.2 | +0.03 | | |
| · rosd_final_hr | 329 | 819 | 1.25 | 28.6 | −1.23 | | |
| · soda_bottles | 229 | 5560 | 24.17 | 0.0 | −24.17 | | |

La primera fila coincide exactamente con el INFORME de viabilidad, así que el código de evaluación quedó
validado. El MAE total del test está dominado por darians (fotos con una sola botella). Por eso conviene leer
también la tabla **por dataset de origen**: rosd y soda son los más parecidos a una foto con muchas bebidas.
Archivos: `runs/detector/eval_1clase/{seleccion_coco,conteo_test_coco}.csv`.

## Tiempos medidos en CPU (pruebas de 1 época)

Equipo: Ryzen 7 7735HS, sin GPU NVIDIA, torch 2.14.1+cpu, ultralytics 8.4.178. Se usaron 4 hilos de torch,
2 workers, batch 16 e `imgsz=640`, sobre 256 imágenes de train y 64 de val copiadas a
`runs/detector/tmp/<corrida>/`. Otros procesos (el agente Datos, entre otros) usaban la CPU y el disco al
mismo tiempo, así que **los tiempos varían mucho entre corridas**.

| Corrida | Train (s) | Val (s) | s/imagen train | Época completa estimada | 60 épocas estimadas |
|---|---:|---:|---:|---:|---:|
| `1clase_yolo26n_prueba` (train completo: 19 560) | 101.8 | 12.9 | 0.40 | **139 min** | **≈ 139 h** |
| 1clase con tope 4000 (train: 9205) | | | 0.40 | 70 min | ≈ 70 h |
| `multiclase_yolo26n_prueba` (1.ª corrida, train completo de ese momento: 4188) | 92.8 | 14.5 | 0.36 | ≈ 29 min | ≈ 29 h |
| `multiclase_yolo26n_prueba` (2.ª corrida, CPU muy cargada) | 965.4 | 26.2 | 3.77 | (no representativa) | |
| `multiclase_yolo26n_prueba2` (train completo: 4471) | 119.9 | 21.2 | 0.47 | 38 min | ≈ 38 h |

Las pruebas solo verifican que el pipeline funciona de punta a punta. Las métricas de 1 época (por ejemplo, el
multiclase en val: P 0.41, R 0.13, mAP@0.5 0.14) **no dicen nada** sobre el modelo final.

**Decisión: el entrenamiento completo en CPU es inviable.** Serían entre 1,5 y 6 días de CPU dedicada por
variante. Se entrena en **Colab con GPU**. Estimación *no medida* para una T4: yolo26n a 640 procesa del
orden de 100 imágenes/s en entrenamiento, y el límite suelen ser los 2 núcleos de CPU de Colab que hacen el
aumento. Eso da unos 2 a 3 min por época en 1clase con tope 4000 (≈ 2-3 h con early stopping antes de 60) y
alrededor de 1 min por época en multiclase (≈ 1-1,5 h). El notebook guarda el tiempo real en
`tiempo_colab.json` de cada corrida.

## Cómo entrenar en Colab, paso a paso

1. **En la PC, en la raíz del repo principal** (con el fusionado de Datos ya terminado):
   ```
   python scripts/detector_empaquetar_colab.py              # agregar --con-sinteticos cuando existan
   ```
   Esto genera `runs/detector/colab/inventario_detector.zip`, de ~1,3-1,5 GB. Adentro van `detector_1clase/`
   y `multiclase/` con `images`, `labels` y un `data.yaml` **sin `path`** (las rutas son relativas a la
   carpeta del yaml), además de `yolo26n.pt` y `LEEME.txt`.
2. **Subir el zip a Google Drive** en `MyDrive/inventario/inventario_detector.zip`.
3. Abrir `notebooks/04_detector_colab.ipynb` en Colab (`Archivo → Subir notebook`) y elegir
   `Entorno de ejecución → Cambiar tipo → T4 GPU`.
4. `Entorno de ejecución → Ejecutar todo`. Drive pide autorización una vez. Las celdas hacen lo siguiente:
   instalan `ultralytics==8.4.178`, copian y descomprimen el zip, abren TensorBoard, entrenan 1clase y
   multiclase, evalúan mAP, P y R en val y test, muestran la matriz de confusión y las curvas, calculan el
   error de conteo en test y empaquetan los resultados.
5. **Bajar** `resultados_detector.zip`: se descarga solo y además queda copiado en
   `MyDrive/inventario/resultados_detector.zip`.
6. **Descomprimirlo y copiar al repo principal:**

   | Del zip | Destino en MAIN |
   |---|---|
   | `detector_1clase.pt`, `detector_multiclase.pt` | `modelos/detector/` |
   | `runs/1clase/` | `runs/detector/colab_1clase/` |
   | `runs/multiclase/` | `runs/detector/colab_multiclase/` |
   | `runs/eval_*/` | `runs/detector/colab/` |
   | `metricas_colab.json`, `conteo_test_*.csv` | `runs/detector/colab/` |

7. **Verificar en la PC:**
   ```
   python -m pytest tests/test_detector.py
   python scripts/detector_evaluar.py todo --variante 1clase       # mAP, P y R, conteo en test, selección + fuga
   python scripts/detector_evaluar.py todo --variante multiclase
   tensorboard --logdir runs/detector
   ```
   Después, volver a ejecutar `notebooks/04_detector.ipynb`: lee todo lo anterior y compara contra la línea
   base COCO.

Si Colab corta la sesión a mitad de camino, los pesos de cada época quedan en `/content/runs/<variante>/weights/last.pt`
mientras la sesión sigue viva. Se puede retomar con `YOLO(last.pt).train(resume=True)`, o bajar
`MAX_POR_ORIGEN` a 2000 o `EPOCAS` para que entre en el tiempo disponible.

## Entrenamiento local (si alguna vez hay GPU)

```
python scripts/detector_entrenar.py --variante 1clase --device 0 --batch 32 --max-por-origen 4000
python scripts/detector_entrenar.py --variante multiclase --device 0 --batch 32 --epocas 80
python scripts/detector_entrenar.py --variante 1clase --prueba          # 1 época en CPU, mide tiempos
```
Con un worktree de agente, antes hay que definir `INVENTARIO_DATOS`, `INVENTARIO_MODELOS` e `INVENTARIO_LOGS`
apuntando al repo principal.
