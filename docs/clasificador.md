# Clasificador de envases (variante A)

El detector YOLO de una clase (`producto`) recorta cada envase; el clasificador decide cuál de
las **9 clases** de `config/clases.json` es cada recorte, o `"desconocido"` si no está seguro.
Es la práctica 01 (perros y gatos) generalizada a 9 clases en color, más transfer learning.

## Archivos

| archivo | qué hace |
|---|---|
| `src/vision/clasificador_modelos.py` | preprocesamiento común (`letterbox`), aumento de datos y las 3 arquitecturas |
| `src/vision/clasificador.py` | `Clasificador(ruta=None, umbral=UMBRAL_CONF).clasificar(recortes)` (contrato de `contratos/vision_types.py`) |
| `scripts/clasificador_entrenar.py` | entrena, evalúa (val/test), compara y elige el mejor |
| `scripts/clasificador_empaquetar_colab.py` | zip para Colab (`empaquetar`) y vuelta de los resultados (`importar`) |
| `scripts/clasificador_sintetico.py` | mini dataset sintético (tests y medición de tiempos) |
| `notebooks/05_clasificador.ipynb` | notebook local: conceptos, prueba barata, TensorBoard, comparación, matriz de confusión |
| `notebooks/05_clasificador_colab.ipynb` | el mismo entrenamiento en Colab con GPU |
| `tests/test_clasificador*.py` | letterbox, salida, umbral → desconocido, orden de clases, ida y vuelta Colab |

Salidas (carpeta principal, fuera de git): `modelos/clasificador/` y `logs/clasificador/`.

## Uso desde el pipeline

```python
from src.vision.clasificador import Clasificador
clf = Clasificador()                          # modelos/clasificador/clasificador.keras, umbral UMBRAL_CONF
clf.clasificar([recorte_rgb_uint8, ...])      # [("coca_lata", 0.97), ("desconocido", 0.41), ...]
clf.probabilidades(recortes)                  # (n, 9) softmax, por si se quiere otra regla
```

- Recortes: `np.ndarray` RGB uint8 `(H, W, 3)` de cualquier tamaño (también acepta gris/RGBA).
  Ojo: si vienen de OpenCV (`cv2.imread`) están en **BGR**: convertir con `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.
- Clasifica en lote; el modelo se carga en la primera llamada y queda en caché (por ruta y fecha).
- Si `clases.json` junto al modelo no tiene el mismo orden que `config/clases.json` → `ErrorClasesModelo`.
- El tamaño de entrada lo lee del propio `.keras` (128 o 160): no hay que configurarlo.

## Decisiones (para defender)

**RGB y no gris.** El color es la pista principal (Coca roja, Fanta naranja, Sprite/7up verdes,
Pepsi azul). En perros y gatos el gris alcanzaba; acá Coca y Fanta en gris se parecen.

**Letterbox.** Se escala el lado mayor al tamaño final y se rellena con gris (128). Estirar a un
cuadrado haría que una botella parezca una lata, justo lo que hay que distinguir.

**Tamaño.** CNN desde cero: 128 px (100 px deja ilegibles logos de recortes chicos; el costo de
una CNN crece con el cuadrado del lado). MobileNetV2: 160 px, tamaño con pesos de ImageNet
publicados, ~la mitad de cómputo que 224 (160²/224² ≈ 0,51); los pesos ya estaban en la caché de
Keras de la PC.

**Mismo preprocesamiento en entrenamiento e inferencia.** `letterbox()` es la única función;
la normalización (`Rescaling(1/255)` o `Rescaling(1/127.5, offset=-1)` = `preprocess_input` de
MobileNetV2 sin capa `Lambda`) y el aumento van **dentro** del modelo. El `.keras` recibe 0-255.
Hay un test que verifica que el tensor de `tf.data` es idéntico al de inferencia.

**Aumento de datos** (capas Keras, sin `ImageDataGenerator`): `RandomRotation(0.028)` (±10°),
`RandomZoom(0.15)`, `RandomTranslation(0.08)`, `RandomBrightness(0.2)`, `RandomContrast(0.2)`.
- Sin volteo vertical: no hay botellas cabeza abajo en un estante.
- **Sin volteo horizontal**: invierte texto y logo; un "Coca-Cola" en espejo no existe y las marcas
  se separan justamente por logo/tipografía (Sprite vs 7up son ambas verdes). En perros y gatos sí
  servía (un perro en espejo es un perro).
- Sin cambio de tono: cambiaría el color, la pista principal.

**Softmax de 9 + `sparse_categorical_crossentropy`** (en la práctica 01: 1 neurona sigmoid +
`binary_crossentropy`). Las 9 probabilidades suman 1; la máxima es la confianza. `sparse_` porque
las etiquetas son enteros 0..8 (no one-hot). "desconocido" NO es una salida: es la regla
`conf < umbral`.

**Desbalance.** `class_weight` inverso a la frecuencia, topeado en 10. En los datos actuales
`fanta_lata` tiene ~30 recortes de train y 4 en val/test (sus métricas por clase no son
confiables con tan pocos; ver pendientes).

**Modelos comparados**
- `cnn` = `modeloCNN`: Conv32/64/128 + MaxPool, Flatten, Dense100, Dense9 softmax. Sin regularizar.
- `cnn_ad` = `modeloCNN2_AD`: igual + aumento + Dropout(0.5) + Dense250.
- `mobilenet`: transfer learning. MobileNetV2 (ImageNet) congelada + GAP + Dropout(0.3) + Dense9
  (lr 1e-3); después fine-tuning de las últimas 30 capas de la base **sin** BatchNormalization y
  con la base en `training=False`, lr 1e-5.
- Todos: Adam, lote 32, EarlyStopping(val_loss, restaura la mejor), ModelCheckpoint (mejor
  `val_loss`), ReduceLROnPlateau, TensorBoard en `logs/clasificador/<modelo>_<fecha>`.

**Elección.** El de mayor `val_accuracy` (desempata `val_loss`) se copia a `clasificador.keras`.
El test se mira pero no se usa para elegir.

**Umbral sugerido.** Menor umbral ≥ 0,5 (el del contrato) cuyo accuracy sobre los recortes
aceptados llega a 95 % en validación; si ninguno llega, el de mejor accuracy con ≥ 50 % de
cobertura. Queda en `metricas.json` (`umbral_sugerido`); el que se usa es `UMBRAL_CONF` del `.env`.

## Comandos

```bash
# desde la raíz del repo, entorno inventariovision
python scripts/clasificador_entrenar.py --modelo todos --prueba     # 2 épocas, 16 recortes/clase: mide tiempos
python scripts/clasificador_entrenar.py --modelo mobilenet          # entrenamiento real de uno
python scripts/clasificador_entrenar.py --modelo todos --actualizar-doc
python scripts/clasificador_entrenar.py --solo-comparar --actualizar-doc   # rehace tabla y elección
tensorboard --logdir logs/clasificador
python -m pytest tests/test_clasificador*.py
```

En un worktree, definir antes `INVENTARIO_DATOS`, `INVENTARIO_MODELOS` e `INVENTARIO_LOGS`
apuntando a la carpeta principal (el notebook lo hace solo).

## Colab (GPU)

1. En la PC: `python scripts/clasificador_empaquetar_colab.py empaquetar`
   → `modelos/clasificador/colab/clasificador_colab.zip` (código + `datos/recortes`, ~100 MB).
2. En Colab: abrir `notebooks/05_clasificador_colab.ipynb`, entorno con GPU (T4), correr todo;
   la celda 2 pide **subir** `clasificador_colab.zip`. Entrena los 3 modelos y **descarga**
   `resultados_clasificador.zip` (`modelos/clasificador/*` + `logs/clasificador/*`).
3. En la PC: `python scripts/clasificador_empaquetar_colab.py importar <Descargas>/resultados_clasificador.zip`
   → lo ubica en `modelos/clasificador/` y `logs/clasificador/`.
4. `python scripts/clasificador_entrenar.py --solo-comparar --actualizar-doc` para refrescar esta tabla.

## Tiempos medidos en la PC (Ryzen 7 7735HS, CPU, TensorFlow con 4 hilos, otros agentes usando la CPU)

Corrida `--prueba` sobre los recortes reales (16 por clase, 2 épocas; se mide la 2.ª):

| modelo | s por 1000 imágenes (train+val) por época | estimación lineal con ~5600 imágenes |
|---|---|---|
| cnn | 7,0 | ~0,7 min/época × 30 = ~20 min máx. |
| cnn_ad | 6,4 | ~0,6 min/época × 40 = ~24 min máx. |
| mobilenet | 12,4 (cabeza) | ~1,2 min/época × (15 + 10 fine-tuning) = ~29 min máx. |

Una segunda corrida `--prueba` (desde el notebook, con otra carga de CPU) dio cnn 18,4 / cnn_ad
6,6 / mobilenet 11,3 s por 1000 imágenes: con la CPU compartida, la medición varía ×2-3.

**Entrenamiento real de `mobilenet` (medido):** 4958 recortes de train + 661 de val, 15 épocas de
cabeza + 10 de fine-tuning (EarlyStopping no cortó: `val_loss` siguió bajando hasta la época 25)
→ **37,6 min**. Las primeras épocas tardaron 260-180 s (decodificación de los JPEG + otros agentes
entrenando a la vez); las últimas, 45-60 s. La estimación de la prueba (~29 min) se quedó corta
×1,3. Con ese factor, `cnn` y `cnn_ad` costarían cada uno del orden de 25-35 min en esta PC: se
pueden entrenar localmente de noche o en Colab (minutos con GPU).

## Comparación de modelos

<!-- COMPARACION:INICIO -->

_Generada por `scripts/clasificador_entrenar.py` el 2026-10-16 03:17 desde `modelos/clasificador`._

| modelo | entrada | parámetros | épocas | min | val acc | val loss | test acc | test loss | test F1 macro | bot↔lata (test) | sprite↔7up (test) | umbral sug. | n train | prueba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet | 160px | 2.269.513 | 25 | 37.570 | 0.971 | 0.106 | 0.967 | 0.157 | 0.944 | 6 | 1 | 0.500 | 4958 | no |

<!-- COMPARACION:FIN -->

**Estado (2026-10-16):** solo se entrenó `mobilenet` con los datos reales (snapshot de
`datos/recortes` del 2026-10-15 23:19: 6249 recortes; train 4958 / val 661 / test 630). `cnn` y
`cnn_ad` están implementados y probados con la corrida `--prueba`, pero **todavía no se
entrenaron completos**: faltan sus filas en la comparación (local o Colab).

### Lectura del resultado de `mobilenet`

- val accuracy 0,971 / test 0,967; F1 macro test 0,944 (lo baja `fanta_lata`, ver abajo).
- Por clase (test), precisión/recall: coca_botella 0,965/0,965 · coca_lata 1,000/0,964 ·
  fanta_botella 0,973/0,957 · fanta_lata 0,667/1,000 (n=4) · sprite_botella 0,978/0,971 ·
  sprite_lata 0,885/0,958 · pepsi_botella 0,879/1,000 · pepsi_lata 1,000/0,941 · 7up_lata 0,973/1,000.
- Confusiones clave en test: botella↔lata 6 (3 sprite_botella→sprite_lata, 2 pepsi_lata→pepsi_botella,
  1 coca_lata→coca_botella); Sprite↔7up 1 (sprite_lata→7up_lata).
- Fine-tuning: la mejor época congelada tenía val_accuracy 0,968 / val_loss 0,126; tras el
  fine-tuning, 0,971 / 0,106. Mejora chica pero consistente.
- Umbral: en validación, con 0,5 se acepta el 100 % con 97,1 % de acierto (ya supera el 95 %
  objetivo, por eso el sugerido es 0,5); con 0,9 se acepta el 96,1 % con 98,9 % de acierto.
- **`fanta_lata` tiene 30 recortes de train y 4 en val y en test**: sus métricas no son confiables.
  Hacen falta más datos de esa clase.
- Son recortes de cajas anotadas en datasets públicos. Falta medirlo sobre recortes que produzca
  el detector YOLO en fotos de la maqueta (el dominio real).
