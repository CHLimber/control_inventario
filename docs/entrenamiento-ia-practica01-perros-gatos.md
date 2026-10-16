# Entrenamiento IA — Práctica 01: Clasificador de perros y gatos (CNN + aumento de datos + TensorFlow.js)

Documento de contexto: Parte I resume la práctica y lo que el docente quiso enseñar; Parte II reúne herramientas, modo de trabajo, errores resueltos y plantilla para próximos entrenamientos.

Curso: Preparación de Decisiones — IA (UAGRM, semestre 2-2026).
Estado: completada. Entorno local funcionando (Anaconda + VS Code + Jupyter) e interfaz web con cámara funcionando.
Carpeta en la PC: `C:\Users\limber\Documents\Semestres\2-2026\Preparacion Decisiones\IA\clasificador-perros-gatos`

# Parte I: La práctica

## 1. Qué pidió el docente

El docente hizo en clase una demostración en Google Colab. Después pidió **reproducirla localmente con Anaconda, Jupyter y VS Code** y que terminara en una **interfaz HTML** donde la cámara web clasifica en vivo "Perro" o "Gato".

Material entregado:
- `Clasificador_de_perros_y_gatos.ipynb`: notebook completo (versión limpia y una versión ejecutada en Colab con salidas).
- `index.html`: interfaz web con TensorFlow.js 2.0 y Bootstrap 5 por CDN.
- `group1-shard1of2.bin` y `group1-shard2of2.bin`: pesos de SU modelo `modeloCNN_AD` ya convertidos a TensorFlow.js.
- Logo y favicons.
- **Faltaba `model.json`**, el archivo que carga la página. Sin él la IU no funciona; se reconstruyó (ver sección 5).

## 2. Lo que el docente quiere que se entienda (objetivos de aprendizaje)

La práctica es una comparación de arquitecturas y técnicas sobre el mismo problema de clasificación binaria:

1. **Preprocesamiento de imágenes**: redimensionar a 100x100, pasar a escala de grises (1 canal) y normalizar dividiendo entre 255 (rango 0-1). Forma final de X: `(23262, 100, 100, 1)`.
2. **Salida binaria con sigmoid**: una sola neurona de salida con `sigmoid` y pérdida `binary_crossentropy`. La salida cercana a 0 significa gato (etiqueta 0) y cercana a 1 significa perro (etiqueta 1). El umbral es 0.5. Se usa sigmoid en lugar de softmax a propósito, para mostrar esa alternativa.
3. **Red densa vs. red convolucional (CNN)**: la densa aplana la imagen y pierde la estructura espacial, así que rinde mal. La CNN extrae rasgos locales (bordes, texturas, formas) y rinde mucho mejor.
4. **Sobreajuste (overfitting)**: la CNN sin regularización llega a ~100 % en entrenamiento, pero se estanca en ~83 % en validación. Memoriza en vez de generalizar.
5. **Dropout como regularización**: `modeloCNN2` agrega `Dropout(0.5)` para reducir el sobreajuste.
6. **Aumento de datos (data augmentation)**: genera variantes de las imágenes en cada época (rotación, desplazamiento, zoom, inclinación, volteos), de modo que el modelo nunca ve exactamente la misma imagen. Es la técnica principal contra el sobreajuste en esta práctica.
7. **Monitoreo con TensorBoard**: comparar las curvas de accuracy y loss de entrenamiento vs. validación por época.
8. **Despliegue**: guardar el modelo (`.h5`), convertirlo a TensorFlow.js y ejecutarlo en el navegador con la cámara. Es decir, del entrenamiento en Python a un producto usable.

## 3. Los 6 modelos del notebook

| Modelo | Arquitectura | Datos | Épocas |
|---|---|---|---|
| `modeloDenso` | Flatten → Dense150 relu → Dense150 relu → Dense1 sigmoid | originales, `validation_split=0.15` | 100 |
| `modeloCNN` | Conv32 → MaxPool → Conv64 → MaxPool → Conv128 → MaxPool → Flatten → Dense100 relu → Dense1 sigmoid | originales | 100 |
| `modeloCNN2` | igual que CNN + `Dropout(0.5)` antes de Flatten, Dense250 | originales | 100 |
| `modeloDenso_AD` | igual que Denso | aumento de datos | 100 |
| `modeloCNN_AD` | igual que CNN | aumento de datos | 150 |
| `modeloCNN2_AD` | igual que CNN2 | aumento de datos | 100 |

Todos usan kernels 3x3, MaxPooling 2x2, `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']` y `batch_size=32`.

El modelo que se despliega en la web es **`modeloCNN_AD`**: 1.372.873 parámetros, unos 5,5 MB en float32, repartidos en 2 shards de 4 MB.

**Aumento de datos usado** (`ImageDataGenerator`):
```python
rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
shear_range=15, zoom_range=[0.7, 1.4], horizontal_flip=True, vertical_flip=True
```

**Separación para el entrenamiento con aumento de datos**: índice fijo 19700 (85 %) para entrenamiento y el resto (3562) para validación. El iterador `datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)` se pasa a `fit` con `steps_per_epoch = ceil(19700/32)` y `validation_data=(X_validacion, y_validacion)`.

**Resultados de referencia (salidas del notebook del docente en Colab)**:
- Denso: val_accuracy de ~0.55 a 0.57 a las 100 épocas, casi azar. Confirma que una red densa no sirve para imágenes.
- CNN sin aumento: accuracy de entrenamiento 0.9999 y val_accuracy ~0.83 ya en la época 50. Es un sobreajuste claro.
- Lo esperado con aumento de datos: la brecha entre entrenamiento y validación se reduce y la validación sigue mejorando más épocas.

**Dataset**: `cats_vs_dogs` de TensorFlow Datasets, el set de Kaggle/Microsoft (Asirra). Tiene 23.262 imágenes útiles; 1.738 corruptas se descartan automáticamente.

## 4. Cómo funciona la interfaz web (`index.html`)

1. Pide la cámara con `getUserMedia` (400x400) y dibuja el video en un canvas cada 20 ms.
2. Cada 150 ms reduce el canvas a 100x100 con un filtro Hermite (`resample_single`).
3. Convierte cada píxel a gris con `(R+G+B)/3/255` y arma un tensor `[1, 100, 100, 1]`.
4. Ejecuta `modelo.predict(tensor)`: si el resultado es `<= 0.5` muestra "Gato" y si no, "Perro".
5. Carga el modelo con `tf.loadLayersModel("modelo/model.json")`. En la versión original del docente era `"model.json"` en la misma carpeta.

Detalle técnico: en el entrenamiento el gris se calculó con `cv2.COLOR_BGR2GRAY` sobre imágenes RGB, que es un promedio ponderado con los canales invertidos. La web usa un promedio simple. Son ligeramente distintos, aunque en la práctica funciona igual.

**La página debe servirse por HTTP** (`python -m http.server`). Abierta con doble clic (`file://`), el navegador bloquea la carga de `model.json`.

## 5. Qué se hizo para que funcione localmente

- **Se reconstruyó el `model.json` faltante** a partir de la arquitectura de `modeloCNN_AD` y los `.bin` del docente. Verificación: los bytes de los pesos reexportados son idénticos a los originales; Keras y TensorFlow.js 2.0 dan las mismas predicciones; una foto real de gato da 0.02, o sea "Gato".
- **Se reemplazó `tensorflowjs_converter`** (no instala en Windows con Python moderno) por un exportador propio, `src/exportar_tfjs.py`. Ver la guía de entorno.
- **Se ordenó el proyecto** con una estructura técnica (ver sección 6).
- **Se adaptó el notebook**: celda de configuración de rutas, logs en `logs/`, `.h5` en `modelos/` y exportación directa a `web/modelo/`.
- **Errores resueltos durante la ejecución**: bug de `tensorflow_datasets` en Windows, `tensorboard` no instalado y `%tensorboard` usando el binario equivocado. El detalle y la solución de cada uno están en la guía de entorno.

## 6. Estructura del proyecto

```
clasificador-perros-gatos/
├── README.md
├── environment.yml          entorno conda "perrosgatos"
├── iniciar_web.bat          sirve web/ en http://localhost:8000
├── notebooks/clasificador_perros_gatos.ipynb
├── src/exportar_tfjs.py
├── web/
│   ├── index.html
│   ├── assets/              logo, favicons
│   └── modelo/              model.json + group1-shard*.bin  (lo que usa la página)
├── modelos/
│   ├── docente_tfjs/        respaldo del modelo del docente (TF.js)
│   └── *.h5                 modelos propios
├── logs/                    TensorBoard
└── material_original/       lo que entregó el docente, sin tocar
```

## 7. Cómo correrlo

```
conda activate perrosgatos
cd "C:\Users\limber\Documents\Semestres\2-2026\Preparacion Decisiones\IA\clasificador-perros-gatos"
python -m http.server 8000 --directory web
```
Abrir `http://localhost:8000` y permitir la cámara. El notebook se abre en VS Code con el kernel `perrosgatos`. Al terminar el entrenamiento, la última sección exporta el modelo a `web/modelo/`; después se recarga la página con Ctrl+F5. Para volver al modelo del docente, se copia `modelos/docente_tfjs/*` a `web/modelo/`.

---

# Parte II: Herramientas, entorno y modo de trabajo (reutilizable para próximos entrenamientos)

## 8. Herramientas y para qué sirve cada una

| Herramienta | Rol en el flujo | Notas |
|---|---|---|
| **Google Colab** | Entorno donde el docente hace las demos (Linux, GPU gratis, librerías preinstaladas) | El código del docente asume Colab: `!pip`, `%tensorboard` y rutas Linux. Al pasarlo a Windows hay que adaptar. |
| **Anaconda / Miniconda** | Gestor de entornos de Python (`conda`) | Cada proyecto tiene su entorno aislado. Se usa **Anaconda Prompt**, no el CMD normal. |
| **conda env (`environment.yml`)** | Define Python y librerías del proyecto en un archivo | Se crea con `conda env create -f environment.yml`. Es reproducible. |
| **VS Code** | Editor; corre notebooks con las extensiones **Python** y **Jupyter** | Elegir el kernel del entorno conda en "Select Kernel". |
| **Jupyter Notebook** | Ejecución por celdas, con estado en memoria | También se abre con `jupyter notebook` desde Anaconda Prompt. |
| **TensorFlow + Keras 3** | Definir, entrenar, evaluar y guardar redes neuronales | `tf.keras.models.Sequential`, `compile`, `fit`, `save`. |
| **TensorFlow Datasets (tfds)** | Descarga y prepara datasets públicos (`tfds.load`) | Cache en `C:\Users\limber\tensorflow_datasets`. Solo descarga la primera vez. |
| **OpenCV (`cv2`)** | Redimensionar y pasar a gris (`cv2.resize`, `cv2.cvtColor`) | Paquete `opencv-python`. |
| **NumPy** | Arrays de entrada (`X`, `y`) y normalización | `np.array(X).astype(float) / 255`. |
| **Matplotlib** | Mostrar muestras de imágenes y curvas | `plt.imshow(..., cmap='gray')`. |
| **ImageDataGenerator** | Aumento de datos en tiempo real | Está deprecado en Keras 3 (sale un aviso), pero funciona. Alternativa moderna: capas `RandomFlip`, `RandomRotation`, `RandomZoom`. |
| **TensorBoard** | Gráficas de entrenamiento por época (callback `TensorBoard(log_dir=...)`) | Ya NO viene con TensorFlow ≥2.20: hay que instalarlo. |
| **TensorFlow.js** | Ejecutar el modelo en el navegador | La web usa `tf.loadLayersModel("modelo/model.json")`. |
| **Exportador `exportar_tfjs.py`** | Convierte un modelo Keras a `model.json` + `.bin` | Reemplaza a `tensorflowjs_converter`, que no instala en Windows con Python moderno. |
| **`python -m http.server`** | Servidor web local para la IU | Es obligatorio: con `file://` el navegador bloquea `model.json`. La cámara funciona en `localhost`. |

## 9. Modo de trabajo (cómo encarar cada práctica)

1. **Recibir el material del docente y guardarlo intacto** en `material_original/`. Nunca se trabaja sobre el original.
2. **Leer el notebook entero antes de correr nada**: identificar qué descarga, qué modelos entrena, cuántas épocas tiene, qué guarda y qué despliega. Eso dice qué concepto quiere enseñar.
3. **Detectar lo que es "de Colab"** y adaptarlo: `!pip install` (en local se instala en el entorno), `%tensorboard`, rutas `/content/...`, montaje de Drive y uso de GPU.
4. **Verificar que el material esté completo**. En esta práctica faltaba `model.json`. Para revisarlo: tamaño de los `.bin` = parámetros × 4 bytes.
5. **Crear la estructura del proyecto** (sección 6) y el `environment.yml`.
6. **Crear el entorno y elegir el kernel**. La primera celda imprime `sys.executable` para confirmar que es el entorno correcto.
7. **Correr en orden y probar barato antes de lo caro**: antes de lanzar 100-150 épocas, correr con `epochs=1` para medir el tiempo por época y detectar errores. Tiempo total ≈ tiempo de 1 época × épocas.
8. **Entrenamientos largos**: dejarlos de noche, con la PC enchufada y la suspensión desactivada (Configuración > Sistema > Energía > "Nunca"). Guardar cada modelo apenas termina (`model.save`), para no perderlo si se corta la sesión.
9. **Comparar resultados**: val_accuracy y val_loss de cada modelo, y curvas en TensorBoard. Lo que importa es la **validación**, no el entrenamiento.
10. **Desplegar**: exportar el mejor modelo a `web/modelo/`, servir `web/` y probar con la cámara.

**Reglas de Jupyter que conviene recordar**
- El estado vive en memoria: si se reinicia el kernel, hay que correr de nuevo desde arriba. Los datos descargados quedan en cache en disco.
- `%pip install paquete` instala en el entorno del kernel. `!pip` puede instalar en otro Python.
- Cortar una celda de `fit` a la mitad deja el modelo a medio entrenar y sin guardar.

## 10. Errores encontrados y solución (Windows + Python reciente)

### 10.1 `KeyError: There is no item named 'PetImages\\Cat\\0.jpg' in the archive`
- **Dónde**: `tfds.load('cats_vs_dogs', ...)`.
- **Causa**: bug de `tensorflow_datasets` en Windows. El lector del zip y `os.path.normpath` producen rutas con `\`, pero `zipfile` guarda los nombres con `/`, así que no encuentra la imagen. En Colab (Linux) no ocurre.
- **Solución** (antes de `tfds.load`, después del `setattr` del docente):
```python
import posixpath, types
tfds.image_classification.cats_vs_dogs.os = types.SimpleNamespace(
    path=types.SimpleNamespace(normpath=lambda p: posixpath.normpath(p.replace('\\', '/'))))
```
- Ojo: un primer intento con solo `types.SimpleNamespace(path=posixpath)` NO alcanza, porque el nombre ya llega con `\` desde el lector del zip.

### 10.2 TensorBoard no instalado (falla en `fit` con el callback `TensorBoard`)
- **Causa**: desde TensorFlow 2.20, `tensorboard` ya no es dependencia de `tensorflow`. En Colab viene preinstalado.
- **Solución**: `%pip install tensorboard` en una celda. Si sigue fallando, reiniciar el kernel. Ya quedó agregado al `environment.yml`.

### 10.3 `%tensorboard`: `ModuleNotFoundError: No module named 'tensorboard'` desde `anaconda3\Scripts\tensorboard.exe`
- **Causa**: la magia `%tensorboard` lanza el ejecutable `tensorboard` que aparece primero en el PATH. VS Code no activa el entorno, así que toma el de Anaconda **base**.
- **Solución** (celda antes de `%tensorboard`):
```python
import os, sys
from pathlib import Path
os.environ['TENSORBOARD_BINARY'] = str(Path(sys.executable).parent / 'Scripts' / 'tensorboard.exe')
```
- **Alternativa**: desde Anaconda Prompt, `conda activate perrosgatos` y `tensorboard --logdir logs`, y abrir `http://localhost:6006`.

### 10.4 `pip install tensorflowjs` / `tensorflowjs_converter` fallan
- **Causa**: `tensorflowjs` arrastra dependencias (por ejemplo `tensorflow-decision-forests`) sin soporte en Windows ni en Python 3.12+. Las versiones viejas rompen con NumPy 2 (`np.object`).
- **Solución**: usar `src/exportar_tfjs.py` (sección 11). Otra opción es convertir en Colab subiendo el `.h5`.

### 10.5 La página no carga el modelo
- **Causas típicas**: se abrió con doble clic (`file://`), falta `model.json`, o los `.bin` no coinciden con `model.json`.
- **Revisar**: servir con `python -m http.server`; en la consola del navegador (F12) debe aparecer "Modelo cargado"; los nombres de `paths` en `model.json` deben coincidir con los archivos `.bin`; y el tamaño total de los `.bin` debe ser igual a parámetros × 4.

### 10.6 Avisos que NO son errores
- `UserWarning: Do not pass an input_shape...`: es estilo de Keras 3; no afecta.
- `ImageDataGenerator` deprecado: funciona igual.
- Guardar en `.h5` avisa que es un formato legado: funciona igual.
- `1738 images were corrupted and were skipped`: es normal en `cats_vs_dogs`.

## 11. Exportador Keras → TensorFlow.js (`src/exportar_tfjs.py`)

Soporta modelos `Sequential` con `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten` y `Dense`. Genera la topología en formato Keras 2, compatible con TensorFlow.js 2.x y posteriores, y pesos float32 en shards de 4 MB, igual que el convertidor oficial. Fue verificado: TF.js 2.0 da las mismas predicciones que Keras, y con los pesos del docente los bytes salen idénticos.

Uso:
```python
import sys; sys.path.insert(0, '../src')
from exportar_tfjs import exportar_tfjs
exportar_tfjs(modelo, '../web/modelo')
```
Para otras capas (BatchNormalization, GlobalAveragePooling, etc.) hay que agregar su caso en `_capa_a_config`.

```python
import json
from pathlib import Path
import numpy as np

SHARD_BYTES = 4 * 1024 * 1024

def _init(nombre):
    return {"class_name": nombre, "config": {"seed": None} if nombre != "Zeros" else {}}

def _capa_a_config(capa, primera, input_shape):
    tipo = capa.__class__.__name__
    cfg = {"name": capa.name, "trainable": True, "dtype": "float32"}
    if primera:
        cfg["batch_input_shape"] = [None] + list(input_shape)
    if tipo == "Conv2D":
        cfg.update({"filters": capa.filters, "kernel_size": list(capa.kernel_size),
            "strides": list(capa.strides), "padding": capa.padding,
            "data_format": "channels_last", "dilation_rate": list(capa.dilation_rate),
            "activation": capa.activation.__name__, "use_bias": capa.use_bias,
            "kernel_initializer": _init("GlorotUniform"), "bias_initializer": _init("Zeros"),
            "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None,
            "kernel_constraint": None, "bias_constraint": None})
    elif tipo == "MaxPooling2D":
        cfg.update({"pool_size": list(capa.pool_size), "strides": list(capa.strides),
                    "padding": capa.padding, "data_format": "channels_last"})
    elif tipo == "Dropout":
        cfg.update({"rate": float(capa.rate), "noise_shape": None, "seed": None})
    elif tipo == "Flatten":
        cfg.update({"data_format": "channels_last"})
    elif tipo == "Dense":
        cfg.update({"units": capa.units, "activation": capa.activation.__name__,
            "use_bias": capa.use_bias,
            "kernel_initializer": _init("GlorotUniform"), "bias_initializer": _init("Zeros"),
            "kernel_regularizer": None, "bias_regularizer": None, "activity_regularizer": None,
            "kernel_constraint": None, "bias_constraint": None})
    else:
        raise ValueError(f"Capa no soportada por este exportador: {tipo}")
    return {"class_name": tipo, "config": cfg}

def exportar_tfjs(modelo, carpeta_salida):
    carpeta = Path(carpeta_salida); carpeta.mkdir(parents=True, exist_ok=True)
    input_shape = tuple(modelo.inputs[0].shape[1:])
    capas, pesos, manifiesto = [], [], []
    for i, capa in enumerate(modelo.layers):
        capas.append(_capa_a_config(capa, i == 0, input_shape))
        for nombre, w in zip(["kernel", "bias"], capa.get_weights()):
            w = np.asarray(w, dtype="<f4"); pesos.append(w.ravel())
            manifiesto.append({"name": f"{capa.name}/{nombre}", "shape": list(w.shape), "dtype": "float32"})
    datos = np.concatenate(pesos).tobytes()
    for viejo in carpeta.glob("group1-shard*.bin"): viejo.unlink()
    n = max(1, int(np.ceil(len(datos) / SHARD_BYTES))); rutas = []
    for k in range(n):
        nombre = f"group1-shard{k + 1}of{n}.bin"
        (carpeta / nombre).write_bytes(datos[k * SHARD_BYTES:(k + 1) * SHARD_BYTES]); rutas.append(nombre)
    model_json = {"format": "layers-model", "generatedBy": "keras v2.15.0", "convertedBy": "exportar_tfjs.py",
        "modelTopology": {"keras_version": "2.15.0", "backend": "tensorflow",
            "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": capas}},
            "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"],
                                "optimizer_config": {"class_name": "Adam", "config": {}}}},
        "weightsManifest": [{"paths": rutas, "weights": manifiesto}]}
    (carpeta / "model.json").write_text(json.dumps(model_json), encoding="utf-8")
    print(f"Exportado en {carpeta.resolve()}: model.json + {n} .bin ({modelo.count_params():,} parámetros)")
```

## 12. Plantilla para un nuevo proyecto de entrenamiento

**`environment.yml` base** (ajustar el nombre y agregar librerías):
```yaml
name: <nombre-proyecto>
channels: [conda-forge, nodefaults]
dependencies:
  - python=3.12
  - pip
  - pip:
      - tensorflow
      - tensorboard
      - tensorflow-datasets
      - importlib_resources
      - opencv-python
      - matplotlib
      - pandas
      - ipykernel
      - notebook
```

**Primera celda de todo notebook**:
```python
import os, sys
from pathlib import Path
RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DIR_MODELOS, DIR_WEB_MODELO, DIR_LOGS = RAIZ/'modelos', RAIZ/'web'/'modelo', RAIZ/'logs'
for d in (DIR_MODELOS, DIR_WEB_MODELO, DIR_LOGS): d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RAIZ/'src'))
os.environ['TENSORBOARD_BINARY'] = str(Path(sys.executable).parent/'Scripts'/'tensorboard.exe')
print('Raíz:', RAIZ); print('Python:', sys.executable)   # debe ser ...\envs\<entorno>\python.exe
```

**Checklist**
- [ ] Material original copiado a `material_original/`.
- [ ] Leído el notebook: qué datos, qué modelos, cuántas épocas, qué se despliega, qué concepto enseña.
- [ ] Material completo (por ejemplo `model.json` presente y `.bin` coherentes).
- [ ] Entorno creado y kernel correcto (`sys.executable`).
- [ ] Parche de `tfds` si el dataset sale de un zip en Windows.
- [ ] Prueba con `epochs=1` y medición del tiempo.
- [ ] Entrenamientos largos lanzados con la PC sin suspensión, y cada modelo guardado al terminar.
- [ ] Comparación con val_accuracy, val_loss y TensorBoard.
- [ ] Exportación a `web/modelo/` y prueba con `python -m http.server`.

## 13. Glosario rápido

- **Época**: una pasada completa por todos los datos de entrenamiento.
- **Batch**: grupo de imágenes que se procesa antes de actualizar los pesos (aquí, 32).
- **Validación**: datos que el modelo no usa para aprender; miden la generalización.
- **Overfitting**: accuracy de entrenamiento alta con validación estancada o que empeora; el modelo memoriza.
- **Convolución (Conv2D)**: filtros que recorren la imagen y detectan patrones locales.
- **MaxPooling**: reduce la resolución quedándose con el valor máximo de cada ventana; da invariancia a pequeños desplazamientos.
- **Dropout**: apaga neuronas al azar durante el entrenamiento para que la red no dependa de unas pocas.
- **Aumento de datos**: transformaciones aleatorias de las imágenes de entrenamiento para simular más variedad.
- **Sigmoid + binary_crossentropy**: la combinación estándar para clasificación de 2 clases con una sola neurona de salida.
- **Normalización /255**: lleva los píxeles a 0-1, lo que estabiliza el entrenamiento.
- **Shard (`.bin`)**: fragmento de los pesos del modelo en TensorFlow.js; `model.json` describe la arquitectura e indica qué shards cargar.
