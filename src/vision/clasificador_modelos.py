"""Clasificador de envases (variante A): preprocesamiento común y arquitecturas Keras.

Es la práctica 01 (perros y gatos) generalizada de 2 a 8 clases:

    práctica 01                         acá
    ---------------------------------   -------------------------------------------
    gris 100x100, /255 afuera           RGB (el color separa Coca de Fanta), letterbox
    1 neurona sigmoid                   9 neuronas softmax (una por clase)
    binary_crossentropy                 sparse_categorical_crossentropy
    ImageDataGenerator                  capas Random* de Keras DENTRO del modelo
    modeloCNN / modeloCNN2_AD           "cnn" / "cnn_ad" (mismas ideas)
    —                                   "mobilenet": transfer learning + fine-tuning

Regla de oro: el MISMO preprocesamiento en entrenamiento y en inferencia.
`letterbox()` convierte cualquier recorte RGB uint8 en un cuadrado uint8 [0, 255] sin
deformarlo; la normalización (/255 o [-1, 1]) y el aumento de datos van DENTRO de cada
modelo. Así el `.keras` guardado recibe píxeles 0-255 tal cual y quien lo use (el pipeline,
la API) no necesita saber qué arquitectura ganó.

Este módulo importa keras solo dentro de las funciones que construyen modelos, para que
`letterbox()` se pueda usar sin cargar TensorFlow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    import keras

COLOR_RELLENO = 128          # gris neutro: con la normalización de MobileNetV2 queda ≈ 0
NOMBRE_BASE = "base_mobilenetv2"
MODELOS = ("cnn", "cnn_ad", "mobilenet")

# Hiperparámetros por modelo. "tamano" es el lado de la entrada cuadrada:
#   - CNN desde cero: 128 px. Un logo dentro de un recorte de lata ocupa poco; 100 px (los de
#     la práctica 01) lo dejan ilegible, y 160+ multiplica el costo de una red que aprende de cero
#     sin ganar mucho (el costo de las convoluciones crece con el cuadrado del lado).
#   - MobileNetV2: 160 px. Es uno de los tamaños con pesos de ImageNet publicados
#     (96/128/160/192/224); 224 cuesta ~2x en CPU (224²/160² ≈ 1,96) y para 8 clases de logos
#     grandes y de colores muy distintos no se espera que lo justifique. Además los pesos de
#     160 ya estaban en la caché de Keras de esta PC (no hay que descargar nada).
PARAMETROS: dict[str, dict] = {
    "cnn": {"tamano": 128, "lr": 1e-3, "epocas": 30, "aumento": False},
    "cnn_ad": {"tamano": 128, "lr": 1e-3, "epocas": 40, "aumento": True},
    "mobilenet": {"tamano": 160, "lr": 1e-3, "epocas": 15, "aumento": True,
                  "epocas_finetune": 10, "capas_finetune": 30, "lr_finetune": 1e-5},
}

# Aumento de datos (solo actúa al entrenar; en inferencia las capas Random* son la identidad).
#   rotacion 0.028 → ±0.028·360° ≈ ±10°: una foto de estante casi siempre está derecha.
#   SIN volteo vertical: una botella cabeza abajo no aparece en un estante.
#   SIN volteo horizontal: invierte el texto y el logo ("Coca-Cola" en espejo no existe en la
#     realidad) y las marcas se distinguen justamente por su logo/tipografía (Sprite y Fanta,
#     por ejemplo, difieren en color pero comparten forma de lata). Le enseñaría a la red a ignorar la orientación del texto, una pista útil.
#   SIN cambio de tono (hue): el color es la pista principal (Coca roja, Fanta naranja,
#     Sprite verde, Pepsi azul). Brillo y contraste sí: cambian con la luz del local.
AUMENTO = {"rotacion": 0.028, "zoom": 0.15, "traslacion": 0.08, "brillo": 0.2, "contraste": 0.2}


# ----------------------------------------------------------------------------- preprocesamiento
def a_rgb_uint8(imagen: np.ndarray) -> np.ndarray:
    """Normaliza un recorte a RGB uint8 (H, W, 3): acepta gris (H, W), (H, W, 1) o RGBA."""
    arr = np.asarray(imagen)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0          # float en [0, 1]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Se esperaba una imagen (H, W, 3); llegó forma {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"Cantidad de canales no soportada: {arr.shape[-1]}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Recorte vacío (alto o ancho 0)")
    return arr


def letterbox(imagen: np.ndarray, tamano: int) -> np.ndarray:
    """Recorte RGB → cuadrado tamano×tamano uint8 SIN deformar.

    Escala para que el lado mayor mida `tamano` (conservando la proporción) y centra el
    resultado sobre un fondo gris. Una botella alta queda alta y flaca, con bandas grises a
    los costados; si se la estirara a un cuadrado parecería una lata, y botella vs. lata es
    justamente una de las distinciones que tiene que aprender el modelo.
    """
    arr = a_rgb_uint8(imagen)
    alto, ancho = arr.shape[:2]
    escala = tamano / max(alto, ancho)
    nuevo_ancho = max(1, min(tamano, round(ancho * escala)))
    nuevo_alto = max(1, min(tamano, round(alto * escala)))
    img = Image.fromarray(arr).resize((nuevo_ancho, nuevo_alto), Image.Resampling.BILINEAR)
    lienzo = Image.new("RGB", (tamano, tamano), (COLOR_RELLENO,) * 3)
    lienzo.paste(img, ((tamano - nuevo_ancho) // 2, (tamano - nuevo_alto) // 2))
    return np.asarray(lienzo, dtype=np.uint8)


def cargar_imagen(ruta) -> np.ndarray:
    """Lee un archivo de imagen como RGB uint8 (lo que recibe `letterbox`)."""
    with Image.open(ruta) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def preprocesar_lote(recortes, tamano: int) -> np.ndarray:
    """Varios recortes → (n, tamano, tamano, 3) uint8 listo para el modelo."""
    if len(recortes) == 0:
        return np.zeros((0, tamano, tamano, 3), dtype=np.uint8)
    return np.stack([letterbox(r, tamano) for r in recortes])


# ----------------------------------------------------------------------------- arquitecturas
def capas_aumento() -> "keras.Sequential":
    """Aumento de datos con capas Keras (reemplaza al ImageDataGenerator de la práctica 01)."""
    import keras
    from keras import layers

    p, relleno = AUMENTO, float(COLOR_RELLENO)
    return keras.Sequential([
        layers.RandomRotation(p["rotacion"], fill_mode="constant", fill_value=relleno),
        layers.RandomZoom(p["zoom"], fill_mode="constant", fill_value=relleno),
        layers.RandomTranslation(p["traslacion"], p["traslacion"], fill_mode="constant", fill_value=relleno),
        layers.RandomBrightness(p["brillo"], value_range=(0, 255)),
        layers.RandomContrast(p["contraste"], value_range=(0, 255)),
    ], name="aumento")


def compilar(modelo: "keras.Model", lr: float) -> "keras.Model":
    """Adam + sparse_categorical_crossentropy (etiquetas enteras 0..8, no one-hot)."""
    import keras

    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


def construir_cnn(n_clases: int, tamano: int = 128, lr: float = 1e-3,
                  dropout: bool = False, aumento: bool = False, nombre: str = "cnn") -> "keras.Model":
    """CNN desde cero, igual que la de la práctica 01 pero con salida softmax.

    dropout=False, aumento=False → `modeloCNN`:  Conv32/64/128 + MaxPool, Flatten, Dense100.
    dropout=True,  aumento=True  → `modeloCNN2_AD`: + Dropout(0.5) antes de Flatten, Dense250
                                   y aumento de datos.
    """
    import keras
    from keras import layers

    entrada = keras.Input((tamano, tamano, 3), name="imagen")
    x = capas_aumento()(entrada) if aumento else entrada
    x = layers.Rescaling(1.0 / 255, name="escala_0_1")(x)          # la "/255" de la práctica
    for i, filtros in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(filtros, (3, 3), activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i}")(x)
    if dropout:
        x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.Flatten(name="aplanar")(x)
    x = layers.Dense(250 if dropout else 100, activation="relu", name="densa")(x)
    salida = layers.Dense(n_clases, activation="softmax", name="clases")(x)
    return compilar(keras.Model(entrada, salida, name=nombre), lr)


def construir_mobilenet(n_clases: int, tamano: int = 160, lr: float = 1e-3,
                        pesos: str | None = "imagenet") -> "keras.Model":
    """Transfer learning: MobileNetV2 (ImageNet) CONGELADA + cabeza densa nueva.

    `pesos=None` crea la base sin pesos preentrenados (lo usan los tests, sin descargas).
    Para la segunda fase (fine-tuning) ver `habilitar_finetune`.
    """
    import keras
    from keras import layers

    entrada = keras.Input((tamano, tamano, 3), name="imagen")
    x = capas_aumento()(entrada)
    # Equivale a mobilenet_v2.preprocess_input ([0, 255] → [-1, 1]) sin una capa Lambda:
    # una Lambda no se puede cargar en el modo seguro de Keras 3.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess_input")(x)
    base = keras.applications.MobileNetV2(input_shape=(tamano, tamano, 3), include_top=False,
                                          weights=pesos, name=NOMBRE_BASE)
    base.trainable = False
    # training=False: las BatchNormalization de la base quedan SIEMPRE en modo inferencia
    # (usan las medias/varianzas de ImageNet), también durante el fine-tuning.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pool_global")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    salida = layers.Dense(n_clases, activation="softmax", name="clases")(x)
    return compilar(keras.Model(entrada, salida, name="mobilenet"), lr)


def habilitar_finetune(modelo: "keras.Model", capas: int, lr: float) -> "keras.Model":
    """Descongela las últimas `capas` de la base, salvo las BatchNormalization, y recompila.

    lr bajo (1e-5): los pesos de ImageNet ya son buenos; con un lr alto se "olvidan".
    Las BatchNormalization siguen congeladas: con lotes chicos sus estadísticas se arruinan.
    """
    from keras import layers

    base = modelo.get_layer(NOMBRE_BASE)
    base.trainable = True
    limite = len(base.layers) - capas
    for i, capa in enumerate(base.layers):
        capa.trainable = i >= limite and not isinstance(capa, layers.BatchNormalization)
    return compilar(modelo, lr)


def construir(nombre: str, n_clases: int, pesos: str | None = "imagenet",
              tamano: int | None = None) -> "keras.Model":
    """Construye "cnn", "cnn_ad" o "mobilenet" con los hiperparámetros de PARAMETROS."""
    if nombre not in PARAMETROS:
        raise ValueError(f"Modelo desconocido: {nombre!r} (opciones: {', '.join(MODELOS)})")
    p = PARAMETROS[nombre]
    tamano = tamano or p["tamano"]
    if nombre == "cnn":
        return construir_cnn(n_clases, tamano, p["lr"], dropout=False, aumento=False, nombre="cnn")
    if nombre == "cnn_ad":
        return construir_cnn(n_clases, tamano, p["lr"], dropout=True, aumento=True, nombre="cnn_ad")
    return construir_mobilenet(n_clases, tamano, p["lr"], pesos)
