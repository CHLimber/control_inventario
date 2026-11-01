# Pendientes (actualizado el 2026-11-01)

Lista para retomar más tarde. Lo que ya está hecho está en `docs/datos.md`, `docs/sinteticos.md`,
`docs/clasificador.md` y `docs/detector.md`.

## 1. Entrenamiento en Colab: hecho el 2026-11-01

Los dos modelos se entrenaron en Colab (T4) y están en `modelos/` (pesos fuera de git):

- **Clasificador** (8 clases, `modelos/clasificador/clasificador.keras`): MobileNetV2 160 px, test acc 0,932,
  F1 macro 0,915, 50 confusiones botella↔lata (cnn 0,886 y cnn_ad 0,901). El de 9 clases daba 0,918.
  Los modelos de 9 clases quedaron en `modelos/clasificador/anterior_9clases/`.
- **Detector** (`modelos/detector/detector_1clase.pt` y `detector_multiclase.pt`, yolo26n, con el lote de 10 000
  sintéticos); métricas en `runs/detector/colab/metricas_colab.json`:

  | Variante | test mAP50 | test mAP50-95 | MAE de conteo | Conteo exacto | Sesgo |
  |---|---|---|---|---|---|
  | 1 clase (3 074 imágenes de test) | 0,954 | 0,677 | 0,43 | 85 % | +0,33 (cuenta de más) |
  | multiclase (547 imágenes de test) | 0,972 | 0,853 | 0,11 | 92 % | +0,08 |

  Los tests no son los mismos (el de 1 clase tiene más datasets), así que las dos variantes no se comparan directo.
  La más floja en multiclase es `pepsi_lata` (mAP50-95 0,777).

Falta:

1. **Probar el sistema con estos modelos** (`VARIANTE_VISION=A` y `B`, y el laboratorio web), en particular si subir
   `UMBRAL_DET` corrige el conteo de más del detector de 1 clase.
2. **Medir si los sintéticos ayudan:** entrenar el detector de 1 clase **sin** sintéticos (empaquetar sin
   `--con-sinteticos`) y comparar en los mismos val/test reales.
3. Pasar los resultados a `docs/clasificador.md` y `docs/detector.md`, y opcionalmente recalcular en la PC con
   `python scripts/detector_evaluar.py todo --variante 1clase` (y `multiclase`).
4. En Colab gratis: guardar `runs` en Drive para poder retomar con `resume=True` si se corta la sesión, y repartir
   las corridas entre cuentas si se acaba la cuota de GPU.

## 2. Datos (el límite real del proyecto)

- **Aprobar el remapeo** de `cold_drinks_2`, `hccb_overall` y `smart_vending` (`config/remapeo_clases.yaml`),
  en particular la regla `auto:<marca>[:formato]` que decide botella o lata por la forma de la caja.
- **Hacer el fork en Roboflow** de `bottle_can_cup` y `tusafe_bottle_can` (28 000 imágenes de latas y botellas) y
  generar una versión; sin eso la API no las exporta. También `cooler-detection-oyu5a` (7 397 imágenes con el formato
  en el nombre: `Fanta Tin`, `SpriteBottle`...), que tampoco tiene versión publicada.
- **Latas flojas:** `fanta_lata` 380 recortes y `sprite_lata` 609 (contra más de 1 500 de las botellas). Son las
  que más conviene reforzar.
- **Biblioteca de envases del simulador:** 31 `fanta_lata`, 72 `pepsi_lata`, 77 `sprite_lata` distintas. Los sintéticos
  repiten esos envases miles de veces: riesgo de memorización (no medido). Ideas: bajar el peso de esas clases
  (`Parametros.pesos_clase`) o conseguir fotos de catálogo con fondo limpio.
- Open Food Facts rindió solo 92 fotos (filtros estrictos y la web rechazó las últimas páginas). Sin probar: Wikimedia
  Commons y Openverse.
- Datasets de Roboflow con marca pero sin formato, sin usar (podrían entrar con `auto:`): `conkbepus`,
  `beverages-3pilb`, `drinks-2d6l6`, `sell-products`, `cashierless-store-2`, `final-lnu6h`. Revisar a ojo antes.
- Descartados: `drinks-detection-opw8o` (imágenes web con colores alterados), `trash-detection-ib8rw` (basura aplastada),
  `trash-pyv9e` (su v4 fusionó todas las marcas en `can`).
- `cold_drinks_2` tiene solo 427 fotos originales (cada una viene unas 23 veces): sus 157 000 cajas son en gran parte repetidas.

## 3. Simulador

- Borrar los lotes obsoletos de 9 clases: `datos/sinteticos/lote_500`, `lote_3000`, `lote_3000_v2`
  (`buscar_sinteticos` elige el más reciente, hoy `lote_10000`).
- Idea para más adelante: simulador 3D (Blender) con las etiquetas de los 8 productos. Pide una foto de frente de cada
  producto del estante real. Valorar solo después de medir si los sintéticos 2D ayudan.

## 4. Sistema y evaluación

- No hay fotos de la maqueta ni de un estante real: todo se mide en datasets públicos y sintéticos. Con 30 o 50 fotos
  de cualquier estante real (aunque sea con el celular) habría un test del dominio real; si no, aclarar la limitación
  en el informe.
- Probar de punta a punta con el celular, ahora que hay modelos entrenados.
- **Laboratorio web** (`/laboratorio/`, `src/api/rutas_laboratorio.py`): elegir cualquier detector (`modelos/detector/*.pt`,
  `runs/detector/*/weights/best.pt`, YOLO de COCO) y cualquier clasificador `.keras` (también los de 9 clases), subir una
  foto y mover los umbrales en vivo. No usa la BD ni está en el contrato de la API.

## 5. Repositorio

- `modelos/clasificador/noche/`, `prueba/` y `anterior_9clases/` son salidas locales (ignoradas en git).
- `notebooks/06_clasificador_jupyter.ipynb` y `07_detector_jupyter.ipynb`: entrenamiento a la vista en Jupyter
  (el detector en CPU solo en modo `demo`).
- Ofrecida y no creada: guía de estudio (`docs/guia_estudio.md`) con ejemplos de los propios resultados.
