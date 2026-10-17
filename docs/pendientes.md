# Pendientes (actualizado el 2026-10-17)

Lista para retomar más tarde. Lo que ya está hecho está en `docs/datos.md`, `docs/sinteticos.md`,
`docs/clasificador.md` y `docs/detector.md`.

## 1. Entrenar en Colab (lo más importante)

El detector todavía no tiene ningún entrenamiento completo y los modelos del clasificador son de 9 clases
(el sistema los rechaza con un error claro).

1. Subir los zips (ya generados, fuera de git):
   - `runs/detector/colab/inventario_detector.zip` (2,9 GB, con el lote de 10 000 sintéticos) a
     `MyDrive/inventario/inventario_detector.zip`.
   - `modelos/clasificador/colab/clasificador_colab.zip` (190 MB, 14 162 recortes) en la celda de subida de
     `notebooks/05_clasificador_colab.ipynb`.
2. Correr `notebooks/04_detector_colab.ipynb` (1 clase y multiclase) y `notebooks/05_clasificador_colab.ipynb`.
3. Traer los resultados: `python scripts/clasificador_empaquetar_colab.py importar <resultados_clasificador.zip>` y,
   para el detector, `python scripts/detector_evaluar.py todo --variante 1clase` (y `multiclase`).
4. **Medir si los sintéticos ayudan:** entrenar el detector de 1 clase con y sin `--con-sinteticos` y comparar en val/test reales.
5. Reentrenar el clasificador con las 8 clases. La corrida local larga de MobileNetV2 (alargando el ajuste fino) no
   mejoró el modelo anterior (test 0,906 contra 0,918), así que alargar épocas no es la palanca: son los datos.
6. Copiar los pesos a `modelos/` y probar con `VARIANTE_VISION=A`.

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
- Probar de punta a punta con el celular cuando haya modelos entrenados.

## 5. Repositorio

- `modelos/clasificador/noche/` y `modelos/clasificador/prueba/` son salidas de corridas locales (sin versionar).
- `notebooks/06_clasificador_jupyter.ipynb` y `07_detector_jupyter.ipynb`: entrenamiento a la vista en Jupyter
  (el detector en CPU solo en modo `demo`).
- Ofrecida y no creada: guía de estudio (`docs/guia_estudio.md`) con ejemplos de los propios resultados.
