# Datos públicos: reporte de calidad y conteos

Generado por `scripts/datos_reporte.py` el 2026-10-15. Los datos viven en `DIR_DATOS` (`datos/raw`, `datos/fusion`, `datos/recortes`; no se versionan).

Pipeline (cada paso es un script idempotente; el notebook `notebooks/02_datos.ipynb` los llama):

1. `datos_descargar.py` → `raw/<alias>/` (última versión, formato YOLOv8; licencia en `_fuente.json`)
2. `datos_remapeo.py` → borrador `config/remapeo_clases.yaml`
3. `datos_indexar.py` → `fusion/_indice_imagenes.csv` (pHash, grupos anti-fuga, split)
4. `datos_fusionar.py` → `fusion/detector_1clase/` y `fusion/multiclase/`
5. `datos_recortes.py` → `recortes/<split>/<clase>/` + `indice_recortes.csv`
6. `datos_reporte.py` → este documento

## Criterios

- **Grupo anti-fuga** = misma foto original: mismo proyecto de Roboflow y mismo nombre sin el sufijo `.rf.<hash>` (copias aumentadas y la misma foto en otra versión), unido con toda imagen de pHash idéntico (duplicados, incluso entre datasets). El split se sortea por grupo (80/10/10, hash determinista); un dataset con uso `test_externo` va entero a test.
- Se usa 1 imagen por foto original (las copias aumentadas offline de Roboflow se descartan: YOLO y Keras aumentan en línea; `--copias-train N` permite usar más en train) y se quitan los duplicados exactos de pHash dentro de cada split. Las fotos distintas de un mismo grupo (ráfagas) se conservan.
- Datasets usados en cada salida: `usar_en` de `src/datos/fuentes.py`. De lamar_grocery_color solo se usa la carpeta train/ (las etiquetas de valid/test están desplazadas).
- **detector_1clase**: toda caja no descartada → `producto`; imágenes sin cajas útiles como negativos (≤ 10 % por split).
- **multiclase**: solo datasets de marcas y cajas de las 9 clases; las otras bebidas quedan como fondo. Se excluyen las imágenes con cajas genéricas sin marca (`Cans`, `Bottle-Detection`...), que podrían esconder una de nuestras marcas sin etiquetar.
- **recortes**: margen 5 %, se descartan cajas con lado < 32 px, borrosas (varianza del Laplaciano < 40 sobre el recorte a 128 px de alto) y duplicadas por pHash (si el duplicado está en otro split, se descarta el de val/test).

## Datasets

| alias | proyecto | versión | licencia | imágenes | estado |
|---|---|---|---|---|---|
| bottle_can_cup | ai-k6ynk/bottle-can-cup-dataset-pvesi | - | - | 0 | no disponible (el proyecto no tiene versiones publicadas) |
| tusafe_bottle_can | tusafes-workspace/bottle-can-data-set | - | - | 0 | no disponible (el proyecto no tiene versiones publicadas) |
| darians_bottle | darians-workspace-4fake/bottle-detection-q9eas | 2 | CC BY 4.0 | 56670 | descargado |
| rosd_final_hr | rosd/final_hr | 1 | CC BY 4.0 | 3466 | descargado |
| coca_raspberry | new-workspace-ihwlk/coca-cola-raspberry | - | - | 0 | no disponible (el proyecto no tiene versiones publicadas) |
| lamar_grocery | lamar-university-venef/grocery-rfn8l | 9 | CC BY 4.0 | 35380 | descargado |
| lamar_grocery_color | lamar-university-venef/grocery-rfn8l | 1 | CC BY 4.0 | 8099 | descargado |
| soda_bottles | roboflow-100/soda-bottles | 4 | CC BY 4.0 | 2249 | descargado |
| sku110k_sub | ultralytics/SKU-110K | - | - | 0 | no disponible (sin descargar) |

Atribución (CC BY 4.0): cada dataset pertenece a su autor en Roboflow Universe (https://universe.roboflow.com/<workspace>/<proyecto>).

### `bottle_can_cup` — ai-k6ynk/bottle-can-cup-dataset-pvesi

**No disponible**: el proyecto no tiene versiones publicadas.


Nota: Licencia declarada CC BY 4.0. Sin versiones publicadas en Universe (15 000 imágenes, clases Bottle/Can/Cup): la API no permite exportarlo. Habría que hacer un fork al workspace propio y generar una versión.

### `tusafe_bottle_can` — tusafes-workspace/bottle-can-data-set

**No disponible**: el proyecto no tiene versiones publicadas.


Nota: Licencia declarada CC BY 4.0. Sin versiones publicadas en Universe (13 800 imágenes, clases Bottle/Can): no exportable. Misma solución que bottle_can_cup (fork + versión).

### `darians_bottle` — darians-workspace-4fake/bottle-detection-q9eas

- **Licencia**: CC BY 4.0 · **versión** v2 (de 2) · uso: `entrenamiento` · tipo: `marcas`
- Preprocesado de Roboflow: auto-orient, resize; aumentos offline: blur, brightness, crop, exposure, flip, hue, image, ninety, noise, rotate, saturation, shear
- Imágenes: **56670** · fotos originales: **23612** · copias aumentadas: 33058 (56670 con sufijo `.rf.<hash>`) · grupos anti-fuga (fotos + duplicados/ráfagas): 11840
- Duplicados por pHash con otra foto: 15595 · con otros datasets: 0 · imágenes sin cajas: 3734
- Tamaño típico: 416×416 px · cajas: 53150 (0.9 por imagen)

| id | clase original | cajas | destino |
|---|---|---|---|
| 1 | Bottle-Detection | 8151 | solo_detector |
| 2 | Cans | 4248 | solo_detector |
| 13 | NonBottle | 3638 | descartar |
| 4 | Coca_Original_Bottle | 2921 | coca_botella |
| 10 | Gatorade_Bottle | 2899 | solo_detector |
| 11 | Lipton_Bottle | 2873 | solo_detector |
| 14 | Penafiel_Limon_Bottel | 2848 | solo_detector |
| 16 | Sprite_Bottle | 2818 | sprite_botella |
| 6 | Electrolit_Bottle | 2814 | solo_detector |
| 8 | Fanta_Bottle | 2799 | fanta_botella |
| 0 | Bonafont_botella | 2798 | solo_detector |
| 3 | Coca_Light_Can | 2430 | solo_detector |
| 7 | Fanta_Berry_Can | 2406 | solo_detector |
| 9 | Fresca_Bottle | 2396 | solo_detector |
| 15 | Penafiel_Naranjada_Bottle | 2383 | solo_detector |
| 12 | Monster_Peach_Can | 2369 | solo_detector |
| 5 | Coca_Original_Can | 2359 | coca_lata |

Clases que no aportan marca: 42253 cajas (79 %).

**Veredicto**: BUENO. Productos mexicanos, casi siempre un envase acostado sobre una mesa, cajas ajustadas al envase entero. Letterbox 416 px y 3 copias aumentadas por foto en train (rotaciones de 90°, flips, recortes): se usa 1 copia por foto. Variantes (Coca Light, Fanta Berry) y otras marcas -> solo_detector. 'Cans' son latas Arizona y 'Bottle-Detection' botellas de agua.

![muestras darians_bottle](datos/muestras_darians_bottle.jpg)

### `rosd_final_hr` — rosd/final_hr

- **Licencia**: CC BY 4.0 · **versión** v1 (de 1) · uso: `entrenamiento` · tipo: `marcas`
- Preprocesado de Roboflow: auto-orient, resize; aumentos offline: ninguno
- Imágenes: **3466** · fotos originales: **3466** · copias aumentadas: 0 (3466 con sufijo `.rf.<hash>`) · grupos anti-fuga (fotos + duplicados/ráfagas): 3225
- Duplicados por pHash con otra foto: 391 · con otros datasets: 0 · imágenes sin cajas: 0
- Tamaño típico: 640×640 px · cajas: 9086 (2.6 por imagen)

| id | clase original | cajas | destino |
|---|---|---|---|
| 1 | beer_can | 2169 | solo_detector |
| 5 | juice_can | 1102 | solo_detector |
| 15 | water_bottle | 868 | solo_detector |
| 7 | mirinda_can | 480 | solo_detector |
| 4 | fanta_bottle | 431 | fanta_botella |
| 10 | redbull_can | 430 | solo_detector |
| 0 | 7up_can | 429 | 7up_lata |
| 14 | tea_can | 423 | solo_detector |
| 13 | tea_bottle | 409 | solo_detector |
| 3 | coke_can | 404 | coca_lata |
| 8 | pepsi_bottle | 400 | pepsi_botella |
| 9 | pepsi_can | 355 | pepsi_lata |
| 11 | sprite_bottle | 346 | sprite_botella |
| 2 | coke_bottle | 344 | coca_botella |
| 6 | milk_can | 259 | solo_detector |
| 12 | sprite_can | 237 | sprite_lata |

Clases que no aportan marca: 6140 cajas (68 %).

**Veredicto**: BUENO y CLAVE: única fuente de pepsi_botella, pepsi_lata y 7up_lata. Envases chinos (logos en chino para Coca y Sprite), fotos de mano/mesa, cajas completas, sin aumentos. Mirinda (lata naranja, parecida a Fanta) queda como solo_detector.

![muestras rosd_final_hr](datos/muestras_rosd_final_hr.jpg)

### `coca_raspberry` — new-workspace-ihwlk/coca-cola-raspberry

**No disponible**: el proyecto no tiene versiones publicadas.


Nota: Licencia declarada CC BY 4.0. No disponible: el proyecto está vacío (0 imágenes, sin versiones).

### `lamar_grocery` — lamar-university-venef/grocery-rfn8l

- **Licencia**: CC BY 4.0 · **versión** v9 (de 3) · uso: `entrenamiento` · tipo: `generico`
- Preprocesado de Roboflow: auto-orient, contrast, resize, grayscale; aumentos offline: blur, brightness, exposure, hue, image, noise, rotate, saturation
- Imágenes: **35380** · fotos originales: **1065** · copias aumentadas: 34315 (35380 con sufijo `.rf.<hash>`) · grupos anti-fuga (fotos + duplicados/ráfagas): 980
- Duplicados por pHash con otra foto: 692 · con otros datasets: 3065 · imágenes sin cajas: 0
- Tamaño típico: 640×640 px · cajas: 110980 (3.1 por imagen)

| id | clase original | cajas | destino |
|---|---|---|---|
| 12 | CocaCola20Oz | 4732 | coca_botella |
| 18 | DietCocaCola20Oz | 4245 | solo_detector |
| 53 | Sprite20Oz | 4178 | sprite_botella |
| 59 | VanillaCocaCola20Oz | 3569 | solo_detector |
| 51 | SkittlesShareSize | 3487 | descartar |
| 27 | FantaCan16Oz | 3476 | fanta_lata |
| 34 | LennyLarrysBirthdayCake | 2713 | descartar |
| 39 | LennyLarrysSnickerdoodle | 2713 | descartar |
| 38 | LennyLarrysPeanutButterChocolateChips | 2656 | descartar |
| 35 | LennyLarrysChocolateChips | 2656 | descartar |
| 40 | MinuteMaidBlueRaspberry20Oz | 2623 | solo_detector |
| 54 | Sprite40Oz | 2589 | sprite_botella |
| 37 | LennyLarrysPeanutButter | 2568 | descartar |
| 26 | Fanta20Oz | 2535 | fanta_botella |
| 29 | FantaZero20Oz | 2464 | solo_detector |
| 11 | CocaCola16Oz | 2393 | coca_lata |
| 36 | LennyLarrysDoubleChocolateChips | 2392 | descartar |
| 42 | MinuteMaidLemonade20Oz | 2196 | solo_detector |
| 15 | CocaColaZero16Oz | 2193 | solo_detector |
| 44 | MountainDew16Oz | 2193 | solo_detector |
| 14 | CocaColaSpiced20Oz | 2030 | solo_detector |
| 58 | SpriteZero20Oz | 2006 | solo_detector |
| 16 | Crunch | 1956 | descartar |
| 0 | BargsBlack20Oz | 1954 | solo_detector |
| 41 | MinuteMaidFruitPunch | 1925 | solo_detector |
| 47 | Oreo | 1874 | descartar |
| 43 | MinuteMaidPinkLemonade | 1872 | solo_detector |
| 57 | SpriteTropicalMix20Oz | 1868 | solo_detector |
| 10 | ChipsAhoyKingSize | 1854 | descartar |
| 49 | OreoKingSize | 1854 | descartar |
| 48 | OreoDoubleStufKingSize | 1854 | descartar |
| 24 | DrPapPepper20Oz | 1826 | solo_detector |
| 9 | ChipsAhoy | 1792 | descartar |
| 56 | SpriteCherry20Oz | 1617 | solo_detector |
| 1 | BuenoShareSize | 1613 | descartar |
| 60 | WhatchamacallitKingSize | 1613 | descartar |
| 46 | NerdsShareSize | 1613 | descartar |
| 52 | SourPunchShareSize | 1613 | descartar |
| 28 | FantaGrape20Oz | 1472 | solo_detector |
| 8 | CherryVanillaCocaCola20Oz | 1395 | solo_detector |
| 19 | DietCokeCan16Oz | 1388 | solo_detector |
| 7 | CherryCocaCola20Oz | 1384 | solo_detector |
| 50 | PaydayShareSize | 1349 | descartar |
| 25 | DrPapPepperCan16Oz | 1342 | solo_detector |
| 55 | SpriteCan16Oz | 1311 | sprite_lata |
| 31 | LaysBarbecue | 963 | descartar |
| 33 | LaysLimon | 955 | descartar |
| 4 | CheetosCrunchyFlaminHotLimon | 954 | descartar |
| 6 | CheetosPuffs | 948 | descartar |
| 21 | DoritosNachoCheese | 935 | descartar |
| 20 | DoritosCoolRanch | 915 | descartar |
| 2 | CheetosCrunchy | 662 | descartar |
| 22 | DoritosSpicyNacho | 654 | descartar |
| 5 | CheetosCrunchyXXTRAFlaminHot | 646 | descartar |
| 30 | FunyunsOnionFlavoredRingsFlaminHot | 586 | descartar |
| 32 | LaysClassic | 523 | descartar |
| 3 | CheetosCrunchyFlaminHot | 409 | descartar |
| 17 | Crush16Oz | 220 | solo_detector |
| 45 | MtnDew16Oz | 208 | solo_detector |
| 61 | ZeroCocaCola16Oz | 200 | solo_detector |
| 23 | DrPapPepper1L | 164 | solo_detector |
| 13 | CocaCola350Ml | 92 | solo_detector |

Clases que no aportan marca: 89766 cajas (81 %).

**Veredicto**: NO SE USA. La última versión (v9) viene en ESCALA DE GRISES (preprocesado grayscale + contraste), así que no sirve para marcas, y comparte fotos con la v1 (lamar_grocery_color). Igual que en la v1, las etiquetas de valid/test están DESPLAZADAS respecto de los productos. Más de la mitad de las clases son snacks (descartar).

![muestras lamar_grocery](datos/muestras_lamar_grocery.jpg)

### `lamar_grocery_color` — lamar-university-venef/grocery-rfn8l

- **Licencia**: CC BY 4.0 · **versión** v1 (de 3) · uso: `entrenamiento` · tipo: `marcas`
- Preprocesado de Roboflow: auto-orient, resize; aumentos offline: blur, brightness, exposure, image, noise, rotate
- Imágenes: **8099** · fotos originales: **1374** · copias aumentadas: 6725 (8099 con sufijo `.rf.<hash>`) · grupos anti-fuga (fotos + duplicados/ráfagas): 1351
- Duplicados por pHash con otra foto: 127 · con otros datasets: 1597 · imágenes sin cajas: 28
- Tamaño típico: 640×640 px · cajas: 15106 (1.9 por imagen)

| id | clase original | cajas | destino |
|---|---|---|---|
| 18 | DietCoca-cola 20Oz | 483 | solo_detector |
| 15 | DORITOS-Cool-Ranch | 483 | descartar |
| 28 | Lay-s-Limon | 483 | descartar |
| 4 | Cheetos-Puffs | 476 | descartar |
| 6 | Cheetos-crunchy-XXTRA-Flamin-Hot | 469 | descartar |
| 25 | Funyuns-Onion-flavored-rings-Flamin-hot | 469 | descartar |
| 17 | DORITOS-Spicy-Nacho | 469 | descartar |
| 16 | DORITOS-Nacho-Cheese | 469 | descartar |
| 2 | Cheetos-Crunchy | 469 | descartar |
| 5 | Cheetos-crunchy-Flamin-Hot-Limon | 469 | descartar |
| 26 | Lay-s-Barbecue | 469 | descartar |
| 27 | Lay-s-Classic | 469 | descartar |
| 3 | Cheetos-Crunchy-Flamin-Hot | 462 | descartar |
| 11 | Coca-cola 20Oz | 441 | coca_botella |
| 54 | Vanilla-Coca-cola 20Oz | 420 | solo_detector |
| 52 | Sprite20Oz | 406 | sprite_botella |
| 39 | MtnDew-16Oz | 245 | solo_detector |
| 58 | zeroCocaCola-16Oz | 238 | solo_detector |
| 12 | CocaCola-16Oz | 238 | coca_lata |
| 53 | Sprite40Oz | 238 | sprite_botella |
| 8 | CherryVanilla-Coca-cola 20Oz | 231 | solo_detector |
| 14 | Crush-16Oz | 231 | solo_detector |
| 7 | Cherry-Coca-cola 20Oz | 231 | solo_detector |
| 29 | Lenny-Larry-s-BirthdayCake | 224 | descartar |
| 23 | Fanta20Oz | 224 | fanta_botella |
| 33 | Lenny-Larry-s-PeanutButter-ChocolateChips | 224 | descartar |
| 34 | Lenny-Larry-s-Snickerdoodle | 224 | descartar |
| 30 | Lenny-Larry-s-ChocolateChips | 224 | descartar |
| 32 | Lenny-Larry-s-PeanutButter | 217 | descartar |
| 37 | MinuteMaidBlueRaspberry-20Oz | 217 | solo_detector |
| 24 | FantaZero20Oz | 210 | solo_detector |
| 31 | Lenny-Larry-s-DoubleChocolateChips | 203 | descartar |
| 36 | MinuteMaid-PinkLemonade | 196 | solo_detector |
| 51 | Sprite-Zero-20Oz | 196 | solo_detector |
| 35 | MinuteMaid-FruitPunch | 189 | solo_detector |
| 38 | MinuteMaidLemonade-20Oz | 189 | solo_detector |
| 19 | Dr.Papper-1L | 182 | solo_detector |
| 50 | Sprite-TropicalMix-20Oz | 182 | solo_detector |
| 47 | Spiced-Coca-cola 20Oz | 175 | solo_detector |
| 0 | Barg-sBlack-20Oz | 175 | solo_detector |
| 49 | Sprite-Cherry-20Oz | 161 | solo_detector |
| 13 | Crunch | 161 | descartar |
| 10 | Chips-Ahoy-KingSize | 161 | descartar |
| 42 | Oreo-DoubleStuf-KingSize | 161 | descartar |
| 43 | Oreo-KingSize | 161 | descartar |
| 57 | skittles-smoothies-shareSize | 154 | descartar |
| 41 | Oreo | 154 | descartar |
| 9 | Chips-Ahoy | 147 | descartar |
| 22 | Fanta-Grape-20Oz | 147 | solo_detector |
| 1 | Bueno-shareSize | 126 | descartar |
| 40 | Nerds-shareSize | 126 | descartar |
| 55 | Whatchamacallit-kingSize | 126 | descartar |
| 45 | Skittles-shareSize | 126 | descartar |
| 46 | SourPunch-shareSize | 126 | descartar |
| 56 | dietCoke-Can16oz | 119 | solo_detector |
| 20 | Dr.Papper-Can16oz | 112 | solo_detector |
| 21 | Fanta-Can16oz | 112 | fanta_lata |
| 48 | Sprite-Can16oz | 112 | sprite_lata |
| 44 | PayDay-shareSize | 105 | descartar |

Clases que no aportan marca: 13335 cajas (88 %).

**Veredicto**: REGULAR. Versión en color (v1). Estudio: productos sobre un plato giratorio, fondo blanco, cajas completas en train. PERO las etiquetas de las carpetas valid/ y test/ de Roboflow están desplazadas (no coinciden con los productos): solo se usan las imágenes de train/ (7 copias aumentadas por foto). Las fotos se llaman '01', '02'... (el nombre no identifica la foto): se agrupan por pHash cercano. Única fuente de fanta_lata (pocas: ~120 fotos).

![muestras lamar_grocery_color](datos/muestras_lamar_grocery_color.jpg)

### `soda_bottles` — roboflow-100/soda-bottles

- **Licencia**: CC BY 4.0 · **versión** v4 (de 2) · uso: `entrenamiento` · tipo: `marcas`
- Preprocesado de Roboflow: remap; aumentos offline: ninguno
- Imágenes: **2249** · fotos originales: **2249** · copias aumentadas: 0 (2249 con sufijo `.rf.<hash>`) · grupos anti-fuga (fotos + duplicados/ráfagas): 2243
- Duplicados por pHash con otra foto: 12 · con otros datasets: 0 · imágenes sin cajas: 0
- Tamaño típico: 1920×1080 px · cajas: 54686 (24.3 por imagen)

| id | clase original | cajas | destino |
|---|---|---|---|
| 2 | sprite | 18747 | sprite_botella |
| 0 | coca-cola | 18513 | coca_botella |
| 1 | fanta | 17426 | fanta_botella |

**Veredicto**: REGULAR para nuestro caso. Cámara cenital en heladeras: las cajas son TAPAS de botellas vistas desde arriba (~25 por imagen). Útil para el detector (escenas densas); no para el multiclase ni el clasificador, porque una tapa no representa la vista frontal de la góndola y sus ~54 000 cajas aplastarían a las demás fuentes.

![muestras soda_bottles](datos/muestras_soda_bottles.jpg)

### `sku110k_sub` — ultralytics/SKU-110K

**No disponible**: no descargado.


Nota: PENDIENTE (opcional, 13,6 GB). Góndolas densas, 1 clase. Licencia de investigación (no comercial).

## Conteos finales

### detector_1clase

| split | imagenes | cajas | negativos |
|---|---|---|---|
| train | 19560 | 63297 | 1778 |
| val | 2526 | 8701 | 229 |
| test | 2504 | 8263 | 227 |

| clase | img train | cajas train | img val | cajas val | img test | cajas test |
|---|---|---|---|---|---|---|
| producto | 17782 | 63297 | 2297 | 8701 | 2277 | 8263 |

Por dataset de origen (imágenes):

| alias | train | val | test |
|---|---|---|---|
| darians_bottle | 14355 | 1827 | 1801 |
| lamar_grocery_color | 877 | 117 | 145 |
| rosd_final_hr | 2552 | 344 | 329 |
| soda_bottles | 1776 | 238 | 229 |

Verificación de fugas: **sin fugas** (ningún grupo ni pHash en dos splits).

### multiclase

| split | imagenes | cajas | negativos |
|---|---|---|---|
| train | 4471 | 5130 | 406 |
| val | 598 | 703 | 54 |
| test | 543 | 659 | 49 |

| clase | img train | cajas train | img val | cajas val | img test | cajas test |
|---|---|---|---|---|---|---|
| coca_botella | 1067 | 1108 | 152 | 156 | 137 | 146 |
| coca_lata | 937 | 948 | 108 | 110 | 117 | 118 |
| fanta_botella | 1013 | 1030 | 151 | 153 | 114 | 118 |
| fanta_lata | 31 | 31 | 4 | 4 | 4 | 4 |
| sprite_botella | 1074 | 1092 | 137 | 139 | 140 | 144 |
| sprite_lata | 179 | 181 | 27 | 27 | 24 | 24 |
| pepsi_botella | 241 | 266 | 35 | 40 | 26 | 30 |
| pepsi_lata | 217 | 223 | 30 | 31 | 36 | 39 |
| 7up_lata | 243 | 251 | 41 | 43 | 35 | 36 |

Por dataset de origen (imágenes):

| alias | train | val | test |
|---|---|---|---|
| darians_bottle | 3185 | 414 | 367 |
| lamar_grocery_color | 293 | 40 | 43 |
| rosd_final_hr | 993 | 144 | 133 |

Verificación de fugas: **sin fugas** (ningún grupo ni pHash en dos splits).

### recortes (clasificador)

Recortes guardados (clase × split):

| clase | train | val | test | total |
|---|---|---|---|---|
| coca_botella | 1070 | 150 | 141 | 1361 |
| coca_lata | 913 | 97 | 110 | 1120 |
| fanta_botella | 992 | 144 | 115 | 1251 |
| fanta_lata | 30 | 4 | 4 | 38 |
| sprite_botella | 1047 | 131 | 137 | 1315 |
| sprite_lata | 180 | 27 | 24 | 231 |
| pepsi_botella | 264 | 38 | 29 | 331 |
| pepsi_lata | 218 | 31 | 34 | 283 |
| 7up_lata | 244 | 39 | 36 | 319 |

Por dataset de origen:

| clase | darians_bottle | lamar_grocery_color | rosd_final_hr |
|---|---|---|---|
| coca_botella | 944 | 133 | 284 |
| coca_lata | 718 | 78 | 324 |
| fanta_botella | 870 | 65 | 316 |
| fanta_lata | 0 | 38 | 0 |
| sprite_botella | 875 | 185 | 255 |
| sprite_lata | 0 | 38 | 193 |
| pepsi_botella | 0 | 0 | 331 |
| pepsi_lata | 0 | 0 | 283 |
| 7up_lata | 0 | 0 | 319 |

Descartes por motivo:

| clase | borrosa | chica | duplicada | duplicada_entre_splits | duplicada_otra_clase |
|---|---|---|---|---|---|
| coca_botella | 4 | 5 | 5 | 3 | 0 |
| coca_lata | 0 | 0 | 2 | 3 | 0 |
| fanta_botella | 3 | 5 | 3 | 4 | 0 |
| fanta_lata | 0 | 0 | 0 | 0 | 0 |
| sprite_botella | 2 | 1 | 5 | 1 | 0 |
| sprite_lata | 0 | 0 | 0 | 0 | 0 |
| pepsi_botella | 0 | 0 | 2 | 2 | 1 |
| pepsi_lata | 1 | 0 | 5 | 4 | 0 |
| 7up_lata | 2 | 0 | 5 | 3 | 1 |

Nitidez (varianza del Laplaciano a 128 px de alto): percentiles 5/25/50 = 177 / 413 / 770.

Verificación de fugas: **sin fugas**.

![recortes](datos/recortes_muestras.jpg)

## Avisos

- multiclase: `fanta_lata` tiene 31 cajas en train (< 300)
- multiclase: `sprite_lata` tiene 181 cajas en train (< 300)
- multiclase: `pepsi_botella` tiene 266 cajas en train (< 300)
- multiclase: `pepsi_lata` tiene 223 cajas en train (< 300)
- multiclase: `7up_lata` tiene 251 cajas en train (< 300)
- recortes: `fanta_lata` tiene 30 recortes en train (< 300)
- recortes: `sprite_lata` tiene 180 recortes en train (< 300)
- recortes: `pepsi_botella` tiene 264 recortes en train (< 300)
- recortes: `pepsi_lata` tiene 218 recortes en train (< 300)
- recortes: `7up_lata` tiene 244 recortes en train (< 300)

## Conclusiones y propuestas

- **No hay test externo**: ningún dataset de marcas se puede reservar entero sin dejar clases en cero
  (rosd_final_hr es la única fuente de pepsi y 7up; darians_bottle aporta ~80 % de coca_lata).
  El test se arma con grupos (fotos originales) de todos los datasets. La prueba de dominio real
  será la maqueta.
- **Grupos anti-fuga**: además del id original de Roboflow, se unen las imágenes con pHash idéntico.
  En darians_bottle eso junta ráfagas de hasta ~240 fotos de la misma escena (mismo split).
  lamar_grocery nombra sus fotos '01', '02'... (el nombre no identifica la foto): esos ids se
  subdividen por pHash cercano (Hamming ≤ 10).
- **lamar_grocery**: las etiquetas de sus carpetas valid/ y test/ están desplazadas respecto de los
  productos (verificado a ojo en v1 y v9); solo se usa train/ de la v1 (color).
- **fanta_lata casi no tiene datos** (~31 cajas / 30 recortes en train, todos de lamar) y sprite_lata,
  pepsi_botella, pepsi_lata y 7up_lata quedan por debajo de 300. Paliativo inmediato: correr
  `datos_recortes.py --copias-train 3` (usa más copias aumentadas de Roboflow, sin fugas) o
  ponderar clases al entrenar. Propuesta de fondo, en orden:
  1. Agregar `cold-drinks-cftkw/cold-drinks-qki32` (CC BY 4.0, v1, 1950 imágenes, ya usado en el
     proyecto clasificador-bebidas): 'fanta can' 184, 'sprite can' 128, '7up can' 179, 'pepsi can' 126,
     'pepsi bottel' 177, 'coca cola can' 156 cajas.
  2. Agregar `again/aug-tygyc` (CC BY 4.0, v5): 'fanta lon 330' 200, 'sprite lon 330' 185,
     'pepsi lon 330' 142, '7up lon 330' 114, 'pepsi_bottle' 1064 cajas ('lon' = lata, 'chai' = botella).
  3. Sintéticos (notebook 03): recortes de latas recortadas con rembg pegados sobre fondos de góndola,
     sobre todo para fanta_lata y sprite_lata.
  4. Fotos propias de la maqueta para las latas que siguen cortas.
  Agregar un dataset es una entrada nueva en `src/datos/fuentes.py` + correr el pipeline.

## Pendientes

- SKU-110K (13,6 GB): opcional, no descargado. `scripts/datos_sku110k.py --max-imagenes 1500 --confirmar`
  extrae un subconjunto en streaming (igual baja los 13,6 GB) y lo suma al detector_1clase.
- bottle_can_cup y tusafe_bottle_can: requieren fork al workspace propio + generar versión en Roboflow.
- El remapeo (`config/remapeo_clases.yaml`) es un BORRADOR a aprobar.
