"""Catálogo de datasets públicos usados por el pipeline de datos.

- tipo:    "generico" (envases sin marca) o "marcas".
- uso:     "entrenamiento" (se reparte en train/val/test agrupando por foto original) o
           "test_externo" (dataset COMPLETO reservado para test: mide la generalización a otra fuente).
- usar_en: en qué salidas entra: "detector" (detector_1clase), "multiclase", "recortes".
           Vacío = se descarga y se reporta, pero no se usa.
- splits_origen: (opcional) solo se usan las imágenes de estas carpetas del export de Roboflow.
- version: None = la última disponible (se resuelve con la API al descargar).
- nota:    veredicto de calidad (sale en docs/datos.md).
La licencia y la versión efectiva quedan registradas en DIR_DATOS/raw/<alias>/_fuente.json.
"""
from __future__ import annotations

TODO = ("detector", "multiclase", "recortes")

FUENTES: list[dict] = [
    # --- Detector genérico ---
    {"alias": "bottle_can_cup", "workspace": "ai-k6ynk", "project": "bottle-can-cup-dataset-pvesi",
     "tipo": "generico", "uso": "entrenamiento", "version": None, "usar_en": ("detector",),
     "nota": "Licencia declarada CC BY 4.0. Sin versiones publicadas en Universe (15 000 imágenes, clases Bottle/Can/Cup): la API no "
             "permite exportarlo. Habría que hacer un fork al workspace propio y generar una versión."},
    {"alias": "tusafe_bottle_can", "workspace": "tusafes-workspace", "project": "bottle-can-data-set",
     "tipo": "generico", "uso": "entrenamiento", "version": None, "usar_en": ("detector",),
     "nota": "Licencia declarada CC BY 4.0. Sin versiones publicadas en Universe (13 800 imágenes, clases Bottle/Can): no exportable. "
             "Misma solución que bottle_can_cup (fork + versión)."},
    # --- Marcas ---
    {"alias": "darians_bottle", "workspace": "darians-workspace-4fake", "project": "bottle-detection-q9eas",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": TODO,
     "nota": "BUENO. Productos mexicanos, casi siempre un envase acostado sobre una mesa, cajas ajustadas al "
             "envase entero. Letterbox 416 px y 3 copias aumentadas por foto en train (rotaciones de 90°, "
             "flips, recortes): se usa 1 copia por foto. Variantes (Coca Light, Fanta Berry) y otras marcas "
             "-> solo_detector. 'Cans' son latas Arizona y 'Bottle-Detection' botellas de agua."},
    {"alias": "rosd_final_hr", "workspace": "rosd", "project": "final_hr",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": TODO,
     "nota": "BUENO y CLAVE: única fuente de pepsi_botella, pepsi_lata y 7up_lata. Envases chinos (logos en "
             "chino para Coca y Sprite), fotos de mano/mesa, cajas completas, sin aumentos. Mirinda (lata "
             "naranja, parecida a Fanta) queda como solo_detector."},
    {"alias": "coca_raspberry", "workspace": "new-workspace-ihwlk", "project": "coca-cola-raspberry",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": TODO,
     "nota": "Licencia declarada CC BY 4.0. No disponible: el proyecto está vacío (0 imágenes, sin versiones)."},
    {"alias": "lamar_grocery", "workspace": "lamar-university-venef", "project": "grocery-rfn8l",
     "tipo": "generico", "uso": "entrenamiento", "version": None, "usar_en": (),
     "nota": "NO SE USA. La última versión (v9) viene en ESCALA DE GRISES (preprocesado grayscale + "
             "contraste), así que no sirve para marcas, y comparte fotos con la v1 (lamar_grocery_color). "
             "Igual que en la v1, las etiquetas de valid/test están DESPLAZADAS respecto de los productos. "
             "Más de la mitad de las clases son snacks (descartar)."},
    {"alias": "lamar_grocery_color", "workspace": "lamar-university-venef", "project": "grocery-rfn8l",
     "tipo": "marcas", "uso": "entrenamiento", "version": 1, "usar_en": TODO, "splits_origen": ("train",),
     "nota": "REGULAR. Versión en color (v1). Estudio: productos sobre un plato giratorio, fondo blanco, "
             "cajas completas en train. PERO las etiquetas de las carpetas valid/ y test/ de Roboflow están "
             "desplazadas (no coinciden con los productos): solo se usan las imágenes de train/ "
             "(7 copias aumentadas por foto). Las fotos se llaman '01', '02'... (el nombre no identifica "
             "la foto): se agrupan por pHash cercano. Única fuente de fanta_lata (pocas: ~120 fotos)."},
    {"alias": "soda_bottles", "workspace": "roboflow-100", "project": "soda-bottles",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": ("detector",),
     "nota": "REGULAR para nuestro caso. Cámara cenital en heladeras: las cajas son TAPAS de botellas vistas "
             "desde arriba (~25 por imagen). Útil para el detector (escenas densas); no para el multiclase "
             "ni el clasificador, porque una tapa no representa la vista frontal de la góndola y sus "
             "~54 000 cajas aplastarían a las demás fuentes."},
    # --- Opcional, fuera de Roboflow: lo arma scripts/datos_sku110k.py (no lo baja datos_descargar.py) ---
    {"alias": "sku110k_sub", "workspace": "ultralytics", "project": "SKU-110K", "origen": "sku110k",
     "tipo": "generico", "uso": "entrenamiento", "version": None, "usar_en": ("detector",),
     "nota": "PENDIENTE (opcional, 13,6 GB). Góndolas densas, 1 clase. Licencia de investigación (no comercial)."},
]

# Conclusiones y pendientes que se agregan al final de docs/datos.md (se escriben a mano).
NOTAS_REPORTE = """\
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
"""


def fuente(alias: str) -> dict:
    for f in FUENTES:
        if f["alias"] == alias:
            return f
    raise KeyError(f"Dataset desconocido: {alias}")
