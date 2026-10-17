"""Catálogo de datasets públicos usados por el pipeline de datos.

- tipo:    "generico" (envases sin marca) o "marcas".
- uso:     "entrenamiento" (se reparte en train/val/test agrupando por foto original) o
           "test_externo" (dataset COMPLETO reservado para test: mide la generalización a otra fuente).
- usar_en: en qué salidas entra: "detector" (detector_1clase), "multiclase", "recortes".
           Vacío = se descarga y se reporta, pero no se usa.
- splits_origen: (opcional) solo se usan las imágenes de estas carpetas del export de Roboflow.
- version: None = la última disponible (se resuelve con la API al descargar).
- biblioteca: (opcional) True = sus imágenes de train alimentan la biblioteca de envases de los sintéticos
           (scripts/datos_biblioteca.py) aunque no entren al multiclase.
- max_recortes_por_clase: (opcional) tope de recortes por clase en train para datasets muy grandes
           (en val/test el 12 %): evita que una sola fuente aplaste a las demás.
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
     "nota": "BUENO y CLAVE: única fuente de pepsi_botella y pepsi_lata (7up queda como solo_detector). Envases chinos (logos en "
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
    # --- Agregadas en la 2.ª ronda (aprobadas por el estudiante) para las latas y Pepsi ---
    {"alias": "cold_drinks", "workspace": "cold-drinks-cftkw", "project": "cold-drinks-qki32",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": TODO,
     "nota": "BUENO para marcas. Imágenes de catálogo/web (envase sobre fondo blanco, pocas fotos reales), "
             "cajas completas, botellas y latas separadas. Trae variantes rotuladas como la marca "
             "(Fanta frutilla/uva, Sprite Zero, Pepsi Diet): el control de color filtra las "
             "más obvias. Brecha de dominio con la góndola real."},
    {"alias": "again_aug", "workspace": "again", "project": "aug-tygyc",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": ("recortes",),
     "nota": "REGULAR. Góndolas reales de supermercado (muy cerca de nuestro caso), pero con ETIQUETADO "
             "INCOMPLETO: muchas botellas y todas las latas visibles quedan sin caja, así que no sirve para "
             "los datasets YOLO (enseñaría que un producto es fondo). La v5 solo trae 11 clases sin formato "
             "(las 77 del proyecto con 'lon'/'chai' no están en ninguna versión); son botellas -> *_botella. "
             "Aumentos fuertes (mosaicos, grises, ruido): el control de color descarta los recortes en grises. "
             "Solo se usa para los recortes del clasificador."},
    # --- Agregadas en la 3.ª ronda (2026-10-17): más datos reales de góndola para entrenar en Colab ---
    {"alias": "cold_drinks_2", "workspace": "cold-drinks", "project": "2-4uv7p",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": ("detector", "recortes"),
     "max_recortes_por_clase": 1500, "biblioteca": True,
     "nota": "BUENO y CLAVE para el detector. Fotos reales de góndolas y heladeras con filas densas de envases "
             "de frente, ~10 000 imágenes sin aumentos y ~157 000 cajas (unas 16 por foto). La v7 fusionó las "
             "131 clases del proyecto en 5 marcas (Coca Cola, Fanta, Pepsi, Sprite, Mountain Dew): se perdió el "
             "formato ('chai' = botella, 'lon' = lata en el proyecto original). Casi no hay latas: revisado a ojo, todo "
             "lo que la forma de la caja daba como lata eran botellas de 2 L tapadas o cortadas. Por eso el "
             "destino es 'auto:<marca>:botella' (solo botellas, por la forma de la caja). No entra al multiclase: "
             "las demás cajas quedarían sin etiquetar. Para los recortes se topa a 1 500 por clase y 2 por imagen. "
             "Alguna foto es de banco de imágenes (marca de agua). Envases asiáticos. Solo ~430 fotos originales "
             "(cada una viene ~23 veces con variaciones leves)."},
    {"alias": "hccb_overall", "workspace": "coca-cola", "project": "hccb-overall",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": ("detector", "recortes"),
     "biblioteca": True,
     "nota": "BUENO. Góndolas reales de retail (HCCB, India), 400 imágenes con cajas por SKU: formato en el "
             "nombre ('Can' / 'PET'). Aporta latas de Coca-Cola y Sprite. Envases de la India. No entra al multiclase: "
             "la mitad de las cajas son marcas que no son nuestras (Thums Up, Limca, Kinley) y quedarían sin etiquetar."},
    # --- Latas (3.ª ronda): latas de verdad, pero en escenas sueltas (no góndolas): solo recortes ---
    {"alias": "smart_vending", "workspace": "tuduchiep", "project": "smart-vending-4classes",
     "tipo": "marcas", "uso": "entrenamiento", "version": None, "usar_en": ("recortes",),
     "max_recortes_por_clase": 1200,
     "nota": "REGULAR. Latas de Coca-Cola y Pepsi (más de 1 000 cajas cada una), fotos de una lata en la mano o sobre una "
             "mesa en interiores. No son góndolas, así que solo se usa para los recortes del clasificador (topado a "
             "1 200 por clase). Se cuelan botellas de 2 L rotuladas como lata y muchas fotos son de la tapa: por eso "
             "el destino es 'auto:<marca>:lata' (solo latas de frente por la forma de la caja). 'water' y 'milk_box' no se usan. (sodacans, de it-gymnasiet-gteborg/trash-pyv9e, se "
             "descartó: su v4 fusionó todas las marcas en una sola clase 'can'.)"},
    # --- Opcional, fuera de Roboflow: lo arma scripts/datos_sku110k.py (no lo baja datos_descargar.py) ---
    {"alias": "sku110k_sub", "workspace": "ultralytics", "project": "SKU-110K", "origen": "sku110k",
     "tipo": "generico", "uso": "entrenamiento", "version": None, "usar_en": ("detector",),
     "nota": "PENDIENTE (opcional, 13,6 GB). Góndolas densas, 1 clase. Licencia de investigación (no comercial)."},
]

# Conclusiones y pendientes que se agregan al final de docs/datos.md (se escriben a mano).
NOTAS_REPORTE = """\
## Conclusiones y propuestas

- **No hay test externo**: ningún dataset de marcas se puede reservar entero sin dejar clases en cero
  (rosd_final_hr es la única fuente de pepsi; darians_bottle aporta ~80 % de coca_lata).
  El test se arma con grupos (fotos originales) de todos los datasets. La prueba de dominio real
  será la maqueta.
- **Grupos anti-fuga**: además del id original de Roboflow, se unen las imágenes con pHash idéntico.
  En darians_bottle eso junta ráfagas de hasta ~240 fotos de la misma escena (mismo split).
  lamar_grocery nombra sus fotos '01', '02'... (el nombre no identifica la foto): esos ids se
  subdividen por pHash cercano (Hamming ≤ 10).
- **lamar_grocery**: las etiquetas de sus carpetas valid/ y test/ están desplazadas respecto de los
  productos (verificado a ojo en v1 y v9); solo se usa train/ de la v1 (color).
- **Segunda ronda** (aprobada): se sumaron cold_drinks (todas las salidas) y again_aug (solo recortes:
  góndolas reales pero con etiquetado incompleto) y un **control de color por clase** que descarta
  cajas con etiqueta equivocada (ver la sección de control de color). Los splits de la primera ronda
  se mantuvieron: 0 imágenes cambiaron de split; solo se sortearon las nuevas.
- **Tercera ronda** (2026-10-17, propuesta pendiente de aprobación): se quitó la clase `7up_lata` (quedan 8 clases;
  las cajas de 7Up pasan a `solo_detector`) y se sumaron `cold_drinks_2` (~10 000 fotos de góndola, ~157 000
  cajas) y `hccb_overall` (400 góndolas de retail). Ninguna entra al multiclase (tienen cajas sin etiquetar o de
  marcas que no son nuestras); sí al detector de 1 clase y a los recortes. La v7 de `cold_drinks_2` solo trae
  la marca, así que el formato sale de la forma de la caja (`auto:<marca>:botella` en `config/remapeo_clases.yaml`,
  umbrales en `src/datos/remapeo.py`; solo botellas: lo que medía como lata eran botellas tapadas, verificado a
  ojo) y sus recortes se topan a 1 500 por clase.
- Con eso todas las clases superan 300 recortes en train. En el multiclase **fanta_lata sigue corta**
  (cajas en train < 300; las fuentes de latas de Fanta son lamar y cold_drinks). Propuesta: sintéticos
  de fanta_lata (notebook 03) y fotos propias de la maqueta; como paliativo, ponderar la clase.

## Pendientes

- SKU-110K (13,6 GB): opcional, no descargado. `scripts/datos_sku110k.py --max-imagenes 1500 --confirmar`
  extrae un subconjunto en streaming (igual baja los 13,6 GB) y lo suma al detector_1clase.
- bottle_can_cup y tusafe_bottle_can: requieren fork al workspace propio + generar versión en Roboflow.
- Remapeo aprobado en la 1.ª ronda; las entradas de cold_drinks, again_aug, cold_drinks_2 y hccb_overall están
  pendientes de aprobación (en particular la regla `auto:` de formato por la forma de la caja).
- Revisar a ojo los recortes que salen de `cold_drinks_2` (botella vs. lata por forma): una botella con el
  cuello tapado puede parecer lata.
"""


def fuente(alias: str) -> dict:
    for f in FUENTES:
        if f["alias"] == alias:
            return f
    raise KeyError(f"Dataset desconocido: {alias}")
