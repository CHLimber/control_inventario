# Guía de construcción de la maqueta

No tenemos acceso a un estante real de tienda, así que armamos una **maqueta de estantería** con
envases reales y la fotografiamos con el celular. Con esas fotos hacemos tres cosas:

1. **ajuste fino** del detector y del clasificador con imágenes parecidas a las de uso real;
2. **validación/prueba** del conteo contra una verdad de terreno conocida (`datos/maqueta/escenas.csv`);
3. **la demo** final (ver [DEMO.md](DEMO.md)).

Documentos relacionados: [PROTOCOLO.md](PROTOCOLO.md) (cómo fotografiar) ·
[plan de la sesión 1](../datos/maqueta/escenas_plan_sesion01.md) (qué armar en cada escena).

Tiempo estimado: **una tarde** de construcción (3-4 h) + juntar envases durante la semana.

---

## 1. Diseño

### Supuestos del sistema (importantes)

- **Una sola fila por nivel.** El estante tiene 10-15 cm de profundidad: entra un solo envase en
  fondo. El sistema cuenta lo que ve de frente; no hay productos escondidos detrás de otros. Esto es
  un **supuesto documentado** del sistema y la maqueta lo respeta a propósito.
- **3 niveles, numerados de ABAJO hacia ARRIBA** (así los usan los scripts y el plan):

| Nivel | Contenido | Alto libre (aprox.) |
|---|---|---|
| 1 (abajo) | botellas grandes (2 L) | ≈ 35 cm |
| 2 (medio) | botellas chicas (500-600 ml) | ≈ 25 cm |
| 3 (arriba) | latas (350 ml) | ≈ 15 cm |

- **Tira portaprecios** al frente de cada nivel (como en las tiendas): aporta realismo y a veces
  tapa la base de los envases, que es justo lo que pasa en la vida real.
- **Fondos intercambiables** (2-3): blanco, madera y negro/gris, para que el modelo no aprenda
  "el fondo de mi casa".

> **Medí tus envases antes de cortar.** Las alturas de arriba son orientativas. Regla:
> *alto libre del nivel = envase más alto de ese nivel + 3-4 cm* (para meter la mano sin voltear
> nada). Valores típicos: botella 2 L ≈ 31-33 cm de alto y ≈ 10.5 cm de diámetro; botella 3 L
> ≈ 36-37 cm (¡no entra en 35!); botella 500-600 ml ≈ 21-24 cm y ≈ 6.5-7.5 cm; lata 350 ml
> ≈ 12.2 cm y 6.6 cm. Anotá tus medidas reales en la tabla de la sección 6.

### Capacidad (cuántos envases entran en fila)

Con 70 cm de ancho interior, 1 cm de margen a cada lado y 0.5 cm entre envases:

| Nivel | Diámetro | Entran |
|---|---|---|
| 1 · botellas grandes | 10.5 cm | 6 |
| 2 · botellas chicas | 7.0 cm | 9 |
| 3 · latas | 6.6 cm | 9 |

`scripts/maqueta_plan.py` hace esta cuenta con `--ancho` (60-80 cm). Si tu estante mide distinto,
pasale el ancho real y el plan respeta la capacidad.

### Diagrama

Vista de frente (sin fondo puesto), ancho interior 70 cm:

```
          |<-------------------- 70 cm -------------------->|
          +==================================================+  <- techo (opcional)
          |  [L] [L] [L] [L] [L] [L] [L] [L] [L]              |
 nivel 3  |  latas (hasta 9)                                  |  ≈15 cm
          |__________________________________________________|
          |  Bs 7   Bs 7   Bs 8   Bs 7   Bs 8   Bs 7          |  <- tira portaprecios (2-2.5 cm)
          +==================================================+
          |   _    _    _    _    _    _    _    _    _       |
          |  | |  | |  | |  | |  | |  | |  | |  | |  | |      |
 nivel 2  |  |B|  |B|  |B|  |B|  |B|  |B|  |B|  |B|  |B|      |  ≈25 cm
          |  |_|  |_|  |_|  |_|  |_|  |_|  |_|  |_|  |_|      |
          |__________________________________________________|
          |  Bs 5   Bs 5   Bs 6   Bs 5   Bs 6                 |
          +==================================================+
          |    __      __      __      __      __      __    |
          |   |  |    |  |    |  |    |  |    |  |    |  |   |
          |   |  |    |  |    |  |    |  |    |  |    |  |   |
 nivel 1  |   |BG|    |BG|    |BG|    |BG|    |BG|    |BG|   |  ≈35 cm
          |   |  |    |  |    |  |    |  |    |  |    |  |   |
          |   |__|    |__|    |__|    |__|    |__|    |__|   |
          |__________________________________________________|
          |  Bs 15  Bs 15  Bs 16  Bs 15  Bs 16  Bs 15         |
          +==================================================+  <- base
           lateral                                    lateral
```

Vista de costado (un nivel):

```
      fondo intercambiable (cartulina / plancha, se apoya atrás)
      |
      |   envase        tira portaprecios
      |   ____          |
      |  |    |         v
      |  |    |        +--+
      |  |    |        |Bs|   (2-2.5 cm de alto, pegada al canto de la repisa)
  ====+==+====+========+--+
      |<---- 12 cm ---->|
```

Altura total ≈ 35 + 25 + 15 + espesores (4 repisas × ~0.5-1.5 cm) ≈ **80-82 cm**.

---

## 2. Materiales y costo (Bs, precios aproximados en Santa Cruz)

Dos opciones. La **económica** sirve perfectamente para la práctica.

| Material | Cantidad | Económica (cartón) | Mejor (MDF 3-5 mm) |
|---|---|---|---|
| Cartón prensado / cajas de cartón grueso (de electrodomésticos, del mercado) | 3-4 cajas o 1 plancha | 0-20 | — |
| MDF 3-5 mm cortado en barraca (2 laterales 12×82, 4 repisas 70×12, 1 trasera 72×82) | 1 juego | — | 90-150 |
| Listón 2×2 cm para refuerzo de repisas (opcional) | 2 m | — | 15-25 |
| Cinta de embalaje ancha | 1 rollo | 8-12 | 8-12 |
| Cinta doble faz / masking tape | 1 c/u | 10-15 | 10-15 |
| Silicona fría o cola blanca (o clavitos/tornillos para MDF) | 1 | 10-15 | 10-20 |
| Escuadras metálicas chicas (MDF) | 8 | — | 15-25 |
| Cartulina blanca y negra (fondos + tiras de precio) | 2 + 2 | 6-10 | 6-10 |
| Papel contact imitación madera (fondo "madera") o un tablero/mantel de madera | 1 m | 15-20 | 15-20 |
| Colorantes vegetales (caramelo/negro y naranja) | 2 | 6-10 | 6-10 |
| Café instantáneo (alternativa al colorante oscuro) | 1 sobre | 2-5 | 2-5 |
| Etiquetas / stickers para precios (o imprimir) | 1 hoja | 2-5 | 2-5 |
| Cúter, regla metálica, lápiz (si no tenés) | — | 15-25 | 15-25 |
| Foco de luz cálida (si en casa todo es luz fría) | 1 | 15-25 | 15-25 |
| Envases vacíos (pedidos en casa, a vecinos o a una tienda) | 4-6 por clase | 0 | 0 |
| **Total aproximado** | | **≈ Bs 90-160** | **≈ Bs 200-320** |

> Si ya tenés cinta, cúter, regla y luz cálida en casa, la versión de cartón sale **≈ Bs 50-80**.
> Alternativa rápida: un mueble/repisa que ya tengas de 60-80 cm, recortando la profundidad
> con una tabla o caja de 12 cm al frente para obligar a una sola fila.

### Envases (lo más importante)

- **4-6 por clase**, 8 clases ⇒ unos 35-45 envases. Los scripts suponen **5 por clase** y, de las
  botellas, **2 grandes (2 L) + 3 chicas (500-600 ml)** por marca. Si tenés otra cantidad, pasala al
  plan con `--envases` y `--grandes` (ver sección 5).
- Clases: `coca_botella`, `coca_lata`, `fanta_botella`, `fanta_lata`, `sprite_botella`,
  `sprite_lata`, `pepsi_botella`, `pepsi_lata`.
- Etiquetas **en buen estado**, con su **tapa original** (el color de la tapa también es una pista
  para el modelo). Varias versiones de etiqueta de la misma marca es bueno: más variedad real.
- **Relleno con agua coloreada** (que se vean como llenas):

| Marca | Relleno |
|---|---|
| Coca-Cola, Pepsi | agua + colorante caramelo/negro, o café instantáneo bien cargado (oscuro, casi negro) |
| Fanta | agua + colorante naranja |
| Sprite | agua sola (transparente) |

  Llená hasta el nivel normal del producto (unos 2-3 cm debajo de la tapa) y cerrá bien. Si usás
  café, cambialo cada 1-2 semanas (fermenta). Unas gotas de lavandina evitan hongos.
- **Latas:** no se pueden rellenar y cerrar. Enjuagalas, poneles un poco de arena o agua para que
  pesen y no se volteen, y fotografiá a la altura del estante (desde arriba se ve la abertura).
  Si alguna está abollada, que quede de espaldas o no la uses.

---

## 3. Pasos de construcción

1. **Medí los envases** (alto y diámetro del más alto de cada tipo) y anotá en la tabla de la
   sección 6. Ajustá las alturas de los niveles (envase más alto + 3-4 cm).
2. **Cortá las piezas** (con cartón, doble capa con la onda cruzada para que no se doble):
   - 2 laterales: 12 cm × ≈ 82 cm;
   - 4 repisas (base, dos intermedias y techo): 70 cm × 12 cm (+ el espesor de los laterales
     si van por fuera);
   - 1 trasera: ≈ 72 × 82 cm (opcional con cartón: los fondos se apoyan directo).
3. **Marcá en los laterales** las alturas de las repisas desde abajo: 0, 35, 35+25 = 60,
   60+15 = 75 cm (más los espesores). Usá la escuadra: si las repisas quedan torcidas, los
   envases se inclinan y la foto se ve rara.
4. **Armá**: pegá/atornillá las repisas a los laterales. Con cartón, reforzá cada unión con cinta
   de embalaje por arriba y por abajo, y poné un listón o una tira de cartón doblada en "L" debajo
   de cada repisa para que no se combe con las botellas de 2 L (≈ 2 kg cada una).
5. **Tiras portaprecios**: tiras de cartulina blanca de 2-2.5 cm de alto × 70 cm, pegadas al canto
   frontal de cada repisa. Escribí o imprimí precios ficticios (p. ej. latas Bs 7-8, chicas Bs 5-6,
   2 L Bs 14-16). **No más de 2.5 cm de alto**: si tapa media botella, el modelo no la ve.
6. **Fondos intercambiables**: tres planchas del tamaño de la trasera (≈ 72 × 82 cm):
   - **blanco**: cartulina blanca mate;
   - **madera**: papel contact imitación madera sobre cartón (o una tabla real);
   - **negro/gris**: cartulina negra o gris oscuro mate.
   Se apoyan contra la trasera y se sujetan con 2 clips o masking tape para cambiarlos en segundos.
   Mate, no brillante (lo brillante refleja la luz).
7. **Estabilidad**: el estante lleno pesa > 10 kg. Apoyalo contra la pared o sujetalo con una cinta
   para que no se vuelque hacia adelante. Probalo lleno ANTES de la primera sesión.
8. **Lugar**: cerca de una ventana (para la luz natural) y con un enchufe cerca (lámpara cálida).
   Marcá en el piso con masking tape dónde va el trípode/celular para las tomas frontal y ±20°
   (ver PROTOCOLO, sección 3).

---

## 4. Consejos para las fotos

- **Reflejos**: las botellas PET y las latas reflejan. Luz **de costado (≈45°)**, nunca detrás del
  celular apuntando al estante; difuminá la lámpara con una hoja bond o papel manteca. Que no se
  vea la ventana reflejada en las latas. El flash solo en las escenas "flash" del plan.
- **Altura**: celular a la **altura del nivel 2** (≈ 45-50 cm del piso de la maqueta), apuntando
  derecho, sin inclinar hacia abajo.
- **Encuadre**: el estante completo con un poco de margen, en **vertical** (el estante es más alto
  que ancho). A ≈ 0.8-1.2 m con la cámara principal (1×). **Sin zoom digital**, sin modo retrato,
  sin filtros, sin "embellecer". HDR automático está bien.
- **Foco**: tocá la pantalla sobre el nivel 2 para enfocar; esperá a que enfoque antes de disparar.
  Si la foto sale movida, repetila **y borrá la mala** (cada escena tiene exactamente 3 fotos).
- **Formato JPG**: en iPhone, *Ajustes > Cámara > Formatos > Más compatible* (si no, salen HEIC).
- **Fecha y hora** del celular correctas: el script de ingesta ordena por la fecha EXIF.
- **Etiquetas de frente** salvo cuando el plan pide girar alguna; envases derechos, sin tocarse
  (≈ 0.5 cm entre sí), alineados al frente de la repisa.
- **Nada extra** en el cuadro: ni manos, ni otros productos, ni la hoja del plan.

---

## 5. Plan concreto de la sesión 1

Ya está generado (26 escenas, ≈ 78 fotos, unas 2 horas):

- CSV: [`datos/maqueta/escenas_plan.csv`](../datos/maqueta/escenas_plan.csv)
- **Para imprimir**: [`datos/maqueta/escenas_plan_sesion01.md`](../datos/maqueta/escenas_plan_sesion01.md)
  (abrilo en VS Code → vista previa → imprimir, o pegalo en un documento).

Supone ancho 70 cm y 5 envases por clase (2 grandes + 3 chicas por marca de botella). **Si tu
maqueta es distinta, regeneralo antes de empezar**, por ejemplo:

```bat
conda activate inventariovision
python scripts\maqueta_plan.py --sesiones 1 --escenas 26 --ancho 65 --envases 4,pepsi_lata=6 --grandes 1
```

El plan dice, escena por escena, qué marca va en cada lugar de cada nivel (izquierda → derecha),
con qué luz y fondo, y qué variaciones hacer (girar un envase, tapar media etiqueta). Incluye
escenas llenas, con un nivel medio vacío, con una marca entera en cero y **acercamientos**
(se encuadra un solo nivel sin mover nada). Las sesiones 2 y 3 se generan igual
(`--sesiones 2` / `--sesiones 3`) cuando llegue el momento; la 3 es **solo para prueba**
(ver PROTOCOLO).

---

## 6. Mis medidas (completar)

| | Alto (cm) | Diámetro (cm) | Cantidad |
|---|---|---|---|
| Botella grande (2 L) | | | |
| Botella chica (500-600 ml) | | | |
| Lata 350 ml | | | |
| Ancho interior del estante | | — | — |
| Alto libre nivel 1 / 2 / 3 | | — | — |
