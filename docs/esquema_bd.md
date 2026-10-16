# Esquema de la base de datos

Contrato fijado en la Fase 0. El código está en `contratos/modelos_bd.py` (SQLAlchemy 2.0). SQLite por defecto (`URL_BD=sqlite:///datos/inventario.db`). Para pasar a MySQL alcanza con cambiar `URL_BD`, porque solo se usan tipos estándar y los "enum" son `String` con `CHECK`.

```
producto 1───* movimiento *───0..1 cierre
   │                                │
   └──────* cierre_detalle *────────┤
   └──0..1* deteccion *─────────────┘
```

## Tablas

### `producto`
| Campo | Tipo | Notas |
|---|---|---|
| id | int PK | |
| nombre | varchar(100) | "Coca-Cola lata" |
| marca | varchar(50) | |
| formato | varchar(20) | `botella` o `lata` |
| precio | decimal(10,2) | |
| clase_modelo | varchar(50) **único** | clave en `config/clases.json`; une la BD con el modelo |
| activo | bool | baja lógica |

### `movimiento`
| Campo | Tipo | Notas |
|---|---|---|
| id | int PK | |
| id_producto | FK producto | |
| tipo | varchar(20) | `stock_inicial`, `entrada`, `venta`, `ajuste` |
| cantidad | int | ≥ 0; solo `ajuste` lleva signo |
| fecha | datetime | |
| usuario | varchar(50) | |
| id_cierre | FK cierre, nulo | se llena en los ajustes generados al confirmar un cierre |

### `cierre`
| Campo | Tipo | Notas |
|---|---|---|
| id | int PK | |
| fecha | datetime | momento del escaneo |
| usuario | varchar(50) | |
| imagen_path | varchar(255) | imagen anotada, relativa a `datos/escaneos/` (las originales, en la misma carpeta `cierre_<id>/`) |
| estado | varchar(20) | `pendiente` o `confirmado` |
| variante_modelo | varchar(20) | `mock`, `A` o `B` |
| observacion | text | |

### `cierre_detalle` (PK: id_cierre + id_producto)
| Campo | Tipo | Notas |
|---|---|---|
| esperado | int | stock según la BD al momento del escaneo |
| detectado | int | lo que contó el modelo |
| ajustado | int, nulo | lo que confirmó el cajero |
| diferencia | int | `(ajustado ?? detectado) − esperado`. Negativa = **faltante**, positiva = **sobrante** |
| confianza_media | float, nulo | promedio de confianza de las cajas de esa clase |

### `deteccion`
Una fila por caja: `id_cierre`, `id_producto` (nulo = `desconocido`), `indice_imagen`, `x, y, w, h` en píxeles (esquina superior izquierda), `conf_detector` y `conf_clasificador` (nulo en la variante B).

## Reglas de negocio

**Stock esperado** de un producto a una fecha `t`:

```
base   = último movimiento stock_inicial con fecha ≤ t   (0 si no hay)
stock  = base.cantidad
       + Σ entradas  (fecha en (base.fecha, t])
       − Σ ventas    (ídem)
       + Σ ajustes   (ídem, con su signo)
```

Un `stock_inicial` nuevo "reinicia" el stock: sirve para la demo, donde se carga lo que hay físicamente en la maqueta.

**Conciliación (cierre):**
1. `POST /api/escaneos` guarda las fotos en `datos/escaneos/cierre_<id>/`, llama a `contar()`, calcula el esperado de cada producto activo y crea el cierre `pendiente` con su detalle y sus detecciones.
2. `POST /api/cierres/{id}/confirmar` guarda el `ajustado`, recalcula la `diferencia` y, por cada producto con diferencia ≠ 0, crea un movimiento `ajuste` con esa cantidad y el `id_cierre`. Así el esperado del día siguiente parte del conteo real y el faltante queda registrado (es lo que consulta el RAG). Un cierre confirmado no se puede volver a confirmar.
3. Estado de cada fila: `ok` si la diferencia es 0, `faltante` si es < 0 y `sobrante` si es > 0.
