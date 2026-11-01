# Demostración (video)

## Preparación (una vez, antes de grabar)

```
conda activate inventariovision
python scripts/demo_poblar.py          # BORRA la BD (antes la respalda en datos/respaldos/) y la llena
scripts\iniciar_servidor.bat
```

Correr `demo_poblar.py` **con el servidor cerrado** y **el mismo día de la grabación**: los movimientos de «hoy» y
el escaneo en vivo se arman con la fecha del día en que se corre.

- `.env` con `VARIANTE_VISION=B`: el escáner cuenta con el YOLO multiclase. Con `A` usa el detector de 1 clase
  más MobileNetV2; con `mock`, conteos falsos.
- `demo_poblar.py` deja 14 cierres confirmados (unas dos semanas, sin domingos) y **hoy sin cierre**. Los primeros
  7 cierres los contó la variante A (el cajero corrigió varias veces, con la corrección anotada en la observación)
  y los últimos 7 la B. Historia para el asistente: falta una Pepsi lata en 8 de los 14 cierres, dos latas de Fanta
  abolladas, una entrada de botellas mal registrada (sobrante) y una Coca lata faltante.
- Al arrancar, el servidor carga los modelos en segundo plano (unos 10 s). Esperar a que termine antes del primer
  escaneo. En el laboratorio, la **primera** pasada con la variante A tarda unos 10 s más, porque carga
  TensorFlow: hacerla una vez antes de grabar.
- Si se vuelve a correr `demo_poblar.py`, se pierde todo lo que se haya hecho en la demo (queda en el respaldo).
- Si el navegador muestra una versión vieja de una página, recargar con Ctrl+F5 (el servidor ya pide no guardar
  caché, pero un navegador que la guardó antes la puede seguir usando una vez).

Fotos en `datos/demo/` (estantes sintéticos; `verdad.json` tiene el conteo exacto de cada una):

| Archivo | Para qué |
|---|---|
| `estante_hoy.jpg` | **el escaneo en vivo**: da 1 Pepsi lata faltante y 1 Coca lata sobrante contra el stock de hoy |
| `estante_ordenado.jpg`, `estante_normal_5niveles.jpg`, `estante_desordenado.jpg` | laboratorio o escáner |

## Guion sugerido

1. **Panel → Productos y Stock** (`/panel/`): los 8 productos, su precio y el stock esperado de hoy.
2. **Movimientos**: ventas y entradas del día (las registra el sistema de ventas; acá se ven).
3. **Escáner** (`/escaner/`, o desde el celular por HTTPS, ver `docs/https_celular.md`):
   *Galería* → `estante_hoy.jpg` → aparece la imagen anotada y la conciliación: Pepsi lata −1, Coca lata +1.
   El cajero corrige si hace falta, escribe una observación y confirma: se generan los ajustes.
4. **Panel → Cierres**: el cierre nuevo arriba. Abrir uno de la primera semana (variante A) para mostrar las
   correcciones del cajero, y uno de la segunda (variante B).
5. **Historial**: el gráfico de diferencias y el resumen por producto. Pepsi lata es el problema.
6. **Asistente**: «¿Qué producto tuvo más faltantes en las últimas dos semanas?», «¿Cuánto se perdió en Pepsi
   lata en octubre?», «¿Qué pasó en el cierre 5?».
7. **Laboratorio** (`/laboratorio/`): *Estante sintético* → *Generar y analizar*. La tabla compara lo que ve el
   modelo con el conteo real. Cambiar entre **A** y **B** con los botones y mover los umbrales en vivo. Tocar una
   caja para ver las probabilidades del clasificador. *Solo clasificador* clasifica fotos de un envase (con la
   variante B, que no tiene clasificador, usa el principal).

## Qué decir con honestidad

- Los estantes son **sintéticos**: no hay fotos de un estante real. Los detectores se entrenaron con 10 000
  estantes del mismo generador y los mismos envases, así que en sintéticos aciertan casi siempre: la variante B
  contó exacto 45 de 45 estantes nuevos y la A, 12 de 45 (su clasificador solo vio recortes de fotos reales).
  Las métricas que valen son las de las fotos reales de test (`runs/detector/colab/metricas_colab.json`,
  `modelos/clasificador/comparacion.md`).
- El historial es inventado por `demo_poblar.py`, aunque cada cierre pasó por el modelo real.

## Si algo falla

- **El asistente responde con plantilla** (en el log del servidor: `AuthenticationError`): la clave
  `ANTHROPIC_API_KEY` de `.env` no es válida. Las respuestas siguen saliendo de la BD, pero sin redacción de
  Claude.
- **El escaneo da conteos al azar**: `VARIANTE_VISION` quedó en `mock`, o faltan los pesos en `modelos/detector/`.
- **El laboratorio no muestra *Estante sintético***: falta la biblioteca de envases `datos/sinteticos/_objetos/`.
