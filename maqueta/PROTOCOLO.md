# Protocolo de captura de fotos de la maqueta

Objetivo: fotos con **verdad de terreno exacta** (cuántos envases de cada clase hay en cada foto),
variadas en lo que importa (cantidades, orden, luz, ángulo, fondo) y separadas de forma que la
evaluación sea honesta. Antes de esto, la maqueta tiene que estar armada ([GUIA.md](GUIA.md)).

---

## 1. Conceptos

- **Escena** = una configuración física del estante con cantidades **conocidas y registradas ANTES
  de fotografiar**. Una escena = una fila en `datos/maqueta/escenas.csv`.
- **Foto** = cada toma de una escena. Por escena se sacan **3 fotos**, siempre en el mismo orden:

| Foto | Ángulo | Cómo |
|---|---|---|
| 1 | `frontal` | de frente, centrado en el estante |
| 2 | `izq20` | ≈ 20° a la izquierda (a 1 m de distancia ⇒ ≈ 35 cm al costado), apuntando al centro |
| 3 | `der20` | ≈ 20° a la derecha, igual |

  Marcá las tres posiciones en el piso con masking tape para repetirlas siempre igual.
- **Sesión** = un día de fotos. Meta: **60-100 escenas en al menos 3 sesiones**, en **días y con luz
  distintos**. La **sesión 3 se reserva SOLO para prueba** (sección 5).

### `escenas.csv` (verdad de terreno)

```
sesion,escena,fecha,luz,fondo,coca_botella,coca_lata,fanta_botella,fanta_lata,sprite_botella,sprite_lata,pepsi_botella,pepsi_lata,7up_lata,observacion,distancia,celular
1,1,2026-10-18,natural,blanco,3,3,3,1,4,1,5,2,2,lleno; girada 1 coca_botella,completo,Xiaomi Redmi Note 12
```

| Columna | Valores |
|---|---|
| `luz` | `fria_techo` (foco/tubo de techo), `natural` (ventana, de día), `calida` (foco amarillo/lámpara), `poca_luz` (luces apagadas, algo de luz indirecta), `flash` (flash del celular) |
| `fondo` | `blanco`, `madera`, `negro` |
| clases | cantidad **en todo lo que muestra la foto** (entero ≥ 0) |
| `observacion` | texto libre: qué se giró/tapó, si algo salió distinto al plan |
| `distancia` | `completo` (estante entero) o `nivel_k` (acercamiento a un solo nivel; los conteos son **solo de ese nivel**) |
| `celular` | modelo del celular (si la dejás vacía, la ingesta la completa con el EXIF) |

Las columnas `distancia` y `celular` son nuevas y van **al final**; las anteriores no cambian.

Las fotos quedan en `datos/maqueta/sesion_XX/sesionXX_escenaYY_fotoZ.jpg` (no se versionan en git;
`escenas.csv`, `escenas_plan.csv` y `fotos.csv` sí).

---

## 2. Qué variar

**Entre escenas** (cada escena distinta de la anterior; el plan ya lo hace):

- **cantidades** por clase, incluyendo **marcas enteras en cero** (p. ej. "sin Fanta": ni botella
  ni lata) y clases sueltas en cero;
- **niveles medio vacíos** (huecos en la fila, como un estante a fin de día) y escenas llenas;
- **orden**: agrupado por marca y mezclado;
- **productos girados** (de costado o de espalda: el modelo tiene que reconocerlos igual o al
  menos no inventar otra clase);
- **etiqueta parcialmente oculta** (papel, tira de precio, otro envase delante en diagonal);
- **luz** y **fondo** (el fondo se cambia por bloques para no perder tiempo);
- **distancia**: estante completo, y cada ≈ 6 escenas un **acercamiento** a un solo nivel
  (no se mueve nada; es casi gratis);
- **celular**: si podés, usá otro celular (de un familiar o compañero) en parte de la sesión 2 o
  en la 3, y anotalo en `celular`. Cámaras distintas = colores y nitidez distintos.

**Dentro de una escena** solo cambia el **ángulo** (las 3 fotos). Luz y fondo son de la escena.

---

## 3. Procedimiento por escena (≈ 4-5 min)

1. **Armar** según la hoja del plan (`escenas_plan_sesionXX.md`): nivel por nivel, de izquierda a
   derecha. Etiquetas de frente salvo lo que pida girar/tapar.
2. **Revisar** contando de nuevo, nivel por nivel, contra "Cantidades" de la hoja.
3. **Registrar ANTES de fotografiar**: tildá "anotada" en la hoja; si cambiaste algo respecto del
   plan (te faltó un envase, se volteó una lata), **corregí las cantidades en la hoja en ese
   momento**. Si preferís, escribí la fila directamente en `escenas.csv` en la PC.
4. **Luz**: prendé/apagá lo que corresponda; esperá 2-3 s a que el celular ajuste la exposición.
5. **3 fotos**: frontal → izq20 → der20, desde las marcas del piso, a la altura del nivel 2.
   Si una sale mal (movida, desenfocada, con la mano), **repetila y borrá la mala en el momento**:
   la ingesta agrupa las fotos de a 3 en orden y una foto de más corre todas las escenas siguientes.
6. Tildá las 3 fotos en la hoja y pasá a la siguiente escena.

> En las escenas de **acercamiento** no se toca el estante: se encuadra solo el nivel indicado
> (a ≈ 40-50 cm), con el nivel completo de borde a borde y sin que entren envases de otros niveles.

---

## 4. Después de cada sesión: ingesta

```bat
conda activate inventariovision
:: 1) copiar las fotos de la sesión del celular a una carpeta, p. ej. D:\fotos\sesion1
:: 2) simular (no escribe nada) y leer los avisos
python scripts\maqueta_ingesta.py D:\fotos\sesion1 --sesion 1 --simular
:: 3) si cuadra, de verdad; --copiar-plan pasa a escenas.csv las filas del plan que no anotaste a mano
python scripts\maqueta_ingesta.py D:\fotos\sesion1 --sesion 1 --copiar-plan
```

La ingesta:
- ordena por fecha EXIF (`--orden nombre` para ordenar por nombre de archivo);
- copia (no mueve, **nunca pisa**) a `datos/maqueta/sesion_01/sesion01_escenaYY_fotoZ.jpg`;
- avisa de **fotos sobrantes** (si el total no es múltiplo de 3), escenas sin fila en
  `escenas.csv`, escenas anotadas sin fotos, **conteos imposibles** (negativos, más envases de los
  que tenés o más de los que entran en un nivel) y escenas cuyas fotos están muy separadas en el
  tiempo (síntoma de que se corrió el orden);
- registra cada foto en `datos/maqueta/fotos.csv` (ángulo, archivo original, fecha, celular);
- si cargás una sesión en partes, usá `--escena-inicial 14` para la segunda tanda.

Si la maqueta tiene otras medidas que las por defecto, pasale los mismos `--ancho`, `--envases` y
`--grandes` que al plan para que el control de capacidad sea el correcto.

Después de la ingesta, **mirá 3-4 fotos al azar** contra su fila de `escenas.csv`.

---

## 5. Sesiones y separación entrenamiento / prueba

| Sesión | Escenas | Uso | Condiciones sugeridas |
|---|---|---|---|
| 1 | 20-30 | entrenamiento (ajuste fino) | un día, de tarde, las 5 luces |
| 2 | 20-30 | entrenamiento + **validación** (elegir umbrales, comparar variantes) | otro día, otra hora (p. ej. de noche + mañana), otro celular si se puede |
| 3 | 20-40 | **SOLO PRUEBA** (las métricas del informe) | otro día, idealmente con algún cambio que no se vio antes |

**¿Por qué la sesión 3 queda reservada y no se mezclan fotos al azar?** Las 3 fotos de una escena
son casi la misma imagen (mismos envases, mismo orden, misma luz; solo cambia un poco el ángulo), y
las escenas de un mismo día comparten luz, fondo y hasta el polvo del lente. Si una foto de una
escena cae en entrenamiento y otra de la **misma escena** en prueba, el modelo "ya vio" esa prueba:
la métrica sale inflada y no dice nada sobre cómo le va con un estante nuevo (es una **fuga de
datos**). Separando **por sesión completa**, la prueba tiene escenas, días y luz que el modelo nunca
vio, que es lo que pasa en el uso real.

Reglas:
- La sesión 3 **no se usa para nada** antes de la evaluación final: ni para entrenar, ni para
  elegir umbrales, ni para "mirar cómo va".
- Si después de evaluar en la sesión 3 cambiás algo y querés volver a medir, hace falta una
  **sesión 4** nueva (la 3 ya quedó "vista").
- Los acercamientos y las tres fotos de una escena van siempre juntos al mismo lado de la división.

Para generar los planes de las sesiones 2 y 3 (mismos parámetros que la 1):

```bat
python scripts\maqueta_plan.py --sesiones 2 --escenas 26
python scripts\maqueta_plan.py --sesiones 3 --escenas 30
```

Cada sesión tiene su propia semilla derivada: el plan de la sesión 1 no cambia al generar las otras.

---

## 6. Checklist imprimible por sesión

Imprimí esta hoja junto con `escenas_plan_sesionXX.md`.

```
SESIÓN: ____    FECHA: ____/____/2026    HORA INICIO: ______    CELULAR: __________________
Uso:  [ ] entrenamiento   [ ] entrenamiento + validación   [ ] SOLO PRUEBA (sesión 3)

ANTES
[ ] Maqueta firme, contra la pared; repisas derechas
[ ] Envases completos (contar por clase) y limpios; tapas puestas; latas con peso
[ ] Fondos blanco / madera / negro a mano; clips o masking
[ ] Luces: foco de techo [ ]  ventana con luz de día [ ]  lámpara cálida [ ]  flash probado [ ]
[ ] Marcas en el piso: frontal / izq20 / der20 (y la de acercamiento)
[ ] Celular: fecha y hora OK, JPG (no HEIC), lente limpio, sin zoom ni filtros, batería > 50 %
[ ] Álbum/carpeta nueva solo para esta sesión
[ ] Hoja del plan impresa + lapicera

DURANTE (cada escena)
[ ] Armar según el plan  →  [ ] contar de nuevo  →  [ ] anotar/corregir cantidades
[ ] Luz de la escena  →  [ ] foto 1 frontal  [ ] foto 2 izq20  [ ] foto 3 der20
[ ] Foto mala: repetir y BORRAR la mala en el momento

DESPUÉS
[ ] Total de fotos = 3 × escenas  (escenas: ____  fotos: ____)
[ ] Copiar fotos a la PC (carpeta solo de esta sesión)
[ ] python scripts\maqueta_ingesta.py <carpeta> --sesion N --simular   → sin problemas
[ ] python scripts\maqueta_ingesta.py <carpeta> --sesion N --copiar-plan
[ ] Revisar 3-4 fotos al azar contra escenas.csv
[ ] Commit de escenas.csv y fotos.csv (las .jpg NO van a git)

Notas / cambios respecto al plan:
______________________________________________________________________________
______________________________________________________________________________
```
