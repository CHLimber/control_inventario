# Guion de la demo con la maqueta (≈ 6-8 min)

Idea en una frase: **es el arqueo de caja, pero de mercadería.** Cargamos lo que hay en el estante,
vendemos registrando, el celular escanea y todo cuadra; después "desaparece" una Fanta sin
registrar, el escaneo detecta el faltante, se confirma el cierre y el asistente lo explica.

Roles: **presentador** (habla y maneja el panel en la PC/proyector) y **cajero** (maneja la maqueta y
el celular). Puede ser la misma persona.

---

## 0. Preparación (antes de que entre el público)

- [ ] Servidor levantado en la PC y panel abierto en el proyector (ver README).
- [ ] Celular en la misma red, con la página del escáner abierta y probada (una foto de prueba).
- [ ] `VARIANTE_VISION` con el modelo final (no `mock`) y la variante anotada para decirla.
- [ ] Productos cargados en el panel con su `clase_modelo` y precio (las 8 clases).
- [ ] Maqueta armada con la **configuración demo** (tabla de abajo), con el **fondo y la luz que
      mejor resultado dieron** en la sesión de validación. Etiquetas de frente, nada girado.
- [ ] Marcas en el piso para las fotos (frontal, izq20, der20). Celular cargado.
- [ ] Ensayo completo hecho al menos una vez **en el mismo lugar y con la misma luz**.
- [ ] Plan B listo: video grabado del ensayo y fotos del ensayo en el celular.

### Configuración demo (17 envases)

Agrupada por marca para que se vea clara. Niveles de abajo hacia arriba, izquierda → derecha:

| Nivel | Envases |
|---|---|
| 3 · latas | Coca-Cola · Coca-Cola · Fanta · Fanta · Sprite · Pepsi |
| 2 · botellas chicas | Coca-Cola · Fanta · Fanta · Sprite · Pepsi · Pepsi |
| 1 · botellas grandes | Coca-Cola · Coca-Cola · Fanta · Sprite · Pepsi |

| Clase | Stock inicial |
|---|---|
| coca_botella | 3 |
| coca_lata | 2 |
| fanta_botella | 3 |
| fanta_lata | 2 |
| sprite_botella | 2 |
| sprite_lata | 1 |
| pepsi_botella | 3 |
| pepsi_lata | 1 |

---

## 1. Cargar el stock inicial (≈ 1 min)

**Decir:** "Empezamos el día. Esto es lo que hay físicamente en el estante."

- En el panel, registrar un movimiento **`stock_inicial`** por producto con las cantidades de la
  tabla (o `POST /api/movimientos` con `tipo: "stock_inicial"`). Un `stock_inicial` nuevo reinicia
  el stock esperado, así que la demo se puede repetir.
- Mostrar la vista de **stock**: coincide con la tabla.

## 2. Vender registrando (≈ 1 min)

**Decir:** "Llega un cliente y compra; el cajero lo registra como siempre."

- El cajero **retira** del estante 1 **Coca-Cola lata** y 1 **Pepsi botella chica** (las deja
  fuera de cuadro).
- En el panel registrar dos movimientos **`venta`** de 1 unidad: `coca_lata` y `pepsi_botella`.
- Stock esperado ahora: coca_lata 1, pepsi_botella 2 (el resto igual).

## 3. Escanear y ver que cuadra (≈ 1.5 min)

**Decir:** "Al cierre, en vez de contar a mano, sacamos tres fotos."

- Con el escáner del celular, 3 fotos (frontal, izq20, der20) y enviar.
  El servidor detecta y clasifica cada envase en cada foto y combina los conteos con la **mediana
  por clase** (una foto con un error no arruina el conteo).
- En el panel, abrir el cierre **pendiente**: imagen anotada con las cajas, y la tabla
  **esperado / detectado / diferencia**: todas las filas en **ok** (diferencia 0).
- Confirmar el cierre (sin ajustes).

## 4. El faltante (≈ 2 min)

**Decir:** "Ahora, sin que nadie lo registre, desaparece una Fanta."

- Alguien del público (o el presentador) **retira 1 Fanta lata** y se la guarda. **No se registra
  nada en el panel.**
- Escanear de nuevo con 3 fotos.
- En el cierre pendiente: `fanta_lata` **esperado 2, detectado 1, diferencia −1 → faltante**; el
  resto en ok. Señalar en la imagen anotada que hay una sola lata de Fanta.

**Decir:** "El sistema no adivina quién fue; muestra la diferencia y el cajero decide."

- (Si el modelo se equivoca en otra clase, mostrarlo con naturalidad: el cajero corrige la columna
  **ajustado** antes de confirmar. Eso es parte del diseño: la persona tiene la última palabra.)
- **Confirmar el cierre.** Al confirmar se crea un movimiento `ajuste` de −1 para `fanta_lata`
  vinculado al cierre, así el stock esperado de mañana parte del conteo real y el faltante queda
  registrado.

## 5. Preguntarle al asistente (≈ 1 min)

En el panel, en el asistente, escribir:

> **¿Qué faltó hoy?**

Respuesta esperada (en sus palabras): que hoy hubo **un faltante de 1 Fanta lata** en el cierre
de las HH:MM, con su valor en Bs (precio de la Fanta lata), y que el resto cuadró. Si da tiempo,
una segunda pregunta: *"¿Cuánto vendimos de Coca-Cola hoy?"* o *"¿Qué producto tuvo más
faltantes esta semana?"*.

## 6. Cierre (30 s)

**Decir:** "Del estante a la decisión: fotos → conteo por visión → conciliación con la base de datos
→ explicación en lenguaje natural. Las métricas del informe se midieron en la sesión 3 de la
maqueta, que el modelo nunca vio al entrenar."

---

## Si algo sale mal

| Problema | Qué hacer |
|---|---|
| El celular no conecta | Subir las fotos del ensayo desde la PC (mismo endpoint) |
| El conteo falla en una clase | Corregir en **ajustado** y explicar el control humano; seguir |
| Reflejos fuertes en el proyector/lugar | Girar la maqueta para que la luz llegue de costado; evitar el flash |
| Todo se cae | Pasar el video del ensayo y mostrar el informe |

Después de la demo: devolver la Fanta al estante y, si se va a repetir, volver a cargar el
`stock_inicial` de la tabla.
