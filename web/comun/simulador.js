/*
 * MODO SIMULADO de la API: responde en el navegador con la forma exacta de
 * contratos/api_schemas.py. Sirve para probar la web sin backend.
 *
 * - Los datos se guardan en localStorage (si se puede) para que el escáner y
 *   el panel compartan los mismos cierres simulados.
 * - La imagen anotada se dibuja con canvas: sobre la foto enviada (escaneo) o
 *   sobre un estante sintético (cierres de ejemplo).
 * - Pregunta al asistente que contenga "#503" → simula el asistente caído.
 */
(function () {
  'use strict';

  // Copia de config/clases.json (fuente de verdad; GET /api/clases la sirve tal cual).
  const CLASES = {
    version: 1,
    descripcion: "FUENTE DE VERDAD de las clases. El orden de 'clases' es el índice de salida de los modelos (softmax del clasificador y clases del YOLO multiclase). No reordenar sin reentrenar.",
    clases: [
      { id: 0, clase: 'coca_botella', nombre: 'Coca-Cola botella', marca: 'Coca-Cola', formato: 'botella' },
      { id: 1, clase: 'coca_lata', nombre: 'Coca-Cola lata', marca: 'Coca-Cola', formato: 'lata' },
      { id: 2, clase: 'fanta_botella', nombre: 'Fanta botella', marca: 'Fanta', formato: 'botella' },
      { id: 3, clase: 'fanta_lata', nombre: 'Fanta lata', marca: 'Fanta', formato: 'lata' },
      { id: 4, clase: 'sprite_botella', nombre: 'Sprite botella', marca: 'Sprite', formato: 'botella' },
      { id: 5, clase: 'sprite_lata', nombre: 'Sprite lata', marca: 'Sprite', formato: 'lata' },
      { id: 6, clase: 'pepsi_botella', nombre: 'Pepsi botella', marca: 'Pepsi', formato: 'botella' },
      { id: 7, clase: 'pepsi_lata', nombre: 'Pepsi lata', marca: 'Pepsi', formato: 'lata' },
    ],
    desconocido: { clase: 'desconocido', nombre: 'No reconocido' },
    detector_1clase: { id: 0, clase: 'producto', nombre: 'Producto' },
  };

  const COLOR_MARCA = {
    'Coca-Cola': '#d7191c', Fanta: '#f28e1c', Sprite: '#1a9850', Pepsi: '#2c5fb8',
  };
  const CLAVE_ESTADO = 'inventario_simulado_v1';
  const PREFIJO_IMG = 'inventario_simulado_img_';
  const VARIANTE = 'mock';

  // ------------------------------------------------------------ utilidades
  function aleatorio(semilla) { // mulberry32: determinista para los datos de ejemplo
    let a = semilla >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const dos = (n) => String(n).padStart(2, '0');
  function isoLocal(d) {
    return `${d.getFullYear()}-${dos(d.getMonth() + 1)}-${dos(d.getDate())}T${dos(d.getHours())}:${dos(d.getMinutes())}:${dos(d.getSeconds())}`;
  }
  function diaRelativo(dias, h, m) {
    const d = new Date();
    d.setDate(d.getDate() + dias);
    d.setHours(h, m || 0, 0, 0);
    return d;
  }
  const clon = (o) => JSON.parse(JSON.stringify(o));
  const esperar = (ms) => new Promise((r) => setTimeout(r, ms));
  const resp = (status, data) => ({ status, data });
  const error = (status, detail) => resp(status, { detail });

  function enRango(fechaIso, desde, hasta) {
    const dia = fechaIso.slice(0, 10);
    if (desde && dia < desde) return false;
    if (hasta && dia > hasta) return false;
    return true;
  }

  // ------------------------------------------------------------ estado
  let estado = null;
  const imagenesMemoria = {}; // id_cierre → data URL anotada
  const sinteticas = {};      // id_cierre → data URL de estante sintético

  function guardar() {
    try { localStorage.setItem(CLAVE_ESTADO, JSON.stringify(estado)); } catch (e) { /* sin almacenamiento */ }
  }

  function cargar() {
    if (estado) return;
    try {
      const txt = localStorage.getItem(CLAVE_ESTADO);
      if (txt) { estado = JSON.parse(txt); return; }
    } catch (e) { /* sin almacenamiento */ }
    sembrar();
    guardar();
  }

  function reiniciar() {
    try {
      Object.keys(localStorage).filter((k) => k.startsWith(PREFIJO_IMG)).forEach((k) => localStorage.removeItem(k));
      localStorage.removeItem(CLAVE_ESTADO);
    } catch (e) { /* nada */ }
    estado = null;
    Object.keys(imagenesMemoria).forEach((k) => delete imagenesMemoria[k]);
    Object.keys(sinteticas).forEach((k) => delete sinteticas[k]);
    cargar();
  }

  function sembrar() {
    estado = { productos: [], movimientos: [], cierres: [], sig: { producto: 1, movimiento: 1, cierre: 1 } };
    const precios = { botella: 10.0, lata: 7.0 };
    CLASES.clases.forEach((c) => {
      estado.productos.push({
        id: estado.sig.producto++, nombre: c.nombre, marca: c.marca, formato: c.formato,
        precio: precios[c.formato] + (c.marca === 'Coca-Cola' ? 1 : 0), clase_modelo: c.clase, activo: true,
      });
    });
    const rnd = aleatorio(20261015);
    // Stock inicial hace 7 días.
    estado.productos.forEach((p) => {
      agregarMovimiento(p.id, 'stock_inicial', p.formato === 'botella' ? 12 : 18, diaRelativo(-7, 8), 'admin');
    });
    for (let d = -7; d <= 0; d++) {
      if (d === -4) estado.productos.forEach((p) => agregarMovimiento(p.id, 'entrada', 6, diaRelativo(d, 9), 'admin'));
      estado.productos.forEach((p) => {
        const n = Math.floor(rnd() * 3); // 0..2 ventas por producto y día
        for (let i = 0; i < n; i++) agregarMovimiento(p.id, 'venta', 1, diaRelativo(d, 10 + Math.floor(rnd() * 9), Math.floor(rnd() * 60)), 'cajero');
      });
      if (d >= -3 && d <= -1) {
        const c = crearCierre(diaRelativo(d, 21, 30), 'cajero', rnd, 0);
        const ajustes = [];
        // A veces el cajero corrige un conteo del modelo.
        if (d === -2) {
          const f = c.filas.find((x) => x.diferencia !== 0);
          if (f) ajustes.push({ id_producto: f.id_producto, ajustado: f.esperado });
        }
        confirmar(c, { ajustes, observacion: d === -2 ? 'Se recontó a mano una fila.' : null, usuario: 'cajero' }, diaRelativo(d, 21, 45));
      }
    }
  }

  function agregarMovimiento(idProducto, tipo, cantidad, fecha, usuario, idCierre) {
    const m = {
      id: estado.sig.movimiento++, id_producto: idProducto, tipo, cantidad,
      fecha: typeof fecha === 'string' ? fecha : isoLocal(fecha), usuario: usuario || 'cajero', id_cierre: idCierre || null,
    };
    estado.movimientos.push(m);
    return m;
  }

  function stockHasta(limiteIso) {
    return estado.productos.filter((p) => p.activo).map((p) => {
      const it = { id_producto: p.id, clase_modelo: p.clase_modelo, nombre: p.nombre, stock_inicial: 0, entradas: 0, ventas: 0, ajustes: 0, stock: 0 };
      estado.movimientos.forEach((m) => {
        if (m.id_producto !== p.id || m.fecha > limiteIso) return;
        if (m.tipo === 'stock_inicial') it.stock_inicial += m.cantidad;
        else if (m.tipo === 'entrada') it.entradas += m.cantidad;
        else if (m.tipo === 'venta') it.ventas += m.cantidad;
        else if (m.tipo === 'ajuste') it.ajustes += m.cantidad;
      });
      it.stock = it.stock_inicial + it.entradas - it.ventas + it.ajustes;
      return it;
    });
  }

  function estadoFila(dif) { return dif < 0 ? 'faltante' : dif > 0 ? 'sobrante' : 'ok'; }

  function calcularTotales(filas) {
    const t = { esperado: 0, detectado: 0, diferencia: 0, productos_con_diferencia: 0 };
    filas.forEach((f) => {
      t.esperado += f.esperado;
      t.detectado += (f.ajustado ?? f.detectado);
      t.diferencia += f.diferencia;
      if (f.diferencia !== 0) t.productos_con_diferencia++;
    });
    return t;
  }

  /** Simula el conteo del modelo: esperado ± pequeñas diferencias, con cajas. */
  function crearCierre(fecha, usuario, rnd, nImagenes) {
    const fechaIso = isoLocal(fecha);
    const stock = stockHasta(fechaIso);
    const filas = [];
    const cajas = [];
    stock.forEach((s) => {
      const r = rnd();
      const delta = r < 0.55 ? 0 : r < 0.8 ? -1 : r < 0.9 ? -2 : 1;
      const detectado = Math.max(0, s.stock + delta);
      const confs = [];
      for (let i = 0; i < detectado; i++) {
        const cd = 0.7 + rnd() * 0.29;
        const cc = 0.6 + rnd() * 0.39;
        confs.push(cc);
        cajas.push({ x: 0, y: 0, w: 0, h: 0, clase: s.clase_modelo, conf_det: +cd.toFixed(3), conf_cls: +cc.toFixed(3), indice_imagen: 0 });
      }
      const dif = detectado - s.stock;
      filas.push({
        id_producto: s.id_producto, clase_modelo: s.clase_modelo, nombre: s.nombre,
        esperado: s.stock, detectado, ajustado: null, diferencia: dif, estado: estadoFila(dif),
        confianza_media: confs.length ? +(confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(3) : null,
      });
    });
    const noRec = Math.floor(rnd() * 3);
    for (let i = 0; i < noRec; i++) {
      cajas.push({ x: 0, y: 0, w: 0, h: 0, clase: 'desconocido', conf_det: +(0.5 + rnd() * 0.3).toFixed(3), conf_cls: +(0.2 + rnd() * 0.3).toFixed(3), indice_imagen: 0 });
    }
    // Reparte las cajas entre las imágenes enviadas (si hay más de una).
    if (nImagenes > 1) cajas.forEach((c, i) => { c.indice_imagen = i % nImagenes; });
    ubicarCajas(cajas);
    const c = {
      id: estado.sig.cierre++, fecha: fechaIso, usuario: usuario || 'cajero', estado: 'pendiente',
      variante_modelo: VARIANTE, totales: calcularTotales(filas), observacion: null,
      imagen_anotada_url: null, imagenes_urls: [], filas, cajas, no_reconocidos: noRec,
      tiempo_ms: Math.round(600 + rnd() * 1800),
    };
    estado.cierres.push(c);
    return c;
  }

  /** Coordenadas en píxeles (esquina sup. izq.) sobre un lienzo de 1280×800, en 3 estantes. */
  function ubicarCajas(cajas) {
    const W = 1280, H = 800, filas = 3, margen = 40;
    const porEstante = Math.max(1, Math.ceil(cajas.length / filas));
    const ancho = (W - 2 * margen) / porEstante;
    const altoEstante = (H - 60) / filas;
    cajas.forEach((c, i) => {
      const fila = Math.floor(i / porEstante);
      const col = i % porEstante;
      const esLata = /lata/.test(c.clase) || (c.clase === 'desconocido' && i % 2 === 0);
      const h = altoEstante * (esLata ? 0.45 : 0.82);
      const w = Math.min(ancho * 0.86, esLata ? 70 : 60);
      const base = 60 + (fila + 1) * altoEstante - 12;
      c.x = Math.round(margen + col * ancho + (ancho - w) / 2);
      c.y = Math.round(base - h);
      c.w = Math.round(w);
      c.h = Math.round(h);
    });
  }

  function confirmar(c, entrada, fecha) {
    const ajustes = new Map((entrada.ajustes || []).map((a) => [a.id_producto, a.ajustado]));
    c.filas.forEach((f) => {
      f.ajustado = ajustes.has(f.id_producto) ? ajustes.get(f.id_producto) : f.detectado;
      f.diferencia = f.ajustado - f.esperado;
      f.estado = estadoFila(f.diferencia);
    });
    c.totales = calcularTotales(c.filas);
    c.estado = 'confirmado';
    c.observacion = entrada.observacion || null;
    const fechaMov = fecha ? isoLocal(fecha) : isoLocal(new Date());
    c.filas.forEach((f) => {
      if (f.diferencia !== 0) agregarMovimiento(f.id_producto, 'ajuste', f.diferencia, fechaMov, entrada.usuario || 'cajero', c.id);
    });
  }

  // ------------------------------------------------------------ imágenes
  const COLOR_CLASE = {};
  CLASES.clases.forEach((c) => { COLOR_CLASE[c.clase] = COLOR_MARCA[c.marca]; });

  function dibujarCajas(ctx, cajas, escalaX, escalaY) {
    ctx.lineWidth = 3;
    ctx.font = 'bold 13px sans-serif';
    cajas.forEach((c) => {
      const color = COLOR_CLASE[c.clase] || '#888';
      const x = c.x * escalaX, y = c.y * escalaY, w = c.w * escalaX, h = c.h * escalaY;
      ctx.strokeStyle = c.clase === 'desconocido' ? '#ff00ff' : '#ffeb3b';
      ctx.setLineDash(c.clase === 'desconocido' ? [6, 4] : []);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      const txt = c.clase === 'desconocido' ? '?' : c.clase.split('_')[0] + ' ' + (c.conf_cls ?? c.conf_det).toFixed(2);
      const tw = ctx.measureText(txt).width + 6;
      ctx.fillStyle = c.clase === 'desconocido' ? '#ff00ff' : color;
      ctx.fillRect(x, y - 16, tw, 16);
      ctx.fillStyle = '#fff';
      ctx.fillText(txt, x + 3, y - 4);
    });
  }

  function marcaAgua(ctx, W, texto) {
    ctx.fillStyle = 'rgba(0,0,0,.6)';
    ctx.fillRect(0, 0, W, 34);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 18px sans-serif';
    ctx.fillText(texto, 12, 23);
  }

  /** Estante sintético con botellas/latas dibujadas y sus cajas. */
  function imagenSintetica(c) {
    if (sinteticas[c.id]) return sinteticas[c.id];
    const W = 1280, H = 800;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#d9cbb3'); g.addColorStop(1, '#b8a383');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    const altoEstante = (H - 60) / 3;
    for (let f = 1; f <= 3; f++) {
      ctx.fillStyle = '#6d4c33';
      ctx.fillRect(20, 60 + f * altoEstante - 12, W - 40, 14);
    }
    const cajas = c.cajas.filter((k) => k.indice_imagen === 0);
    cajas.forEach((k) => {
      const color = COLOR_CLASE[k.clase] || '#9e9e9e';
      ctx.fillStyle = color;
      const esBotella = /botella/.test(k.clase);
      if (esBotella) {
        ctx.fillRect(k.x + k.w * 0.32, k.y, k.w * 0.36, k.h * 0.25);
        ctx.fillRect(k.x + 4, k.y + k.h * 0.22, k.w - 8, k.h * 0.78);
      } else {
        ctx.fillRect(k.x + 3, k.y + 4, k.w - 6, k.h - 4);
      }
      ctx.fillStyle = 'rgba(255,255,255,.85)';
      ctx.fillRect(k.x + 4, k.y + k.h * 0.55, k.w - 8, k.h * 0.14);
    });
    dibujarCajas(ctx, cajas, 1, 1);
    marcaAgua(ctx, W, `SIMULADO · cierre #${c.id} · ${cajas.length} cajas en la foto 1`);
    try { sinteticas[c.id] = cv.toDataURL('image/jpeg', 0.8); } catch (e) { sinteticas[c.id] = ''; }
    return sinteticas[c.id];
  }

  /** Dibuja las cajas simuladas sobre la primera foto enviada. */
  async function anotarFoto(c, blob) {
    let fuente;
    try {
      fuente = await createImageBitmap(blob);
    } catch (e) {
      return null;
    }
    const W = 1280, H = 800;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    // "cover" de la foto sobre 1280×800
    const esc = Math.max(W / fuente.width, H / fuente.height);
    const dw = fuente.width * esc, dh = fuente.height * esc;
    ctx.drawImage(fuente, (W - dw) / 2, (H - dh) / 2, dw, dh);
    if (fuente.close) fuente.close();
    ctx.fillStyle = 'rgba(0,0,0,.15)'; ctx.fillRect(0, 0, W, H);
    const cajas = c.cajas.filter((k) => k.indice_imagen === 0);
    dibujarCajas(ctx, cajas, 1, 1);
    marcaAgua(ctx, W, `SIMULADO · cierre #${c.id} · cajas de ejemplo (no son detecciones reales)`);
    return cv.toDataURL('image/jpeg', 0.8);
  }

  function imagenDe(c) {
    if (imagenesMemoria[c.id]) return imagenesMemoria[c.id];
    try {
      const guardada = localStorage.getItem(PREFIJO_IMG + c.id);
      if (guardada) { imagenesMemoria[c.id] = guardada; return guardada; }
    } catch (e) { /* nada */ }
    return imagenSintetica(c);
  }

  function guardarImagen(id, url) {
    imagenesMemoria[id] = url;
    try {
      // Solo se conservan las 3 últimas fotos para no llenar el almacenamiento.
      const claves = Object.keys(localStorage).filter((k) => k.startsWith(PREFIJO_IMG))
        .sort((a, b) => Number(a.slice(PREFIJO_IMG.length)) - Number(b.slice(PREFIJO_IMG.length)));
      while (claves.length >= 3) localStorage.removeItem(claves.shift());
      localStorage.setItem(PREFIJO_IMG + id, url);
    } catch (e) { /* sin espacio: queda solo en memoria */ }
  }

  // ------------------------------------------------------------ salidas
  function cierreOut(c) {
    const o = clon(c);
    o.imagen_anotada_url = imagenDe(c);
    return o;
  }
  function cierreResumen(c) {
    const { id, fecha, usuario, estado: est, variante_modelo, totales } = c;
    return clon({ id, fecha, usuario, estado: est, variante_modelo, totales });
  }

  function validarProducto(b, parcial, idActual) {
    if (!parcial || b.nombre !== undefined) {
      if (!b.nombre || !String(b.nombre).trim()) return 'nombre: no puede estar vacío';
    }
    if (!parcial || b.marca !== undefined) {
      if (!b.marca || !String(b.marca).trim()) return 'marca: no puede estar vacía';
    }
    if ((!parcial || b.formato !== undefined) && !['botella', 'lata'].includes(b.formato)) return "formato: debe ser 'botella' o 'lata'";
    if (!parcial || b.precio !== undefined) {
      const p = Number(b.precio);
      if (b.precio === null || b.precio === '' || Number.isNaN(p) || p < 0) return 'precio: debe ser un número ≥ 0';
    }
    if (!parcial || b.clase_modelo !== undefined) {
      if (!b.clase_modelo) return 'clase_modelo: obligatorio';
      if (estado.productos.some((p) => p.clase_modelo === b.clase_modelo && p.id !== idActual)) {
        return `Ya existe un producto con clase_modelo '${b.clase_modelo}'`;
      }
    }
    return null;
  }

  function responderAsistente(pregunta) {
    const q = pregunta.toLowerCase();
    const prodMencionado = estado.productos.find((p) => {
      const n = p.nombre.toLowerCase();
      return q.includes(n) || (q.includes(p.marca.toLowerCase()) && q.includes(p.formato));
    });
    const confirmados = estado.cierres.filter((c) => c.estado === 'confirmado');
    if (/falt|perd|diferenc/.test(q)) {
      const datos = [];
      confirmados.forEach((c) => c.filas.forEach((f) => {
        if (f.diferencia < 0 && (!prodMencionado || f.id_producto === prodMencionado.id)) {
          datos.push({ id_cierre: c.id, fecha: c.fecha, producto: f.nombre, esperado: f.esperado, contado: f.ajustado ?? f.detectado, diferencia: f.diferencia });
        }
      }));
      if (!datos.length) {
        return { respuesta: 'No encontré faltantes en los cierres confirmados del período consultado.', intencion: 'faltantes_rango', parametros: { producto: prodMencionado ? prodMencionado.nombre : null }, datos: [], sin_datos: true, modelo: null };
      }
      const total = datos.reduce((a, d) => a + d.diferencia, 0);
      const porProd = {};
      datos.forEach((d) => { porProd[d.producto] = (porProd[d.producto] || 0) + d.diferencia; });
      const peor = Object.entries(porProd).sort((a, b) => a[1] - b[1])[0];
      return {
        respuesta: `En los ${confirmados.length} cierres confirmados faltaron ${-total} unidades en total. El producto con más faltantes es ${peor[0]} (${-peor[1]} unidades).`,
        intencion: 'faltantes_rango',
        parametros: { desde: confirmados.length ? confirmados[0].fecha.slice(0, 10) : null, hasta: confirmados.length ? confirmados[confirmados.length - 1].fecha.slice(0, 10) : null, producto: prodMencionado ? prodMencionado.nombre : null },
        datos, sin_datos: false, modelo: null,
      };
    }
    if (/stock|quedan|cu[aá]nt/.test(q) && prodMencionado) {
      const s = stockHasta(isoLocal(new Date())).find((x) => x.id_producto === prodMencionado.id);
      return {
        respuesta: `El stock esperado de ${prodMencionado.nombre} es ${s ? s.stock : 0} unidades.`,
        intencion: 'stock_producto', parametros: { producto: prodMencionado.nombre }, datos: s ? [s] : [], sin_datos: !s, modelo: null,
      };
    }
    if (/cierre|ayer|[uú]ltimo/.test(q) && estado.cierres.length) {
      const c = estado.cierres[estado.cierres.length - 1];
      return {
        respuesta: `El último cierre (#${c.id}, ${c.fecha.slice(0, 10)}) está ${c.estado}: esperado ${c.totales.esperado}, contado ${c.totales.detectado}, diferencia ${c.totales.diferencia} (${c.totales.productos_con_diferencia} productos con diferencia).`,
        intencion: 'detalle_cierre', parametros: { id_cierre: c.id },
        datos: c.filas.map((f) => ({ producto: f.nombre, esperado: f.esperado, detectado: f.detectado, ajustado: f.ajustado, diferencia: f.diferencia })),
        sin_datos: false, modelo: null,
      };
    }
    return {
      respuesta: 'No entendí la pregunta. Probá con: "¿qué faltó esta semana?", "¿cuánto stock queda de Fanta lata?" o "¿cómo salió el último cierre?".',
      intencion: 'desconocida', parametros: {}, datos: [], sin_datos: true, modelo: null,
    };
  }

  // ------------------------------------------------------------ enrutador
  async function atender(metodo, url, json, form) {
    cargar();
    const u = new URL(url, location.origin);
    const q = Object.fromEntries(u.searchParams.entries());
    const ruta = u.pathname.replace(/\/+$/, '');
    const partes = ruta.split('/').filter(Boolean); // ['api', 'productos', '3']
    await esperar(150 + Math.random() * 200); // latencia de red simulada
    if (partes[0] !== 'api') return error(404, 'Not Found');
    const recurso = partes[1];
    const id = partes[2] !== undefined ? Number(partes[2]) : null;

    if (recurso === 'clases' && metodo === 'GET') return resp(200, clon(CLASES));

    if (recurso === 'productos') {
      if (id === null && metodo === 'GET') {
        const todos = q.incluir_inactivos === 'true';
        return resp(200, clon(estado.productos.filter((p) => todos || p.activo)));
      }
      if (id === null && metodo === 'POST') {
        const b = json || {};
        const err = validarProducto(b, false, null);
        if (err) return error(err.startsWith('Ya existe') ? 409 : 422, err);
        const p = {
          id: estado.sig.producto++, nombre: b.nombre.trim(), marca: b.marca.trim(), formato: b.formato,
          precio: Number(b.precio), clase_modelo: b.clase_modelo, activo: b.activo !== false,
        };
        estado.productos.push(p); guardar();
        return resp(201, clon(p));
      }
      const p = estado.productos.find((x) => x.id === id);
      if (!p) return error(404, `Producto ${id} no encontrado`);
      if (metodo === 'GET') return resp(200, clon(p));
      if (metodo === 'PUT') {
        const b = json || {};
        const err = validarProducto(b, true, id);
        if (err) return error(err.startsWith('Ya existe') ? 409 : 422, err);
        ['nombre', 'marca', 'formato', 'clase_modelo', 'activo'].forEach((k) => { if (b[k] !== undefined && b[k] !== null) p[k] = b[k]; });
        if (b.precio !== undefined && b.precio !== null) p.precio = Number(b.precio);
        guardar();
        return resp(200, clon(p));
      }
      if (metodo === 'DELETE') { p.activo = false; guardar(); return resp(204, null); }
    }

    if (recurso === 'movimientos') {
      if (metodo === 'GET') {
        let lista = estado.movimientos.filter((m) => enRango(m.fecha, q.desde, q.hasta));
        if (q.id_producto) lista = lista.filter((m) => m.id_producto === Number(q.id_producto));
        if (q.tipo) lista = lista.filter((m) => m.tipo === q.tipo);
        lista = lista.slice().sort((a, b) => (a.fecha < b.fecha ? 1 : a.fecha > b.fecha ? -1 : b.id - a.id));
        return resp(200, clon(lista));
      }
      if (metodo === 'POST') {
        const b = json || {};
        const p = estado.productos.find((x) => x.id === Number(b.id_producto));
        if (!p) return error(404, `Producto ${b.id_producto} no encontrado`);
        if (!['stock_inicial', 'entrada', 'venta', 'ajuste'].includes(b.tipo)) return error(422, 'tipo: valor no válido');
        const cant = Number(b.cantidad);
        if (!Number.isInteger(cant)) return error(422, 'cantidad: debe ser un entero');
        if (b.tipo === 'ajuste' ? cant === 0 : cant <= 0) {
          return error(422, b.tipo === 'ajuste' ? 'cantidad: un ajuste no puede ser 0' : 'cantidad: debe ser mayor que 0');
        }
        if (b.tipo === 'venta') {
          const s = stockHasta(isoLocal(new Date())).find((x) => x.id_producto === p.id);
          if (s && s.stock - cant < 0) return error(409, `Stock insuficiente de ${p.nombre}: hay ${s.stock}`);
        }
        const m = agregarMovimiento(p.id, b.tipo, cant, b.fecha || new Date(), b.usuario || 'cajero');
        guardar();
        return resp(201, clon(m));
      }
    }

    if (recurso === 'stock' && metodo === 'GET') {
      const limite = q.fecha ? `${q.fecha}T23:59:59` : isoLocal(new Date());
      return resp(200, { fecha: limite, items: stockHasta(limite) });
    }

    if (recurso === 'escaneos' && metodo === 'POST') {
      const imgs = form ? form.getAll('imagenes').filter((f) => f && f.size) : [];
      if (!imgs.length) return error(422, 'Se requiere al menos una imagen');
      if (imgs.length > 5) return error(422, 'Máximo 5 imágenes por escaneo');
      await esperar(1200); // "inferencia"
      const c = crearCierre(new Date(), (form.get('usuario') || 'cajero'), aleatorio(Date.now() & 0xffffffff), imgs.length);
      const anotada = await anotarFoto(c, imgs[0]);
      if (anotada) guardarImagen(c.id, anotada);
      guardar();
      return resp(201, cierreOut(c));
    }

    if (recurso === 'cierres') {
      if (id === null && metodo === 'GET') {
        let lista = estado.cierres.filter((c) => enRango(c.fecha, q.desde, q.hasta));
        if (q.estado) lista = lista.filter((c) => c.estado === q.estado);
        return resp(200, lista.slice().reverse().map(cierreResumen));
      }
      const c = estado.cierres.find((x) => x.id === id);
      if (!c) return error(404, `Cierre ${id} no encontrado`);
      if (partes[3] === 'confirmar' && metodo === 'POST') {
        if (c.estado === 'confirmado') return error(409, `El cierre ${id} ya está confirmado`);
        const b = json || {};
        for (const a of (b.ajustes || [])) {
          if (!c.filas.some((f) => f.id_producto === a.id_producto)) return error(422, `El producto ${a.id_producto} no está en el cierre`);
          if (!Number.isInteger(a.ajustado) || a.ajustado < 0) return error(422, 'ajustado: debe ser un entero ≥ 0');
        }
        confirmar(c, b, null);
        guardar();
        return resp(200, cierreOut(c));
      }
      if (metodo === 'GET' && partes.length === 3) return resp(200, cierreOut(c));
    }

    if (recurso === 'diferencias' && metodo === 'GET') {
      const lista = [];
      estado.cierres.filter((c) => c.estado === 'confirmado' && enRango(c.fecha, q.desde, q.hasta)).forEach((c) => {
        c.filas.forEach((f) => {
          if (q.id_producto && f.id_producto !== Number(q.id_producto)) return;
          lista.push({ id_cierre: c.id, fecha: c.fecha, id_producto: f.id_producto, nombre: f.nombre, esperado: f.esperado, detectado: f.detectado, ajustado: f.ajustado, diferencia: f.diferencia });
        });
      });
      return resp(200, lista);
    }

    if (recurso === 'asistente' && metodo === 'POST') {
      const pregunta = ((json || {}).pregunta || '').trim();
      if (!pregunta) return error(422, 'pregunta: no puede estar vacía');
      if (pregunta.length > 500) return error(422, 'pregunta: máximo 500 caracteres');
      await esperar(600);
      if (pregunta.includes('#503')) return error(503, 'Asistente no configurado (falta ANTHROPIC_API_KEY)');
      return resp(200, responderAsistente(pregunta));
    }

    return error(405, `Ruta simulada no implementada: ${metodo} ${ruta}`);
  }

  window.Simulador = { atender, reiniciar, CLASES };
})();
