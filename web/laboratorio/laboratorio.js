/*
 * Laboratorio de modelos: sube una foto, el servidor corre detector + clasificador y
 * devuelve TODAS las cajas (umbral bajo) con el softmax de cada una. Los umbrales se
 * aplican acá, así moverlos no vuelve a llamar a la API.
 *
 * API: GET /api/laboratorio/modelos · POST /api/laboratorio/analizar · POST /api/laboratorio/clasificar
 * No usa la BD ni el modo simulado del panel.
 */
(function () {
  'use strict';

  const { esc, alerta } = window.Ui;
  const $ = (id) => document.getElementById(id);
  const DESCONOCIDO = 'desconocido';

  // Color por marca; la lata va en un tono más claro que la botella.
  const COLORES = {
    coca_botella: '#d7191c', coca_lata: '#ff6b6b',
    fanta_botella: '#e66a00', fanta_lata: '#ffa94d',
    sprite_botella: '#1b8a3a', sprite_lata: '#5cd65c',
    pepsi_botella: '#1c4fb8', pepsi_lata: '#5c9dff',
    '7up_lata': '#9acd32',
    desconocido: '#e0e0e0', producto: '#ffc107', bottle: '#ffc107', cup: '#ffe08a',
  };
  const colorDe = (clase) => COLORES[clase] || '#ff4fd8';

  const estado = {
    modelos: null,
    umbralDet: 0.25,
    umbralCls: 0.5,
    etiquetas: true,
    archivo: null,      // foto del estante actual
    imagen: null,       // ImageBitmap de esa foto
    resultado: null,    // respuesta de /analizar
    seleccion: null,    // índice de caja elegida
    recortes: [],       // [{archivo, url, resultado|null, error|null}]
  };

  // ------------------------------------------------------------------ preferencias
  function leerPref(clave, defecto) {
    try { const v = localStorage.getItem('lab.' + clave); return v === null ? defecto : JSON.parse(v); } catch { return defecto; }
  }
  function guardarPref(clave, valor) {
    try { localStorage.setItem('lab.' + clave, JSON.stringify(valor)); } catch { /* sin almacenamiento */ }
  }

  // ------------------------------------------------------------------ red
  async function pedir(url, form) {
    const r = await fetch(url, form ? { method: 'POST', body: form } : undefined);
    if (!r.ok) {
      let detalle = '';
      try { detalle = (await r.json()).detail; } catch { /* no era JSON */ }
      throw new Error(typeof detalle === 'string' && detalle ? detalle : `Error HTTP ${r.status}`);
    }
    return r.json();
  }

  function cargando(texto) {
    $('overlay-texto').textContent = texto || 'Cargando…';
    $('overlay').hidden = !texto;
  }

  // ------------------------------------------------------------------ modelos
  const detectorElegido = () => estado.modelos?.detectores.find((d) => d.id === $('sel-detector').value) || null;
  const clasificadorElegido = () => estado.modelos?.clasificadores.find((c) => c.id === $('sel-clasificador').value) || null;

  const TIPOS_DETECTOR = {
    '1clase': 'Detector de 1 clase («producto»): la marca la pone el clasificador.',
    multiclase: 'Detector multiclase (variante B): ya trae la marca. Con un clasificador elegido, manda el clasificador.',
    coco: 'YOLO de COCO sin entrenar: solo se usan sus clases bottle y cup.',
  };

  async function cargarModelos() {
    estado.modelos = await pedir('/api/laboratorio/modelos');
    const { detectores, clasificadores } = estado.modelos;

    const selDet = $('sel-detector');
    selDet.innerHTML = detectores.length
      ? detectores.map((d) => `<option value="${esc(d.id)}">${esc(d.nombre)} · ${esc(d.tipo)}</option>`).join('')
      : '<option value="">(no hay detectores)</option>';
    const selCls = $('sel-clasificador');
    selCls.innerHTML = '<option value="">Ninguno (solo detector)</option>' + clasificadores.map((c) =>
      `<option value="${esc(c.id)}">${esc(c.nombre)} · ${c.clases.length} clases${c.coincide_config ? '' : ' ⚠'}</option>`).join('');

    const det = leerPref('detector', null);
    if (detectores.some((d) => d.id === det)) selDet.value = det;
    const cls = leerPref('clasificador', null);
    if (cls === '' || clasificadores.some((c) => c.id === cls)) selCls.value = cls;
    else if (clasificadores.length) selCls.value = (clasificadores.find((c) => /(^|\/)clasificador\.keras$/.test(c.id)) || clasificadores[0]).id;
    mostrarInfoModelos();

    if (!detectores.length) {
      alerta($('avisos'), 'warning', 'No se encontró ningún detector (<code>modelos/detector/*.pt</code>, ' +
        '<code>runs/detector/*/weights/best.pt</code> o <code>modelos/yolo*.pt</code>). Igual podés usar <b>Solo clasificador</b>.');
    }
  }

  function mostrarInfoModelos() {
    const det = detectorElegido();
    $('info-detector').innerHTML = det ? `${esc(TIPOS_DETECTOR[det.tipo] || '')} <span class="text-nowrap">(${esc(det.fecha)})</span>` : '';
    const cls = clasificadorElegido();
    if (!cls) {
      $('info-clasificador').textContent = 'Sin clasificador: cada caja queda con la clase del detector.';
    } else if (cls.coincide_config) {
      $('info-clasificador').innerHTML = `Clases iguales a <code>config/clases.json</code> (${esc(cls.fecha)}).`;
    } else {
      const sobran = cls.clases.filter((c) => !estado.modelos.clases_config.includes(c));
      $('info-clasificador').innerHTML = `<span class="text-warning-emphasis"><i class="bi bi-exclamation-triangle me-1"></i>` +
        `Entrenado con otras clases${sobran.length ? ' (tiene ' + sobran.map(esc).join(', ') + ')' : ''}: ` +
        `sirve para probar, pero el sistema real lo rechaza.</span> (${esc(cls.fecha)})`;
    }
  }

  // ------------------------------------------------------------------ interpretación de una caja
  function interpretar(caja, clases) {
    const pasaDet = caja.conf_det >= estado.umbralDet;
    if (!caja.probs) return { pasaDet, clase: caja.clase_det, conf: null, orden: [] };
    const orden = caja.probs.map((p, i) => [clases[i], p]).sort((a, b) => b[1] - a[1]);
    const [top, conf] = orden[0];
    return { pasaDet, clase: conf >= estado.umbralCls ? top : DESCONOCIDO, conf, orden };
  }

  function clasesResultado() {
    return estado.resultado?.clasificador?.clases || [];
  }

  // ------------------------------------------------------------------ análisis del estante
  async function analizar() {
    const det = detectorElegido();
    if (!estado.archivo) return;
    if (!det) { alerta($('avisos'), 'warning', 'Elegí un detector.'); return; }
    const form = new FormData();
    form.append('imagen', estado.archivo);
    form.append('detector', det.id);
    form.append('clasificador', $('sel-clasificador').value);
    cargando('Analizando… (la primera vez carga los modelos y tarda más)');
    try {
      estado.resultado = await pedir('/api/laboratorio/analizar', form);
      estado.seleccion = null;
      $('avisos').innerHTML = '';
      $('resultado-estante').hidden = false;
      $('card-detalle').hidden = true;
      dibujarEstante();
    } catch (e) {
      alerta($('avisos'), 'danger', `<b>No se pudo analizar:</b> ${esc(e.message)}`);
    } finally {
      cargando(null);
    }
  }

  async function elegirFotoEstante(archivo) {
    if (!archivo || !archivo.type.startsWith('image/')) return;
    estado.archivo = archivo;
    estado.imagen = await createImageBitmap(archivo, { imageOrientation: 'from-image' });
    await analizar();
  }

  function dibujarEstante() {
    const res = estado.resultado;
    const img = estado.imagen;
    if (!res || !img) return;
    const lienzo = $('lienzo');
    lienzo.width = img.width;
    lienzo.height = img.height;
    const ctx = lienzo.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const escala = img.width / res.ancho;
    const grosor = Math.max(2, Math.round(img.width / 350));
    const fuente = Math.max(12, Math.round(img.width / 55));
    ctx.font = `600 ${fuente}px system-ui, sans-serif`;
    ctx.textBaseline = 'top';

    const clases = clasesResultado();
    const lecturas = res.cajas.map((c) => interpretar(c, clases));
    // Primero las grandes, así las chicas quedan encima y se pueden tocar.
    const orden = res.cajas.map((c, i) => i).sort((a, b) => res.cajas[b].w * res.cajas[b].h - res.cajas[a].w * res.cajas[a].h);
    for (const i of orden) {
      const c = res.cajas[i];
      const l = lecturas[i];
      if (!l.pasaDet) continue;
      const [x, y, w, h] = [c.x * escala, c.y * escala, c.w * escala, c.h * escala];
      const color = colorDe(l.clase);
      ctx.lineWidth = i === estado.seleccion ? grosor * 2.5 : grosor;
      ctx.strokeStyle = color;
      ctx.setLineDash(l.clase === DESCONOCIDO ? [grosor * 3, grosor * 2] : []);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      if (estado.etiquetas || i === estado.seleccion) {
        const texto = `${i + 1} ${l.clase} ${(l.conf ?? c.conf_det).toFixed(2)}`;
        const ancho = ctx.measureText(texto).width + 8;
        const ty = y >= fuente + 6 ? y - fuente - 6 : y;
        ctx.fillStyle = color;
        ctx.fillRect(x, ty, ancho, fuente + 6);
        ctx.fillStyle = '#111';
        ctx.fillText(texto, x + 4, ty + 3);
      }
    }

    // Resumen, conteo y lista
    const visibles = lecturas.filter((l) => l.pasaDet);
    const t = res.tiempos;
    $('resumen-estante').innerHTML = `${esc(estado.archivo.name)} · ${res.ancho}×${res.alto} px · ` +
      `<b>${visibles.length}</b> cajas (de ${res.cajas.length} con conf ≥ 0.05) · detector ${t.deteccion_ms.toFixed(0)} ms` +
      (res.clasificador ? ` · clasificador ${t.clasificacion_ms.toFixed(0)} ms` : '');

    const conteo = new Map();
    (clases.length ? [...clases, DESCONOCIDO] : []).forEach((k) => conteo.set(k, 0));
    visibles.forEach((l) => conteo.set(l.clase, (conteo.get(l.clase) || 0) + 1));
    $('tbody-conteo').innerHTML = [...conteo].map(([k, n]) =>
      `<tr class="${n ? '' : 'text-body-tertiary'}"><td><span class="muestra-color me-2" style="background:${colorDe(k)}"></span>${esc(k)}</td><td class="td-num">${n}</td></tr>`).join('') +
      `<tr class="table-light fw-semibold"><td>Total</td><td class="td-num">${visibles.length}</td></tr>`;

    $('lista-cajas').innerHTML = res.cajas.map((c, i) => {
      const l = lecturas[i];
      const top1 = l.orden[0];
      const top2 = l.orden[1];
      return `<tr data-i="${i}" class="${l.pasaDet ? '' : 'text-body-tertiary'} ${i === estado.seleccion ? 'table-active' : ''}">
        <td>${i + 1}</td>
        <td><span class="muestra-color me-1" style="background:${colorDe(l.clase)}"></span>${esc(l.clase)}</td>
        <td class="td-num">${c.conf_det.toFixed(2)}</td>
        <td>${top1 ? esc(top1[0]) : '<span class="text-body-tertiary">—</span>'}</td>
        <td class="td-num">${top1 ? top1[1].toFixed(2) : ''}</td>
        <td class="small">${top2 ? `${esc(top2[0])} ${top2[1].toFixed(2)}` : ''}</td></tr>`;
    }).join('') || '<tr><td colspan="6" class="text-center text-body-secondary py-3">El detector no encontró nada.</td></tr>';

    if (estado.seleccion !== null) mostrarDetalle(estado.seleccion);
  }

  function barrasProbabilidad(orden, umbral) {
    return orden.map(([clase, p], k) => `
      <div class="d-flex align-items-center gap-2 mb-1 small">
        <div class="text-truncate ${k === 0 ? 'fw-semibold' : ''}" style="width: 7.5rem">${esc(clase)}</div>
        <div class="barra-prob flex-grow-1 position-relative">
          <div style="width:${(p * 100).toFixed(1)}%; background:${colorDe(clase)}"></div>
          <span class="position-absolute top-0 bottom-0 border-start border-dark border-2" style="left:${(umbral * 100).toFixed(1)}%" title="umbral"></span>
        </div>
        <div class="td-num" style="width: 2.8rem">${(p * 100).toFixed(1)}%</div>
      </div>`).join('');
  }

  function mostrarDetalle(i) {
    const res = estado.resultado;
    const c = res.cajas[i];
    const l = interpretar(c, clasesResultado());
    $('card-detalle').hidden = false;
    $('detalle-titulo').innerHTML = `#${i + 1} · <span style="color:${colorDe(l.clase)}">■</span> ${esc(l.clase)}`;

    // Recorte con el mismo margen del 5 % que usa el servidor
    const escala = estado.imagen.width / res.ancho;
    const mx = c.w * 0.05, my = c.h * 0.05;
    const x0 = Math.max(0, (c.x - mx) * escala), y0 = Math.max(0, (c.y - my) * escala);
    const x1 = Math.min(estado.imagen.width, (c.x + c.w + mx) * escala);
    const y1 = Math.min(estado.imagen.height, (c.y + c.h + my) * escala);
    const lr = $('lienzo-recorte');
    lr.width = Math.max(1, Math.round(x1 - x0));
    lr.height = Math.max(1, Math.round(y1 - y0));
    lr.getContext('2d').drawImage(estado.imagen, x0, y0, x1 - x0, y1 - y0, 0, 0, lr.width, lr.height);

    const cabecera = `<div class="small text-body-secondary mb-2">Detector: ${esc(c.clase_det)} ${c.conf_det.toFixed(2)}` +
      `${l.pasaDet ? '' : ' <span class="badge text-bg-secondary">bajo el umbral</span>'} · ${Math.round(c.w)}×${Math.round(c.h)} px</div>`;
    $('detalle-probs').innerHTML = cabecera + (l.orden.length
      ? barrasProbabilidad(l.orden, estado.umbralCls)
      : '<div class="small text-body-secondary">Sin clasificador elegido.</div>');
  }

  function seleccionar(i) {
    estado.seleccion = i;
    dibujarEstante();
    if (i === null) $('card-detalle').hidden = true;
  }

  function cajaEnPunto(evento) {
    const lienzo = $('lienzo');
    const r = lienzo.getBoundingClientRect();
    const escalaVista = lienzo.width / r.width;
    const res = estado.resultado;
    const escala = estado.imagen.width / res.ancho;
    const px = (evento.clientX - r.left) * escalaVista / escala;
    const py = (evento.clientY - r.top) * escalaVista / escala;
    let mejor = null;
    res.cajas.forEach((c, i) => {
      if (c.conf_det < estado.umbralDet) return;
      if (px < c.x || px > c.x + c.w || py < c.y || py > c.y + c.h) return;
      if (mejor === null || c.w * c.h < res.cajas[mejor].w * res.cajas[mejor].h) mejor = i;
    });
    return mejor;
  }

  // ------------------------------------------------------------------ solo clasificador
  async function agregarRecortes(archivos) {
    const cls = clasificadorElegido();
    if (!cls) { alerta($('avisos'), 'warning', 'Elegí un clasificador para esta prueba.'); return; }
    const nuevos = [...archivos].filter((a) => a.type.startsWith('image/'))
      .map((archivo) => ({ archivo, url: URL.createObjectURL(archivo), resultado: null, error: null }));
    estado.recortes.unshift(...nuevos);
    dibujarRecortes();
    await clasificarRecortes(nuevos);
  }

  async function clasificarRecortes(lista) {
    const cls = clasificadorElegido();
    for (const r of lista) {
      r.resultado = null;
      r.error = null;
      if (!cls) { r.error = 'Sin clasificador elegido.'; continue; }
      const form = new FormData();
      form.append('imagen', r.archivo);
      form.append('clasificador', cls.id);
      try {
        r.resultado = await pedir('/api/laboratorio/clasificar', form);
      } catch (e) {
        r.error = e.message;
      }
      dibujarRecortes();
    }
    dibujarRecortes();
  }

  function dibujarRecortes() {
    $('barra-recortes').hidden = !estado.recortes.length;
    $('grilla-recortes').innerHTML = estado.recortes.map((r) => {
      let cuerpo;
      if (r.error) {
        cuerpo = `<div class="text-danger small">${esc(r.error)}</div>`;
      } else if (!r.resultado) {
        cuerpo = '<div class="text-center py-2"><div class="spinner-border spinner-border-sm"></div></div>';
      } else {
        const clases = r.resultado.clasificador.clases;
        const orden = r.resultado.probs.map((p, i) => [clases[i], p]).sort((a, b) => b[1] - a[1]);
        const clase = orden[0][1] >= estado.umbralCls ? orden[0][0] : DESCONOCIDO;
        cuerpo = `<div class="fw-semibold mb-2"><span class="muestra-color me-1" style="background:${colorDe(clase)}"></span>${esc(clase)}
            <span class="small text-body-secondary fw-normal float-end">${r.resultado.tiempo_ms.toFixed(0)} ms</span></div>` +
          barrasProbabilidad(orden.slice(0, 3), estado.umbralCls);
      }
      return `<div class="col-12 col-sm-6 col-xl-4"><div class="card tarjeta-recorte h-100">
          <img src="${r.url}" alt="${esc(r.archivo.name)}">
          <div class="card-body p-2"><div class="small text-truncate text-body-secondary mb-1">${esc(r.archivo.name)}</div>${cuerpo}</div>
        </div></div>`;
    }).join('');
  }

  // ------------------------------------------------------------------ entrada de fotos
  function zonaArrastre(zona, alSoltar) {
    ['dragenter', 'dragover'].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add('arrastrando'); }));
    ['dragleave', 'drop'].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove('arrastrando'); }));
    zona.addEventListener('drop', (e) => alSoltar(e.dataTransfer.files));
  }

  const pestanaRecortesActiva = () => $('tab-recortes').classList.contains('active');

  function iniciarEventos() {
    $('archivo-estante').addEventListener('change', (e) => { elegirFotoEstante(e.target.files[0]); e.target.value = ''; });
    $('archivo-recortes').addEventListener('change', (e) => { agregarRecortes(e.target.files); e.target.value = ''; });
    zonaArrastre($('zona-estante'), (archivos) => elegirFotoEstante(archivos[0]));
    zonaArrastre($('zona-recortes'), agregarRecortes);
    document.addEventListener('paste', (e) => {
      const archivos = [...(e.clipboardData?.files || [])].filter((a) => a.type.startsWith('image/'));
      if (!archivos.length) return;
      e.preventDefault();
      if (pestanaRecortesActiva()) agregarRecortes(archivos);
      else elegirFotoEstante(archivos[0]);
    });

    $('btn-reanalizar').addEventListener('click', analizar);
    $('btn-recargar').addEventListener('click', async () => {
      try { await cargarModelos(); Ui.toast('Lista de modelos actualizada'); } catch (e) { alerta($('avisos'), 'danger', esc(e.message)); }
    });
    $('sel-detector').addEventListener('change', () => { guardarPref('detector', $('sel-detector').value); mostrarInfoModelos(); analizar(); });
    $('sel-clasificador').addEventListener('change', () => {
      guardarPref('clasificador', $('sel-clasificador').value);
      mostrarInfoModelos();
      if (pestanaRecortesActiva()) clasificarRecortes(estado.recortes);
      else analizar();
    });

    $('lienzo').addEventListener('click', (e) => seleccionar(cajaEnPunto(e)));
    $('lista-cajas').addEventListener('click', (e) => {
      const fila = e.target.closest('tr[data-i]');
      if (fila) seleccionar(Number(fila.dataset.i));
    });
    $('btn-cerrar-detalle').addEventListener('click', () => seleccionar(null));
    $('btn-limpiar-recortes').addEventListener('click', () => {
      estado.recortes.forEach((r) => URL.revokeObjectURL(r.url));
      estado.recortes = [];
      dibujarRecortes();
    });

    const rngDet = $('rng-det'), rngCls = $('rng-cls');
    const aplicarUmbrales = () => {
      estado.umbralDet = Number(rngDet.value);
      estado.umbralCls = Number(rngCls.value);
      $('val-det').textContent = estado.umbralDet.toFixed(2);
      $('val-cls').textContent = estado.umbralCls.toFixed(2);
      guardarPref('umbralDet', estado.umbralDet);
      guardarPref('umbralCls', estado.umbralCls);
      dibujarEstante();
      dibujarRecortes();
    };
    rngDet.value = leerPref('umbralDet', 0.25);
    rngCls.value = leerPref('umbralCls', 0.5);
    rngDet.addEventListener('input', aplicarUmbrales);
    rngCls.addEventListener('input', aplicarUmbrales);
    $('chk-etiquetas').addEventListener('change', (e) => { estado.etiquetas = e.target.checked; dibujarEstante(); });
    aplicarUmbrales();
  }

  // ------------------------------------------------------------------ inicio
  (async function () {
    iniciarEventos();
    try {
      await cargarModelos();
    } catch (e) {
      alerta($('avisos'), 'danger', `<b>No se pudo leer la lista de modelos:</b> ${esc(e.message)}. ¿Está corriendo el servidor?`);
    }
  })();
})();
