/*
 * Panel de PC: productos (CRUD), movimientos, stock, cierres, historial de
 * diferencias y asistente. Consume la API del contrato (contratos/api_schemas.py).
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const esc = Ui.esc;
  let productos = [];            // todos (incluye inactivos) para mapear id → nombre
  let clases = null;
  let grafico = null;
  let cierreActual = null;
  let tablaCierre = null;

  const NOMBRE_TIPO = { venta: 'Venta', entrada: 'Entrada', ajuste: 'Ajuste', stock_inicial: 'Stock inicial' };
  const COLOR_TIPO = { venta: 'primary', entrada: 'success', ajuste: 'warning', stock_inicial: 'secondary' };

  function cargando(si, texto) {
    $('overlay').hidden = !si;
    if (texto) $('overlay-texto').textContent = texto;
  }
  function mostrarError(idAvisos, e, prefijo) {
    Ui.alerta($(idAvisos), 'danger', `${prefijo ? `<strong>${esc(prefijo)}</strong> ` : ''}${esc(Ui.textoError(e))}`);
  }
  function limpiarAvisos(idAvisos) { $(idAvisos).innerHTML = ''; }
  const nombreProducto = (id) => (productos.find((p) => p.id === id) || {}).nombre || `#${id}`;
  function filaVacia(cols, texto) { return `<tr><td colspan="${cols}" class="text-center text-body-secondary py-3">${texto}</td></tr>`; }
  function difHtml(d) {
    const cls = d < 0 ? 'text-danger fw-bold' : d > 0 ? 'text-warning-emphasis fw-bold' : 'text-body-secondary';
    return `<span class="${cls}">${Ui.conSigno(d)}</span>`;
  }

  /** Recarga la lista de productos y los <select> que dependen de ella. */
  async function cargarProductos() {
    productos = await Api.get('/api/productos', { incluir_inactivos: true });
    const activos = productos.filter((p) => p.activo);
    const opciones = activos.map((p) => `<option value="${p.id}">${esc(p.nombre)}</option>`).join('');
    $('mov-producto').innerHTML = opciones;
    const todas = '<option value="">Todos</option>' + productos.map((p) => `<option value="${p.id}">${esc(p.nombre)}${p.activo ? '' : ' (baja)'}</option>`).join('');
    ['fm-producto', 'fh-producto'].forEach((id) => {
      const prev = $(id).value;
      $(id).innerHTML = todas;
      $(id).value = prev;
    });
  }

  // ================================================================ PRODUCTOS
  function dibujarProductos() {
    const ver = $('chk-inactivos').checked;
    const lista = productos.filter((p) => ver || p.activo);
    $('tbody-productos').innerHTML = lista.length ? lista.map((p) => `
      <tr class="${p.activo ? '' : 'text-body-secondary'}">
        <td>${p.id}</td>
        <td class="fw-semibold">${esc(p.nombre)}</td>
        <td>${esc(p.marca)}</td>
        <td>${esc(p.formato)}</td>
        <td class="td-num">${Ui.moneda(p.precio)}</td>
        <td><code>${esc(p.clase_modelo)}</code></td>
        <td>${p.activo ? '<span class="badge text-bg-success">activo</span>' : '<span class="badge text-bg-secondary">baja</span>'}</td>
        <td class="text-end text-nowrap">
          <button type="button" class="btn btn-sm btn-outline-primary" data-editar="${p.id}" title="Editar"><i class="bi bi-pencil"></i><span class="d-none d-lg-inline ms-1">Editar</span></button>
          ${p.activo
            ? `<button type="button" class="btn btn-sm btn-outline-danger" data-baja="${p.id}" title="Dar de baja"><i class="bi bi-archive"></i><span class="d-none d-lg-inline ms-1">Baja</span></button>`
            : `<button type="button" class="btn btn-sm btn-outline-success" data-reactivar="${p.id}" title="Reactivar"><i class="bi bi-arrow-counterclockwise"></i><span class="d-none d-lg-inline ms-1">Reactivar</span></button>`}
        </td>
      </tr>`).join('') : filaVacia(8, 'No hay productos');
  }

  async function refrescarProductos() {
    try {
      await cargarProductos();
      dibujarProductos();
    } catch (e) { mostrarError('avisos-productos', e, 'No se pudieron cargar los productos.'); }
  }

  let modalProducto;
  function abrirModalProducto(p) {
    limpiarAvisos('avisos-modal-producto');
    $('titulo-modal-producto').textContent = p ? `Editar producto #${p.id}` : 'Nuevo producto';
    $('prod-id').value = p ? p.id : '';
    const usadas = new Set(productos.filter((x) => !p || x.id !== p.id).map((x) => x.clase_modelo));
    const lista = (clases && clases.clases) || [];
    $('prod-clase').innerHTML = lista.map((c) =>
      `<option value="${esc(c.clase)}" ${usadas.has(c.clase) ? 'disabled' : ''}>${esc(c.clase)} — ${esc(c.nombre)}${usadas.has(c.clase) ? ' (en uso)' : ''}</option>`).join('');
    if (p) {
      if (!lista.some((c) => c.clase === p.clase_modelo)) {
        $('prod-clase').insertAdjacentHTML('afterbegin', `<option value="${esc(p.clase_modelo)}">${esc(p.clase_modelo)}</option>`);
      }
      $('prod-clase').value = p.clase_modelo;
      $('prod-nombre').value = p.nombre;
      $('prod-marca').value = p.marca;
      $('prod-formato').value = p.formato;
      $('prod-precio').value = Number(p.precio).toFixed(2);
      $('prod-activo').checked = p.activo;
    } else {
      const libre = lista.find((c) => !usadas.has(c.clase));
      $('prod-clase').value = libre ? libre.clase : '';
      if (!libre) {
        Ui.alerta($('avisos-modal-producto'), 'warning',
          'Todas las clases de <code>config/clases.json</code> ya tienen un producto (incluidos los dados de baja). ' +
          'Para otro producto hay que agregar la clase al modelo y reentrenar, o editar/reactivar uno existente.');
      }
      autocompletarDesdeClase();
      $('prod-precio').value = '';
      $('prod-activo').checked = true;
    }
    modalProducto.show();
  }

  /** Al elegir una clase en un producto nuevo, propone nombre/marca/formato de clases.json. */
  function autocompletarDesdeClase() {
    if ($('prod-id').value) return;
    const c = ((clases && clases.clases) || []).find((x) => x.clase === $('prod-clase').value);
    if (!c) return;
    $('prod-nombre').value = c.nombre;
    $('prod-marca').value = c.marca;
    $('prod-formato').value = c.formato;
  }

  async function guardarProducto(ev) {
    ev.preventDefault();
    limpiarAvisos('avisos-modal-producto');
    const id = $('prod-id').value;
    const datos = {
      nombre: $('prod-nombre').value.trim(),
      marca: $('prod-marca').value.trim(),
      formato: $('prod-formato').value,
      precio: $('prod-precio').value === '' ? null : Number($('prod-precio').value),
      clase_modelo: $('prod-clase').value,
      activo: $('prod-activo').checked,
    };
    const faltan = [];
    if (!datos.nombre) faltan.push('nombre');
    if (!datos.marca) faltan.push('marca');
    if (datos.precio === null || Number.isNaN(datos.precio) || datos.precio < 0) faltan.push('precio (≥ 0)');
    if (!datos.clase_modelo) faltan.push('clase del modelo');
    if (faltan.length) { Ui.alerta($('avisos-modal-producto'), 'warning', 'Revisá: ' + esc(faltan.join(', '))); return; }
    datos.precio = Number(datos.precio.toFixed(2));
    $('btn-guardar-producto').disabled = true;
    try {
      if (id) await Api.put(`/api/productos/${id}`, datos);
      else await Api.post('/api/productos', datos);
      modalProducto.hide();
      Ui.toast(id ? 'Producto actualizado' : 'Producto creado');
      await refrescarProductos();
    } catch (e) {
      Ui.alerta($('avisos-modal-producto'), 'danger', esc(Ui.textoError(e)));
    } finally {
      $('btn-guardar-producto').disabled = false;
    }
  }

  async function accionProducto(ev) {
    const b = ev.target.closest('button');
    if (!b) return;
    limpiarAvisos('avisos-productos');
    if (b.dataset.editar) {
      abrirModalProducto(productos.find((p) => p.id === Number(b.dataset.editar)));
    } else if (b.dataset.baja) {
      const p = productos.find((x) => x.id === Number(b.dataset.baja));
      if (!confirm(`¿Dar de baja "${p.nombre}"? No se borra: deja de aparecer en el stock y en los conteos.`)) return;
      try { await Api.del(`/api/productos/${p.id}`); Ui.toast('Producto dado de baja'); await refrescarProductos(); } catch (e) { mostrarError('avisos-productos', e, 'No se pudo dar de baja.'); }
    } else if (b.dataset.reactivar) {
      try { await Api.put(`/api/productos/${b.dataset.reactivar}`, { activo: true }); Ui.toast('Producto reactivado'); await refrescarProductos(); } catch (e) { mostrarError('avisos-productos', e, 'No se pudo reactivar.'); }
    }
  }

  // ================================================================ MOVIMIENTOS
  function ayudaMovimiento() {
    const ajuste = $('mov-tipo').value === 'ajuste';
    $('mov-cantidad').min = ajuste ? '' : '1';
    $('mov-ayuda').textContent = ajuste
      ? 'Ajuste: cantidad con signo, distinta de 0 (negativa = se descuenta del stock, positiva = se suma).'
      : 'La cantidad debe ser mayor que 0.';
  }

  async function registrarMovimiento(ev) {
    ev.preventDefault();
    limpiarAvisos('avisos-movimientos');
    const tipo = $('mov-tipo').value;
    const cantidad = Number($('mov-cantidad').value);
    if (!$('mov-producto').value) { Ui.alerta($('avisos-movimientos'), 'warning', 'Elegí un producto.'); return; }
    if (!Number.isInteger(cantidad) || (tipo === 'ajuste' ? cantidad === 0 : cantidad <= 0)) {
      Ui.alerta($('avisos-movimientos'), 'warning', tipo === 'ajuste' ? 'Un ajuste debe ser un entero distinto de 0.' : 'La cantidad debe ser un entero mayor que 0.');
      return;
    }
    const cuerpo = { id_producto: Number($('mov-producto').value), tipo, cantidad, usuario: $('mov-usuario').value.trim() || 'cajero' };
    if ($('mov-fecha').value) cuerpo.fecha = $('mov-fecha').value.length === 16 ? $('mov-fecha').value + ':00' : $('mov-fecha').value;
    try {
      const m = await Api.post('/api/movimientos', cuerpo);
      Ui.toast(`${NOMBRE_TIPO[m.tipo]} registrada: ${nombreProducto(m.id_producto)} (${Ui.conSigno(m.cantidad)})`);
      $('mov-cantidad').value = tipo === 'ajuste' ? '-1' : '1';
      $('mov-fecha').value = '';
      await listarMovimientos();
    } catch (e) { mostrarError('avisos-movimientos', e, 'No se pudo registrar.'); }
  }

  async function listarMovimientos(ev) {
    if (ev) ev.preventDefault();
    try {
      const lista = await Api.get('/api/movimientos', {
        desde: $('fm-desde').value, hasta: $('fm-hasta').value, id_producto: $('fm-producto').value, tipo: $('fm-tipo').value,
      });
      $('tbody-movimientos').innerHTML = lista.length ? lista.map((m) => `
        <tr>
          <td>${m.id}</td>
          <td class="text-nowrap">${Ui.fecha(m.fecha)}</td>
          <td>${esc(nombreProducto(m.id_producto))}</td>
          <td><span class="badge text-bg-${COLOR_TIPO[m.tipo] || 'light'}">${esc(NOMBRE_TIPO[m.tipo] || m.tipo)}</span></td>
          <td class="td-num">${m.tipo === 'venta' ? '−' + m.cantidad : m.tipo === 'ajuste' ? Ui.conSigno(m.cantidad) : '+' + m.cantidad}</td>
          <td>${esc(m.usuario)}</td>
          <td>${m.id_cierre ? `<a href="#cierres" data-ir-cierre="${m.id_cierre}">#${m.id_cierre}</a>` : ''}</td>
        </tr>`).join('') : filaVacia(7, 'Sin movimientos para esos filtros');
      $('pie-movimientos').textContent = `${lista.length} movimiento(s)`;
    } catch (e) { mostrarError('avisos-movimientos', e, 'No se pudieron listar los movimientos.'); }
  }

  // ================================================================ STOCK
  async function consultarStock(ev) {
    if (ev) ev.preventDefault();
    limpiarAvisos('avisos-stock');
    try {
      const s = await Api.get('/api/stock', { fecha: $('stock-fecha').value });
      $('stock-titulo').textContent = `Stock esperado al ${Ui.fecha(s.fecha)}`;
      const t = { stock_inicial: 0, entradas: 0, ventas: 0, ajustes: 0, stock: 0 };
      $('tbody-stock').innerHTML = s.items.length ? s.items.map((i) => {
        Object.keys(t).forEach((k) => { t[k] += i[k]; });
        return `<tr>
          <td class="fw-semibold">${esc(i.nombre)}</td><td><code>${esc(i.clase_modelo)}</code></td>
          <td class="td-num">${i.stock_inicial}</td><td class="td-num">${i.entradas}</td><td class="td-num">${i.ventas}</td>
          <td class="td-num">${Ui.conSigno(i.ajustes)}</td>
          <td class="td-num fw-bold ${i.stock <= 0 ? 'text-danger' : ''}">${i.stock}</td>
        </tr>`;
      }).join('') : filaVacia(7, 'Sin productos');
      $('tfoot-stock').innerHTML = `<tr><td colspan="2">Totales</td><td class="td-num">${t.stock_inicial}</td><td class="td-num">${t.entradas}</td><td class="td-num">${t.ventas}</td><td class="td-num">${Ui.conSigno(t.ajustes)}</td><td class="td-num">${t.stock}</td></tr>`;
    } catch (e) { mostrarError('avisos-stock', e, 'No se pudo consultar el stock.'); }
  }

  // ================================================================ CIERRES
  async function listarCierres(ev) {
    if (ev) ev.preventDefault();
    try {
      const lista = await Api.get('/api/cierres', { desde: $('fc-desde').value, hasta: $('fc-hasta').value, estado: $('fc-estado').value });
      $('tbody-cierres').innerHTML = lista.length ? lista.map((c) => `
        <tr data-cierre="${c.id}" class="${c.estado === 'pendiente' ? 'table-warning' : ''}">
          <td>${c.id}</td><td class="text-nowrap">${Ui.fecha(c.fecha)}</td><td>${esc(c.usuario)}</td>
          <td>${Ui.badgeEstado(c.estado)}</td><td>${esc(c.variante_modelo)}</td>
          <td class="td-num">${c.totales.esperado}</td><td class="td-num">${c.totales.detectado}</td>
          <td class="td-num">${difHtml(c.totales.diferencia)}</td><td class="td-num">${c.totales.productos_con_diferencia}</td>
        </tr>`).join('') : filaVacia(9, 'Sin cierres');
    } catch (e) { mostrarError('avisos-cierres', e, 'No se pudieron listar los cierres.'); }
  }

  function verListaCierres() {
    $('vista-detalle-cierre').classList.add('d-none');
    $('vista-lista-cierres').classList.remove('d-none');
    cierreActual = null;
    listarCierres();
  }

  async function abrirCierre(id) {
    limpiarAvisos('avisos-cierres');
    cargando(true, 'Cargando cierre…');
    try {
      mostrarCierre(await Api.get(`/api/cierres/${id}`));
    } catch (e) { mostrarError('avisos-cierres', e, `No se pudo abrir el cierre #${id}.`); }
    finally { cargando(false); }
  }

  function mostrarCierre(c) {
    cierreActual = c;
    $('vista-lista-cierres').classList.add('d-none');
    $('vista-detalle-cierre').classList.remove('d-none');
    $('dc-titulo').textContent = `Cierre #${c.id}`;
    $('dc-meta').textContent = `${Ui.fecha(c.fecha)} · ${c.usuario} · modelo ${c.variante_modelo}` + (c.tiempo_ms ? ` · ${(c.tiempo_ms / 1000).toFixed(1)} s · ${(c.cajas || []).length} cajas` : '');
    $('dc-estado').innerHTML = Ui.badgeEstado(c.estado);
    const url = Api.urlImagen(c.imagen_anotada_url || (c.imagenes_urls || [])[0]);
    $('dc-imagen').classList.toggle('d-none', !url);
    if (url) $('dc-imagen').src = url; else $('dc-imagen').removeAttribute('src');
    $('dc-originales').innerHTML = (c.imagenes_urls || []).map((u, i) =>
      `<button type="button" class="btn btn-sm btn-outline-secondary" data-original="${esc(u)}"><i class="bi bi-image me-1"></i>Foto ${i + 1}</button>`).join('');
    $('dc-no-reconocidos').innerHTML = c.no_reconocidos > 0
      ? `<div class="alert alert-warning py-1 small mb-0"><i class="bi bi-question-octagon me-1"></i>${c.no_reconocidos} objeto(s) no reconocido(s).</div>` : '';
    const pendiente = c.estado === 'pendiente';
    tablaCierre = Ui.tablaConciliacion($('dc-tabla'), c, {
      editable: pendiente,
      alCambiar: ({ invalidos }) => { $('btn-dc-confirmar').disabled = invalidos > 0; },
    });
    $('dc-confirmar').classList.toggle('d-none', !pendiente);
    $('dc-observacion').value = '';
    $('dc-observacion-ver').innerHTML = !pendiente && c.observacion ? `<strong>Observación:</strong> ${esc(c.observacion)}` : '';
  }

  async function confirmarCierreActual() {
    if (!cierreActual || !tablaCierre) return;
    if (!tablaCierre.valido()) { Ui.toast('Hay cantidades inválidas', 'danger'); return; }
    if (!confirm(`¿Confirmar el cierre #${cierreActual.id}? Se generarán movimientos de ajuste por las diferencias.`)) return;
    cargando(true, 'Confirmando…');
    try {
      const c = await Api.post(`/api/cierres/${cierreActual.id}/confirmar`, {
        ajustes: tablaCierre.ajustes(),
        observacion: $('dc-observacion').value.trim() || null,
        usuario: 'cajero',
      });
      mostrarCierre(c);
      Ui.toast('Cierre confirmado');
    } catch (e) { mostrarError('avisos-cierres', e, 'No se pudo confirmar.'); }
    finally { cargando(false); }
  }

  // ================================================================ HISTORIAL
  const PALETA = ['#d7191c', '#2c7bb6', '#1a9850', '#f28e1c', '#7b3294', '#008080', '#8c510a', '#e7298a', '#666666', '#4daf4a'];

  async function cargarHistorial(ev) {
    if (ev) ev.preventDefault();
    limpiarAvisos('avisos-historial');
    try {
      const lista = await Api.get('/api/diferencias', { id_producto: $('fh-producto').value, desde: $('fh-desde').value, hasta: $('fh-hasta').value });
      lista.sort((a, b) => (a.fecha < b.fecha ? -1 : a.fecha > b.fecha ? 1 : a.id_producto - b.id_producto));
      $('tbody-historial').innerHTML = lista.length ? lista.slice().reverse().map((d) => `
        <tr class="${d.diferencia < 0 ? 'table-danger' : d.diferencia > 0 ? 'table-warning' : ''}">
          <td><a href="#cierres" data-ir-cierre="${d.id_cierre}">#${d.id_cierre}</a></td>
          <td class="text-nowrap">${Ui.fecha(d.fecha)}</td><td>${esc(d.nombre)}</td>
          <td class="td-num">${d.esperado}</td><td class="td-num">${d.detectado}</td><td class="td-num">${d.ajustado ?? '—'}</td>
          <td class="td-num">${difHtml(d.diferencia)}</td>
        </tr>`).join('') : filaVacia(7, 'Sin diferencias registradas (solo cuentan los cierres confirmados)');

      const resumen = new Map();
      lista.forEach((d) => {
        const r = resumen.get(d.id_producto) || { nombre: d.nombre, cierres: 0, falt: 0, sobr: 0, neto: 0 };
        r.cierres++; r.neto += d.diferencia;
        if (d.diferencia < 0) r.falt -= d.diferencia; else r.sobr += d.diferencia;
        resumen.set(d.id_producto, r);
      });
      const filasRes = [...resumen.values()].sort((a, b) => a.neto - b.neto);
      $('tbody-resumen-historial').innerHTML = filasRes.length ? filasRes.map((r) => `
        <tr><td>${esc(r.nombre)}</td><td class="td-num">${r.cierres}</td><td class="td-num text-danger">${r.falt}</td>
        <td class="td-num text-warning-emphasis">${r.sobr}</td><td class="td-num">${difHtml(r.neto)}</td></tr>`).join('') : filaVacia(5, '—');

      dibujarGrafico(lista);
    } catch (e) { mostrarError('avisos-historial', e, 'No se pudo cargar el historial.'); }
  }

  function dibujarGrafico(lista) {
    if (typeof window.Chart === 'undefined') {
      $('grafico-no-disponible').classList.remove('d-none');
      $('grafico-historial').classList.add('d-none');
      return;
    }
    // Eje X: un punto por cierre (ordenado por fecha). Una serie por producto.
    const cierres = [];
    lista.forEach((d) => { if (!cierres.some((c) => c.id === d.id_cierre)) cierres.push({ id: d.id_cierre, fecha: d.fecha }); });
    const etiquetas = cierres.map((c) => `#${c.id} ${Ui.soloFecha(c.fecha)}`);
    const porProducto = new Map();
    lista.forEach((d) => {
      if (!porProducto.has(d.id_producto)) porProducto.set(d.id_producto, { nombre: d.nombre, valores: new Map() });
      porProducto.get(d.id_producto).valores.set(d.id_cierre, d.diferencia);
    });
    const series = [...porProducto.values()].map((s, i) => ({
      label: s.nombre,
      data: cierres.map((c) => (s.valores.has(c.id) ? s.valores.get(c.id) : null)),
      borderColor: PALETA[i % PALETA.length],
      backgroundColor: PALETA[i % PALETA.length],
      tension: 0, spanGaps: true, pointRadius: 4,
    }));
    if (grafico) grafico.destroy();
    grafico = new Chart($('grafico-historial'), {
      type: series.length === 1 ? 'bar' : 'line',
      data: { labels: etiquetas, datasets: series },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } }, tooltip: { mode: 'index', intersect: false } },
        scales: { y: { title: { display: true, text: 'Diferencia (u.)' }, ticks: { precision: 0 } } },
      },
    });
  }

  // ================================================================ ASISTENTE
  function agregarBurbuja(html, clase) {
    const div = document.createElement('div');
    div.className = `burbuja ${clase}`;
    div.innerHTML = html;
    $('chat').appendChild(div);
    $('chat').scrollTop = $('chat').scrollHeight;
    return div;
  }

  function tablaDatos(datos) {
    if (!datos.length) return '<div class="small text-body-secondary">Sin registros.</div>';
    const cols = [...new Set(datos.flatMap((d) => Object.keys(d)))];
    const celda = (v) => (v !== null && typeof v === 'object' ? esc(JSON.stringify(v)) : esc(v ?? '—'));
    return `<div class="table-responsive"><table class="table table-sm table-bordered small mb-0">
      <thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${datos.map((d) => `<tr>${cols.map((c) => `<td>${celda(d[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  async function preguntar(ev) {
    if (ev) ev.preventDefault();
    const pregunta = $('input-pregunta').value.trim();
    if (!pregunta) return;
    agregarBurbuja(esc(pregunta), 'burbuja-usuario');
    $('input-pregunta').value = '';
    $('btn-preguntar').disabled = true;
    const esperando = agregarBurbuja('<span class="spinner-border spinner-border-sm me-2"></span>Pensando…', 'burbuja-asistente');
    try {
      const r = await Api.post('/api/asistente', { pregunta });
      const params = Object.keys(r.parametros || {}).length
        ? `<div class="small mb-1"><strong>Parámetros:</strong> <code>${esc(JSON.stringify(r.parametros))}</code></div>` : '';
      esperando.innerHTML = `<div style="white-space:pre-wrap">${esc(r.respuesta)}</div>
        <div class="small text-body-secondary mt-1">
          <span class="badge text-bg-light border">${esc(r.intencion)}</span>
          ${r.sin_datos ? '<span class="badge text-bg-secondary">sin datos</span>' : ''}
          <span>${r.modelo ? 'Modelo: ' + esc(r.modelo) : 'Respuesta por plantilla'}</span>
        </div>
        <details class="mt-1"><summary class="small">Datos usados (${(r.datos || []).length})</summary>
          <div class="mt-1">${params}${tablaDatos(r.datos || [])}</div></details>`;
    } catch (e) {
      esperando.classList.add('burbuja-error');
      esperando.innerHTML = e.status === 503
        ? '<strong>El asistente no está disponible.</strong> <span class="small">Probá más tarde o revisá la configuración del servidor.</span>'
        : `<strong>Error:</strong> ${esc(Ui.textoError(e))}`;
    } finally {
      $('btn-preguntar').disabled = false;
      $('chat').scrollTop = $('chat').scrollHeight;
      $('input-pregunta').focus();
    }
  }

  // ================================================================ navegación
  const CARGA_PESTANA = {
    productos: refrescarProductos,
    movimientos: listarMovimientos,
    stock: consultarStock,
    cierres: () => { if (!cierreActual) listarCierres(); },
    historial: cargarHistorial,
    asistente: () => $('input-pregunta').focus(),
  };

  function irAPestana(nombre) {
    const btn = document.querySelector(`#pestanas [data-hash="${nombre}"]`);
    if (btn) bootstrap.Tab.getOrCreateInstance(btn).show();
  }

  function enlazar() {
    modalProducto = new bootstrap.Modal($('modal-producto'));
    document.querySelectorAll('#pestanas [data-bs-toggle="tab"]').forEach((btn) => {
      btn.addEventListener('shown.bs.tab', () => {
        const h = btn.dataset.hash;
        if (location.hash !== '#' + h) history.replaceState(null, '', location.pathname + location.search + '#' + h);
        CARGA_PESTANA[h]();
      });
    });
    // Productos
    $('btn-nuevo-producto').addEventListener('click', () => abrirModalProducto(null));
    $('chk-inactivos').addEventListener('change', dibujarProductos);
    $('tbody-productos').addEventListener('click', accionProducto);
    $('form-producto').addEventListener('submit', guardarProducto);
    $('prod-clase').addEventListener('change', autocompletarDesdeClase);
    // Movimientos
    $('mov-tipo').addEventListener('change', ayudaMovimiento);
    $('form-movimiento').addEventListener('submit', registrarMovimiento);
    $('filtros-movimientos').addEventListener('submit', listarMovimientos);
    $('filtros-movimientos').addEventListener('reset', () => setTimeout(listarMovimientos));
    // Stock
    $('form-stock').addEventListener('submit', consultarStock);
    $('btn-stock-ahora').addEventListener('click', () => { $('stock-fecha').value = ''; consultarStock(); });
    // Cierres
    $('filtros-cierres').addEventListener('submit', listarCierres);
    $('filtros-cierres').addEventListener('reset', () => setTimeout(listarCierres));
    $('tbody-cierres').addEventListener('click', (ev) => {
      const tr = ev.target.closest('tr[data-cierre]');
      if (tr) abrirCierre(Number(tr.dataset.cierre));
    });
    $('btn-volver-cierres').addEventListener('click', verListaCierres);
    $('btn-dc-confirmar').addEventListener('click', confirmarCierreActual);
    $('dc-imagen').addEventListener('click', () => Ui.verImagen($('dc-imagen').src, cierreActual ? `Cierre #${cierreActual.id}` : 'Imagen'));
    $('dc-originales').addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-original]');
      if (b) Ui.verImagen(b.dataset.original, 'Foto original');
    });
    // Enlaces a un cierre desde movimientos / historial
    document.addEventListener('click', (ev) => {
      const a = ev.target.closest('[data-ir-cierre]');
      if (!a) return;
      ev.preventDefault();
      cierreActual = { id: Number(a.dataset.irCierre) }; // evita que la pestaña recargue la lista
      irAPestana('cierres');
      abrirCierre(Number(a.dataset.irCierre));
    });
    // Historial
    $('filtros-historial').addEventListener('submit', cargarHistorial);
    $('filtros-historial').addEventListener('reset', () => setTimeout(cargarHistorial));
    // Asistente
    $('form-asistente').addEventListener('submit', preguntar);
    $('sugerencias').addEventListener('click', (ev) => {
      const b = ev.target.closest('button');
      if (!b) return;
      $('input-pregunta').value = b.textContent;
      preguntar();
    });
  }

  async function iniciar() {
    enlazar();
    await Api.iniciar();
    Ui.bannerModo($('banner-modo'));
    try {
      clases = await Api.clases();
    } catch (e) { mostrarError('avisos-productos', e, 'No se pudo leer /api/clases.'); }
    const inicial = (location.hash || '#productos').slice(1);
    if (inicial !== 'productos' && CARGA_PESTANA[inicial]) {
      try { await cargarProductos(); } catch (e) { /* se reintenta en la pestaña */ }
      irAPestana(inicial);
    } else {
      refrescarProductos();
    }
  }

  iniciar();
})();
