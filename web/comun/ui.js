/*
 * Utilidades de interfaz compartidas por el escáner y el panel:
 * formato, avisos, banner de modo simulado, visor de imagen con zoom y la
 * tabla de conciliación editable (esperado / detectado / diferencia).
 * Requiere Bootstrap 5 (bundle) cargado antes.
 */
(function () {
  'use strict';

  function esc(v) {
    return String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function fecha(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return esc(iso);
    return d.toLocaleString('es-BO', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function soloFecha(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return esc(iso);
    return d.toLocaleDateString('es-BO', { day: '2-digit', month: '2-digit' });
  }

  function hoyIso() {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  function conSigno(n) { return n > 0 ? '+' + n : String(n); }

  function moneda(v) {
    const n = Number(v);
    return Number.isNaN(n) ? esc(v) : 'Bs ' + n.toFixed(2);
  }

  /** Mensaje de error legible a partir de una excepción de Api. */
  function textoError(e) {
    if (e && e.detail) return e.status ? `${e.detail} (HTTP ${e.status})` : e.detail;
    return (e && e.message) || 'Error inesperado';
  }

  /** Alerta Bootstrap dentro de un contenedor (reemplaza su contenido). */
  function alerta(contenedor, tipo, html) {
    contenedor.innerHTML = `<div class="alert alert-${tipo} alert-dismissible fade show mb-2" role="alert">${html}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Cerrar"></button></div>`;
  }

  /** Toast flotante (no requiere contenedor). */
  function toast(msg, tipo = 'success') {
    let zona = document.getElementById('zona-toasts');
    if (!zona) {
      zona = document.createElement('div');
      zona.id = 'zona-toasts';
      zona.className = 'toast-container position-fixed bottom-0 end-0 p-3';
      zona.style.zIndex = 1100;
      document.body.appendChild(zona);
    }
    const el = document.createElement('div');
    el.className = `toast align-items-center text-bg-${tipo} border-0`;
    el.setAttribute('role', 'status');
    el.innerHTML = `<div class="d-flex"><div class="toast-body">${esc(msg)}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Cerrar"></button></div>`;
    zona.appendChild(el);
    const t = new bootstrap.Toast(el, { delay: tipo === 'danger' ? 7000 : 3500 });
    el.addEventListener('hidden.bs.toast', () => el.remove());
    t.show();
  }

  /** Muestra un aviso discreto si se está en modo simulado. */
  function bannerModo(contenedor) {
    if (Api.modo !== 'simulado') { contenedor.innerHTML = ''; return; }
    contenedor.innerHTML = `<div class="alert alert-info py-1 px-2 small d-flex align-items-center gap-2 mb-2 banner-simulado">
      <i class="bi bi-cone-striped"></i>
      <span class="flex-grow-1"><strong>Modo simulado</strong> — ${esc(Api.motivoSimulado)}. Los datos son de ejemplo.</span>
      <button type="button" class="btn btn-sm btn-outline-secondary py-0" id="btn-reiniciar-simulado" title="Borra los datos simulados guardados en este navegador">Reiniciar datos</button>
    </div>`;
    contenedor.querySelector('#btn-reiniciar-simulado').addEventListener('click', () => {
      if (!confirm('¿Borrar los datos simulados de este navegador y volver a los de ejemplo?')) return;
      window.Simulador.reiniciar();
      location.reload();
    });
  }

  // ------------------------------------------------------------ visor con zoom
  let visor = null;
  function verImagen(url, titulo) {
    if (!url) return;
    if (!visor) {
      const el = document.createElement('div');
      el.className = 'modal fade';
      el.id = 'modal-visor';
      el.tabIndex = -1;
      el.innerHTML = `<div class="modal-dialog modal-fullscreen"><div class="modal-content bg-dark text-white">
        <div class="modal-header border-secondary py-2">
          <h2 class="modal-title fs-6" id="visor-titulo"></h2>
          <span class="small text-secondary ms-3 d-none d-sm-inline">Tocá la imagen para acercar / alejar</span>
          <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Cerrar"></button>
        </div>
        <div class="modal-body p-0 visor-cuerpo"><img id="visor-img" alt="Imagen ampliada"></div>
      </div></div>`;
      document.body.appendChild(el);
      const img = el.querySelector('#visor-img');
      img.addEventListener('click', (ev) => {
        const cuerpo = el.querySelector('.visor-cuerpo');
        const acercar = !img.classList.contains('zoom');
        // Mantiene el punto tocado bajo el dedo al acercar.
        const rx = ev.offsetX / img.clientWidth, ry = ev.offsetY / img.clientHeight;
        img.classList.toggle('zoom', acercar);
        if (acercar) {
          cuerpo.scrollLeft = rx * img.clientWidth - cuerpo.clientWidth / 2;
          cuerpo.scrollTop = ry * img.clientHeight - cuerpo.clientHeight / 2;
        }
      });
      visor = { el, img, modal: new bootstrap.Modal(el) };
    }
    visor.img.classList.remove('zoom');
    visor.img.src = url;
    visor.el.querySelector('#visor-titulo').textContent = titulo || 'Imagen';
    visor.modal.show();
  }

  // ------------------------------------------------------------ tabla de conciliación
  /**
   * Dibuja la tabla de un CierreOut. Si editable, la columna "detectado" es un
   * input numérico (≥ 0) y la diferencia/estado/totales se recalculan en vivo.
   * Devuelve { ajustes(): AjusteFila[], valido(): boolean }.
   */
  function tablaConciliacion(contenedor, cierre, opciones = {}) {
    const editable = !!opciones.editable;
    const filas = cierre.filas || [];
    const confirmado = cierre.estado === 'confirmado';
    const colConteo = confirmado ? 'Contado' : 'Detectado';

    const cuerpo = filas.map((f, i) => {
      const valor = f.ajustado ?? f.detectado;
      const corregido = f.ajustado !== null && f.ajustado !== undefined && f.ajustado !== f.detectado;
      const celdaConteo = editable
        ? `<input type="number" class="form-control form-control-sm input-conteo text-end" min="0" step="1" inputmode="numeric"
             value="${valor}" data-i="${i}" aria-label="Cantidad contada de ${esc(f.nombre)}">
           <div class="nota-modelo small text-body-secondary d-none">modelo: ${f.detectado}</div>`
        : `${valor}${corregido ? `<div class="small text-body-secondary">modelo: ${f.detectado}</div>` : ''}`;
      const conf = f.confianza_media != null ? `<div class="small text-body-secondary">conf. ${(f.confianza_media * 100).toFixed(0)}%</div>` : '';
      return `<tr data-i="${i}">
        <td><div class="fw-semibold">${esc(f.nombre)}</div>${conf}</td>
        <td class="text-end">${f.esperado}</td>
        <td class="text-end celda-conteo">${celdaConteo}</td>
        <td class="text-end celda-dif"></td>
      </tr>`;
    }).join('');

    contenedor.innerHTML = `<div class="table-responsive">
      <table class="table table-sm align-middle tabla-conciliacion mb-1">
        <thead><tr><th>Producto</th><th class="text-end">Esperado</th><th class="text-end" style="min-width:5.5rem">${colConteo}</th><th class="text-end">Dif.</th></tr></thead>
        <tbody>${cuerpo || '<tr><td colspan="4" class="text-center text-body-secondary">Sin filas</td></tr>'}</tbody>
        <tfoot class="table-group-divider"><tr class="fw-bold">
          <td>Totales</td><td class="text-end" id="tot-esp"></td><td class="text-end" id="tot-det"></td><td class="text-end" id="tot-dif"></td>
        </tr></tfoot>
      </table></div>
      <div class="d-flex flex-wrap gap-2 small mb-2" id="resumen-estados"></div>`;

    function valorFila(i) {
      if (!editable) return filas[i].ajustado ?? filas[i].detectado;
      const inp = contenedor.querySelector(`input[data-i="${i}"]`);
      const n = Number(inp.value);
      return inp.value === '' || !Number.isInteger(n) || n < 0 ? null : n;
    }

    function recalcular() {
      let totEsp = 0, totDet = 0, faltantes = 0, sobrantes = 0, unidadesFalt = 0, unidadesSobr = 0, invalidos = 0;
      filas.forEach((f, i) => {
        const tr = contenedor.querySelector(`tr[data-i="${i}"]`);
        const v = valorFila(i);
        totEsp += f.esperado;
        if (v === null) {
          invalidos++;
          tr.className = '';
          tr.querySelector('.celda-dif').innerHTML = '<span class="text-danger">?</span>';
          if (editable) tr.querySelector('input').classList.add('is-invalid');
          return;
        }
        totDet += v;
        const dif = v - f.esperado;
        tr.className = dif < 0 ? 'fila-faltante' : dif > 0 ? 'fila-sobrante' : '';
        const etiqueta = dif < 0 ? '<span class="badge text-bg-danger ms-1">faltan</span>' : dif > 0 ? '<span class="badge badge-sobrante ms-1">sobran</span>' : '';
        tr.querySelector('.celda-dif').innerHTML = `<span class="fw-bold">${conSigno(dif)}</span>${etiqueta}`;
        if (dif < 0) { faltantes++; unidadesFalt -= dif; }
        if (dif > 0) { sobrantes++; unidadesSobr += dif; }
        if (editable) {
          const inp = tr.querySelector('input');
          inp.classList.remove('is-invalid');
          const editado = v !== f.detectado;
          inp.classList.toggle('editado', editado);
          tr.querySelector('.nota-modelo').classList.toggle('d-none', !editado);
        }
      });
      const totDif = totDet - totEsp;
      contenedor.querySelector('#tot-esp').textContent = totEsp;
      contenedor.querySelector('#tot-det').textContent = invalidos ? '—' : totDet;
      contenedor.querySelector('#tot-dif').textContent = invalidos ? '—' : conSigno(totDif);
      contenedor.querySelector('#resumen-estados').innerHTML = `
        <span class="badge text-bg-danger">${faltantes} producto(s) con faltante · ${unidadesFalt} u.</span>
        <span class="badge badge-sobrante">${sobrantes} con sobrante · ${unidadesSobr} u.</span>
        <span class="badge text-bg-success">${filas.length - faltantes - sobrantes - invalidos} sin diferencia</span>
        ${invalidos ? `<span class="badge text-bg-dark">${invalidos} valor(es) inválido(s)</span>` : ''}`;
      if (opciones.alCambiar) opciones.alCambiar({ invalidos });
    }

    if (editable) {
      contenedor.querySelectorAll('input.input-conteo').forEach((inp) => {
        inp.addEventListener('input', recalcular);
        inp.addEventListener('focus', () => inp.select());
      });
    }
    recalcular();

    return {
      valido: () => filas.every((f, i) => valorFila(i) !== null),
      /** Solo se envían las filas que el cajero cambió (el resto queda ajustado = detectado). */
      ajustes: () => filas.map((f, i) => ({ id_producto: f.id_producto, ajustado: valorFila(i) }))
        .filter((a, i) => a.ajustado !== null && a.ajustado !== filas[i].detectado),
    };
  }

  function badgeEstado(est) {
    return est === 'confirmado'
      ? '<span class="badge text-bg-success">confirmado</span>'
      : '<span class="badge text-bg-warning">pendiente</span>';
  }

  window.Ui = { esc, fecha, soloFecha, hoyIso, conSigno, moneda, textoError, alerta, toast, bannerModo, verImagen, tablaConciliacion, badgeEstado };
})();
