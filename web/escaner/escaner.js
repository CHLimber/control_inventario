/*
 * Escáner móvil: el celular SOLO captura, achica y envía las fotos.
 * Todo el conteo lo hace el servidor (POST /api/escaneos → CierreOut).
 */
(function () {
  'use strict';

  const MAX_FOTOS = 5;          // contratos.api_schemas.MAX_IMAGENES_ESCANEO
  const LADO_MAYOR = 1280;      // px
  const CALIDAD_JPEG = 0.85;

  const $ = (id) => document.getElementById(id);
  const fotos = [];             // { blob, url }
  let cierre = null;            // CierreOut mostrado
  let tabla = null;             // control de la tabla editable
  let flujo = null;             // MediaStream de la cámara en vivo

  // ------------------------------------------------------------ redimensionado
  async function cargarFuente(blob) {
    try {
      // Respeta la orientación EXIF de las fotos del celular.
      const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
      return { fuente: bmp, ancho: bmp.width, alto: bmp.height, liberar: () => bmp.close && bmp.close() };
    } catch (e) {
      // Respaldo: <img> (los navegadores actuales aplican EXIF por defecto).
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.src = url;
      try { await img.decode(); } catch (err) { URL.revokeObjectURL(url); throw new Error('No se pudo leer la imagen'); }
      return { fuente: img, ancho: img.naturalWidth, alto: img.naturalHeight, liberar: () => URL.revokeObjectURL(url) };
    }
  }

  async function redimensionar(blob) {
    const { fuente, ancho, alto, liberar } = await cargarFuente(blob);
    const escala = Math.min(1, LADO_MAYOR / Math.max(ancho, alto));
    const w = Math.round(ancho * escala), h = Math.round(alto * escala);
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(fuente, 0, 0, w, h);
    liberar();
    const salida = await new Promise((res) => cv.toBlob(res, 'image/jpeg', CALIDAD_JPEG));
    if (!salida) throw new Error('No se pudo comprimir la imagen');
    return salida;
  }

  // ------------------------------------------------------------ fotos
  async function agregarArchivos(lista) {
    const archivos = Array.from(lista || []).filter((f) => f.type === '' || f.type.startsWith('image/'));
    if (!archivos.length) return;
    const libres = MAX_FOTOS - fotos.length;
    if (libres <= 0) { Ui.toast(`Máximo ${MAX_FOTOS} fotos por escaneo`, 'warning'); return; }
    if (archivos.length > libres) Ui.toast(`Solo se agregan ${libres} foto(s): el máximo es ${MAX_FOTOS}`, 'warning');
    for (const f of archivos.slice(0, libres)) {
      try {
        const blob = await redimensionar(f);
        fotos.push({ blob, url: URL.createObjectURL(blob) });
      } catch (e) {
        Ui.toast(`No se pudo procesar ${f.name || 'la foto'}: ${e.message}`, 'danger');
      }
    }
    dibujarMiniaturas();
  }

  function quitarFoto(i) {
    const [f] = fotos.splice(i, 1);
    if (f) URL.revokeObjectURL(f.url);
    dibujarMiniaturas();
  }

  function dibujarMiniaturas() {
    const cont = $('miniaturas');
    cont.innerHTML = fotos.map((f, i) => `
      <div class="miniatura">
        <img src="${f.url}" alt="Foto ${i + 1}" data-ver="${i}">
        <span class="num">${i + 1}</span>
        <button type="button" class="btn btn-danger btn-quitar" data-quitar="${i}" aria-label="Quitar foto ${i + 1}"><i class="bi bi-x-lg"></i></button>
      </div>`).join('');
    $('contador-fotos').textContent = `${fotos.length} / ${MAX_FOTOS}`;
    $('btn-enviar').disabled = fotos.length === 0;
    $('txt-enviar').textContent = fotos.length ? `Enviar y contar (${fotos.length} foto${fotos.length > 1 ? 's' : ''})` : 'Enviar y contar';
    const lleno = fotos.length >= MAX_FOTOS;
    $('btn-escanear').classList.toggle('disabled', lleno);
    $('btn-escanear').innerHTML = `<i class="bi bi-camera-fill"></i>${fotos.length ? 'Agregar otra foto' : 'Escanear estante'}`;
    $('btn-capturar').disabled = lleno;
  }

  // ------------------------------------------------------------ cámara en vivo (opcional, HTTPS)
  function configurarVivo() {
    const disponible = window.isSecureContext && navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function';
    $('btn-vivo').classList.toggle('d-none', !disponible);
    $('aviso-https').classList.toggle('d-none', !!disponible);
  }

  async function abrirVivo() {
    try {
      flujo = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      $('video-vivo').srcObject = flujo;
      $('panel-vivo').classList.remove('d-none');
      $('panel-vivo').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      Ui.toast('No se pudo abrir la cámara: ' + (e.message || e.name), 'danger');
    }
  }

  function cerrarVivo() {
    if (flujo) flujo.getTracks().forEach((t) => t.stop());
    flujo = null;
    $('video-vivo').srcObject = null;
    $('panel-vivo').classList.add('d-none');
  }

  async function capturarVivo() {
    const v = $('video-vivo');
    if (!v.videoWidth) return;
    const cv = document.createElement('canvas');
    cv.width = v.videoWidth; cv.height = v.videoHeight;
    cv.getContext('2d').drawImage(v, 0, 0);
    const blob = await new Promise((res) => cv.toBlob(res, 'image/jpeg', 0.92));
    if (blob) await agregarArchivos([blob]);
    if (fotos.length >= MAX_FOTOS) cerrarVivo();
  }

  // ------------------------------------------------------------ envío y resultado
  function cargando(si, texto) {
    $('overlay').hidden = !si;
    if (texto) $('overlay-texto').textContent = texto;
  }

  async function enviar() {
    if (!fotos.length) return;
    $('avisos').innerHTML = '';
    const form = new FormData();
    fotos.forEach((f, i) => form.append('imagenes', f.blob, `foto${i + 1}.jpg`));
    const usuario = $('input-usuario').value.trim();
    if (usuario) form.append('usuario', usuario);
    cargando(true, `Enviando ${fotos.length} foto(s) y contando productos…`);
    try {
      const c = await Api.postForm('/api/escaneos', form);
      cerrarVivo();
      mostrarResultado(c);
    } catch (e) {
      Ui.alerta($('avisos'), 'danger', `<strong>No se pudo procesar el escaneo.</strong> ${Ui.esc(Ui.textoError(e))}`);
    } finally {
      cargando(false);
    }
  }

  function mostrarResultado(c) {
    cierre = c;
    $('paso-captura').classList.add('d-none');
    $('paso-resultado').classList.remove('d-none');
    $('pendientes').innerHTML = '';
    $('res-titulo').textContent = `Cierre #${c.id}`;
    $('res-meta').textContent = `${Ui.fecha(c.fecha)} · ${c.usuario} · modelo ${c.variante_modelo}` + (c.tiempo_ms ? ` · ${(c.tiempo_ms / 1000).toFixed(1)} s` : '');
    $('res-estado').innerHTML = Ui.badgeEstado(c.estado);

    const img = $('res-imagen');
    const url = Api.urlImagen(c.imagen_anotada_url || (c.imagenes_urls || [])[0]);
    img.closest('figure').classList.toggle('d-none', !url);
    if (url) img.src = url;
    $('res-originales').innerHTML = (c.imagenes_urls || []).map((u, i) =>
      `<button type="button" class="btn btn-sm btn-outline-secondary" data-original="${Ui.esc(u)}"><i class="bi bi-image me-1"></i>Foto ${i + 1}</button>`).join('');

    $('res-no-reconocidos').innerHTML = c.no_reconocidos > 0
      ? `<div class="alert alert-warning py-2 small"><i class="bi bi-question-octagon me-1"></i>
          <strong>${c.no_reconocidos}</strong> objeto(s) no reconocido(s) (marcados con “?”). Si son productos, sumalos a mano en la tabla.</div>`
      : '';

    const pendiente = c.estado === 'pendiente';
    tabla = Ui.tablaConciliacion($('res-tabla'), c, {
      editable: pendiente,
      alCambiar: ({ invalidos }) => { $('btn-confirmar').disabled = invalidos > 0; },
    });
    $('ayuda-tabla').classList.toggle('d-none', !pendiente);
    $('bloque-confirmar').classList.toggle('d-none', !pendiente);
    $('bloque-confirmado').classList.toggle('d-none', pendiente);
    if (!pendiente) {
      const t = c.totales;
      $('msg-confirmado').innerHTML = `<i class="bi bi-check-circle-fill me-1"></i><strong>Cierre confirmado.</strong>
        Diferencia total: ${Ui.conSigno(t.diferencia)} u. en ${t.productos_con_diferencia} producto(s).
        El stock se ajustó al conteo real.${c.observacion ? `<div class="small mt-1">Obs.: ${Ui.esc(c.observacion)}</div>` : ''}`;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  async function confirmarCierre() {
    if (!cierre || !tabla) return;
    if (!tabla.valido()) { Ui.toast('Hay cantidades inválidas en la tabla', 'danger'); return; }
    const ajustes = tabla.ajustes();
    const conDif = $('res-tabla').querySelectorAll('tr.fila-faltante, tr.fila-sobrante').length;
    const msg = `¿Confirmar el cierre #${cierre.id}?\n\n` +
      (ajustes.length ? `Corregiste ${ajustes.length} conteo(s). ` : '') +
      (conDif ? `Se generará un ajuste de stock en ${conDif} producto(s) con diferencia.` : 'No hay diferencias con el stock esperado.');
    if (!confirm(msg)) return;
    const cuerpo = {
      ajustes,
      observacion: $('input-observacion').value.trim() || null,
      usuario: $('input-usuario').value.trim() || 'cajero',
    };
    cargando(true, 'Confirmando cierre…');
    try {
      const c = await Api.post(`/api/cierres/${cierre.id}/confirmar`, cuerpo);
      mostrarResultado(c);
      Ui.toast('Cierre confirmado');
    } catch (e) {
      Ui.alerta($('avisos'), 'danger', `<strong>No se pudo confirmar.</strong> ${Ui.esc(Ui.textoError(e))}`);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } finally {
      cargando(false);
    }
  }

  function nuevoEscaneo() {
    if (cierre && cierre.estado === 'pendiente' &&
        !confirm(`El cierre #${cierre.id} queda PENDIENTE (se puede confirmar luego desde el panel). ¿Empezar un escaneo nuevo?`)) return;
    cierre = null; tabla = null;
    fotos.splice(0).forEach((f) => URL.revokeObjectURL(f.url));
    dibujarMiniaturas();
    $('input-observacion').value = '';
    $('avisos').innerHTML = '';
    $('paso-resultado').classList.add('d-none');
    $('paso-captura').classList.remove('d-none');
    buscarPendientes();
  }

  /** Ofrece retomar un cierre pendiente (p. ej. si se recargó la página). */
  async function buscarPendientes() {
    try {
      const lista = await Api.get('/api/cierres', { estado: 'pendiente' });
      if (!lista.length || cierre) { $('pendientes').innerHTML = ''; return; }
      const ult = lista[0];
      $('pendientes').innerHTML = `<div class="alert alert-light border py-2 small d-flex align-items-center gap-2">
        <i class="bi bi-hourglass-split"></i><span class="flex-grow-1">Hay ${lista.length} cierre(s) pendiente(s) de confirmar.</span>
        <button type="button" class="btn btn-sm btn-outline-primary" id="btn-retomar">Retomar #${ult.id}</button></div>`;
      $('btn-retomar').addEventListener('click', async () => {
        cargando(true, 'Cargando cierre…');
        try { mostrarResultado(await Api.get(`/api/cierres/${ult.id}`)); }
        catch (e) { Ui.alerta($('avisos'), 'danger', Ui.esc(Ui.textoError(e))); }
        finally { cargando(false); }
      });
    } catch (e) { /* no es crítico */ }
  }

  // ------------------------------------------------------------ eventos
  function enlazar() {
    $('input-camara').addEventListener('change', async (ev) => { await agregarArchivos(ev.target.files); ev.target.value = ''; });
    $('input-galeria').addEventListener('change', async (ev) => { await agregarArchivos(ev.target.files); ev.target.value = ''; });
    $('miniaturas').addEventListener('click', (ev) => {
      const q = ev.target.closest('[data-quitar]');
      if (q) { quitarFoto(Number(q.dataset.quitar)); return; }
      const v = ev.target.closest('[data-ver]');
      if (v) Ui.verImagen(fotos[Number(v.dataset.ver)].url, `Foto ${Number(v.dataset.ver) + 1}`);
    });
    $('btn-vivo').addEventListener('click', abrirVivo);
    $('btn-cerrar-vivo').addEventListener('click', cerrarVivo);
    $('btn-capturar').addEventListener('click', capturarVivo);
    $('btn-enviar').addEventListener('click', enviar);
    $('btn-confirmar').addEventListener('click', confirmarCierre);
    $('btn-nuevo').addEventListener('click', nuevoEscaneo);
    $('res-imagen').addEventListener('click', () => Ui.verImagen($('res-imagen').src, cierre ? `Cierre #${cierre.id} · imagen anotada` : 'Imagen anotada'));
    $('res-originales').addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-original]');
      if (b) Ui.verImagen(b.dataset.original, 'Foto original');
    });
    $('input-usuario').addEventListener('change', () => {
      try { localStorage.setItem('escaner_usuario', $('input-usuario').value.trim()); } catch (e) { /* nada */ }
    });
    window.addEventListener('pagehide', cerrarVivo);
  }

  async function iniciar() {
    try { const u = localStorage.getItem('escaner_usuario'); if (u) $('input-usuario').value = u; } catch (e) { /* nada */ }
    enlazar();
    configurarVivo();
    dibujarMiniaturas();
    await Api.iniciar();
    Ui.bannerModo($('banner-modo'));
    buscarPendientes();
  }

  iniciar();
})();
