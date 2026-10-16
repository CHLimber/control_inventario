/*
 * Capa de acceso a la API REST (contrato: contratos/api_schemas.py).
 *
 * Modo REAL: llama a /api/... con rutas relativas al mismo servidor.
 * Modo SIMULADO: si la URL tiene ?mock=1, o si la API no responde al iniciar
 * (GET /api/clases falla), todas las llamadas se resuelven en el navegador con
 * datos que tienen la forma EXACTA del contrato (ver simulador.js).
 *
 * Uso:
 *   await Api.iniciar();                 // decide el modo
 *   const productos = await Api.get('/api/productos');
 *   await Api.post('/api/movimientos', {...});
 *   Errores: lanza Api.ApiError {status, detail}.
 */
(function () {
  'use strict';

  class ApiError extends Error {
    constructor(status, detail) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  }

  const params = new URLSearchParams(location.search);
  let modo = params.get('mock') === '1' ? 'simulado' : 'desconocido';
  let motivoSimulado = modo === 'simulado' ? 'Activado con ?mock=1' : '';
  let clasesCache = null;

  /** Convierte el campo detail de FastAPI (texto o lista de errores 422) en texto legible. */
  function textoDetail(detail, status) {
    if (detail == null || detail === '') return `Error ${status}`;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map((e) => {
        const loc = Array.isArray(e.loc) ? e.loc.filter((p) => p !== 'body').join('.') : '';
        return (loc ? loc + ': ' : '') + (e.msg || JSON.stringify(e));
      }).join(' · ');
    }
    return JSON.stringify(detail);
  }

  function armarQuery(query) {
    if (!query) return '';
    const qs = new URLSearchParams();
    Object.entries(query).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') qs.append(k, v);
    });
    const s = qs.toString();
    return s ? '?' + s : '';
  }

  async function fetchConTiempo(url, opciones, ms) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try {
      return await fetch(url, { ...opciones, signal: ctrl.signal });
    } finally {
      clearTimeout(t);
    }
  }

  /** Decide el modo: prueba GET /api/clases; si falla, pasa a simulado. */
  async function iniciar() {
    if (modo === 'real' || modo === 'simulado') return modo;
    try {
      const r = await fetchConTiempo('/api/clases', { headers: { Accept: 'application/json' } }, 3000);
      const tipo = r.headers.get('content-type') || '';
      if (r.ok && tipo.includes('application/json')) {
        clasesCache = await r.json();
        modo = 'real';
      } else {
        modo = 'simulado';
        motivoSimulado = `La API no respondió correctamente (HTTP ${r.status})`;
      }
    } catch (e) {
      modo = 'simulado';
      motivoSimulado = 'La API no responde';
    }
    return modo;
  }

  /**
   * Petición genérica.
   * @param {string} metodo GET | POST | PUT | DELETE
   * @param {string} ruta   /api/...
   * @param {{json?: any, form?: FormData, query?: object}} [op]
   */
  async function pedir(metodo, ruta, op = {}) {
    if (modo === 'desconocido') await iniciar();
    const url = ruta + armarQuery(op.query);

    if (modo === 'simulado') {
      const r = await window.Simulador.atender(metodo, url, op.json, op.form);
      if (r.status >= 400) throw new ApiError(r.status, textoDetail(r.data && r.data.detail, r.status));
      return r.data;
    }

    const init = { method: metodo, headers: { Accept: 'application/json' } };
    if (op.form) {
      init.body = op.form; // el navegador pone el boundary del multipart
    } else if (op.json !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(op.json);
    }
    let r;
    try {
      r = await fetch(url, init);
    } catch (e) {
      throw new ApiError(0, 'No se pudo conectar con el servidor. Revisá la red.');
    }
    if (r.status === 204) return null;
    let datos = null;
    const tipo = r.headers.get('content-type') || '';
    if (tipo.includes('application/json')) {
      try { datos = await r.json(); } catch (e) { datos = null; }
    } else {
      const txt = await r.text();
      datos = txt ? { detail: txt.slice(0, 300) } : null;
    }
    if (!r.ok) throw new ApiError(r.status, textoDetail(datos && datos.detail, r.status));
    return datos;
  }

  async function clases() {
    if (clasesCache) return clasesCache;
    clasesCache = await pedir('GET', '/api/clases');
    return clasesCache;
  }

  /** URL de imagen servida por la API; en simulado puede ser un data: URL. */
  function urlImagen(u) {
    return u || '';
  }

  window.Api = {
    ApiError,
    iniciar,
    pedir,
    clases,
    urlImagen,
    get: (ruta, query) => pedir('GET', ruta, { query }),
    post: (ruta, json) => pedir('POST', ruta, { json }),
    postForm: (ruta, form) => pedir('POST', ruta, { form }),
    put: (ruta, json) => pedir('PUT', ruta, { json }),
    del: (ruta) => pedir('DELETE', ruta),
    get modo() { return modo; },
    get motivoSimulado() { return motivoSimulado; },
  };
})();
