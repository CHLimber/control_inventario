# HTTPS para usar el celular como escáner (opcional)

El celular es solo un **escáner**: saca la foto y la envía a la PC, que hace todo el conteo.

## ¿Hace falta HTTPS?

**No para lo principal.** El botón **Escanear estante** usa
`<input type="file" accept="image/*" capture="environment">`, que abre la cámara del
celular y funciona por HTTP común:

```
http://<IP-de-la-PC>:8000/escaner/
```

HTTPS solo hace falta para la **vista previa en vivo** (`getUserMedia`), porque los
navegadores exigen un "contexto seguro" (HTTPS o `localhost`). Sin HTTPS la página
muestra un aviso discreto y oculta el botón *Cámara en vivo*.

Hay dos caminos: **A) certificado local con mkcert** (todo queda en la red local) o
**B) túnel con cloudflared** (más rápido de armar, pero sale a internet).

---

## A) Certificado local con mkcert

### 1. Instalar mkcert (una vez, en la PC)

```bat
winget install FiloSottile.mkcert
```

(o `choco install mkcert`, o bajar `mkcert-vX.Y.Z-windows-amd64.exe` de
<https://github.com/FiloSottile/mkcert/releases>, renombrarlo a `mkcert.exe` y ponerlo en el PATH).
Cerrar y volver a abrir la consola.

### 2. Generar el certificado

Desde la raíz del repo:

```bat
scripts\generar_certificado.bat
```

El script:

1. ejecuta `mkcert -install` (crea una CA local y la instala en Windows; la primera vez pide confirmación);
2. genera `certs\servidor.pem` y `certs\servidor-key.pem` válidos para `localhost`, `127.0.0.1`, `::1` y la IP local de la PC
   (se detecta sola; o se pasa a mano: `scripts\generar_certificado.bat 192.168.1.50`).

`certs/` y `*.pem` están en `.gitignore`: **nunca se suben al repo**.

> Si la IP de la PC cambia (otra red Wi-Fi, DHCP), hay que volver a correr el script.

### 3. Lanzar la API con HTTPS

```bat
conda activate inventariovision
uvicorn src.api.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\servidor-key.pem --ssl-certfile certs\servidor.pem
```

(ajustar `src.api.main:app` al módulo real de la aplicación FastAPI). La primera vez, Windows
puede preguntar si permite a Python en redes privadas: aceptar. Si el celular no llega,
revisar que la red esté como *Privada* y que el firewall deje pasar el puerto 8443.

En la PC: `https://localhost:8443/panel/`. En el celular (misma Wi-Fi):
`https://<IP-de-la-PC>:8443/escaner/`.

### 4. Hacer que el celular confíe en la CA

El certificado lo firmó la CA local de mkcert; el celular no la conoce, así que hay que
instalarle **solo el certificado raíz** (`rootCA.pem`).

```bat
mkcert -CAROOT
```

muestra la carpeta (p. ej. `C:\Users\<usuario>\AppData\Local\mkcert`). Ahí está `rootCA.pem`.

> **Nunca copies `rootCA-key.pem`**: con esa clave cualquiera podría firmar certificados
> que tu celular aceptaría. Al terminar la práctica se puede desinstalar todo con `mkcert -uninstall`
> y borrando la CA del celular.

Pasar `rootCA.pem` al celular (cable USB, Drive, correo a uno mismo, etc.).

**Android** (los menús cambian según la marca/versión):

1. Copiar el archivo y, si hace falta, renombrarlo a `rootCA.crt`.
2. *Ajustes → Seguridad y privacidad → Más ajustes de seguridad → Cifrado y credenciales →
   Instalar un certificado → Certificado de CA* (en algunos: *Ajustes → Seguridad →
   Instalar desde almacenamiento*).
3. Aceptar la advertencia y elegir el archivo. Queda en *Credenciales de confianza → Usuario*.
4. Chrome para Android acepta las CA de usuario para sitios normales, así que
   `https://<IP>:8443/escaner/` debería abrir sin advertencia.

**iOS / iPadOS**:

1. Enviarse `rootCA.pem` (AirDrop o correo) y abrirlo: aparece "Perfil descargado".
2. *Ajustes → General → VPN y gestión de dispositivos* → elegir el perfil → *Instalar*.
3. **Paso que se suele olvidar:** *Ajustes → General → Información → Ajustes de confianza de
   certificados* → activar la confianza total para la CA de mkcert.
4. Abrir `https://<IP>:8443/escaner/` en Safari.

Comprobación: en la página del escáner debe aparecer el botón **Cámara en vivo** (y no el
aviso de HTTPS). Al tocarlo, el navegador pide permiso para la cámara y usa la trasera.

---

## B) Alternativa sin certificados: túnel con cloudflared

`cloudflared` crea una URL pública `https://….trycloudflare.com` con un certificado válido,
así que el celular no necesita instalar nada. La PC necesita internet, y **cualquiera con la
URL puede entrar mientras el túnel esté abierto**: usarlo solo para la demo y cerrarlo después.

```bat
winget install Cloudflare.cloudflared
```

En una consola, la API por HTTP normal:

```bat
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

En otra consola:

```bat
cloudflared tunnel --url http://localhost:8000
```

Copiar la URL que imprime (`https://<palabras-al-azar>.trycloudflare.com`) y abrir en el
celular `https://<palabras-al-azar>.trycloudflare.com/escaner/`. La URL cambia cada vez
que se lanza el túnel. Las fotos viajan por internet, así que el envío puede ser más lento
que en la red local.

---

## Resumen de URLs

| Modo | Escáner (celular) | Panel (PC) |
|---|---|---|
| HTTP en la red local | `http://<IP>:8000/escaner/` | `http://localhost:8000/panel/` |
| HTTPS con mkcert | `https://<IP>:8443/escaner/` | `https://localhost:8443/panel/` |
| Túnel cloudflared | `https://<…>.trycloudflare.com/escaner/` | `http://localhost:8000/panel/` |
| Sin backend (simulado) | `…/escaner/?mock=1` | `…/panel/?mock=1` |
