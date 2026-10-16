@echo off
REM ============================================================================
REM  Genera un certificado HTTPS local (mkcert) para localhost + la IP de la PC,
REM  para usar la vista previa de cámara en vivo del escáner desde el celular.
REM  Salida: certs\servidor.pem y certs\servidor-key.pem  (certs\ esta en .gitignore)
REM  Ver docs\https_celular.md.  HTTPS es OPCIONAL: el boton "Escanear estante"
REM  funciona por HTTP comun.
REM
REM  Uso:  scripts\generar_certificado.bat            (detecta la IP sola)
REM        scripts\generar_certificado.bat 192.168.1.50
REM ============================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM Raiz del repo = carpeta padre de este script
set "RAIZ=%~dp0.."
pushd "%RAIZ%"

where mkcert >nul 2>nul
if errorlevel 1 (
  echo [ERROR] No se encontro mkcert en el PATH.
  echo   Instalalo con:  winget install FiloSottile.mkcert
  echo              o:  choco install mkcert
  echo   y volve a abrir la consola. Detalles en docs\https_celular.md
  popd
  exit /b 1
)

REM IP local: argumento 1 o la primera IPv4 que no sea 127.x ni 169.254.x
set "IP=%~1"
if "%IP%"=="" (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Sort-Object InterfaceMetric | Select-Object -First 1).IPAddress"`) do set "IP=%%i"
)
if "%IP%"=="" (
  echo [AVISO] No se pudo detectar la IP local; se genera solo para localhost.
  echo         Podes pasarla a mano:  scripts\generar_certificado.bat 192.168.1.50
)

if not exist certs mkdir certs

echo.
echo == 1/2  Instalando la CA local de mkcert en esta PC (pide confirmacion la primera vez)
mkcert -install
if errorlevel 1 ( popd & exit /b 1 )

echo.
echo == 2/2  Generando certificado para: localhost 127.0.0.1 ::1 %IP%
mkcert -cert-file certs\servidor.pem -key-file certs\servidor-key.pem localhost 127.0.0.1 ::1 %IP%
if errorlevel 1 ( popd & exit /b 1 )

for /f "usebackq delims=" %%c in (`mkcert -CAROOT`) do set "CAROOT=%%c"

echo.
echo ============================================================================
echo  Listo. Certificado en certs\servidor.pem y certs\servidor-key.pem
echo.
echo  Lanzar la API con HTTPS:
echo    uvicorn src.api.app:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\servidor-key.pem --ssl-certfile certs\servidor.pem
echo.
echo  En el celular (misma red Wi-Fi):  https://%IP%:8443/escaner/
echo.
echo  Para que el celular confie en el certificado, instala en el la CA raiz:
echo    "!CAROOT!\rootCA.pem"
echo  (NO compartas rootCA-key.pem). Pasos en docs\https_celular.md
echo ============================================================================
popd
endlocal
