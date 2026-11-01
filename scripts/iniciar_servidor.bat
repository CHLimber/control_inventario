@echo off
REM Inicia el servidor del control de inventario (FastAPI + uvicorn) en la red local.
REM Uso: doble clic o desde una consola en la raiz del repo:  scripts\iniciar_servidor.bat
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM Raiz del repo = carpeta padre de este script
pushd "%~dp0.."

REM ---- Python del entorno inventariovision (sin rutas fijas al usuario)
set "PY="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    echo %CONDA_PREFIX% | findstr /i "inventariovision" >nul && set "PY=%CONDA_PREFIX%\python.exe"
)
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\inventariovision\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\inventariovision\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\inventariovision\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\inventariovision\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\inventariovision\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\envs\inventariovision\python.exe"
if not defined PY (
    echo [ERROR] No se encontro el entorno conda "inventariovision".
    echo         Crealo con:  conda env create -f environment.yml
    popd
    exit /b 1
)

REM ---- Puerto: variable PUERTO del entorno o del .env; por defecto 8000
if not defined PUERTO if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        if /i "%%a"=="PUERTO" set "PUERTO=%%b"
    )
)
if not defined PUERTO set "PUERTO=8000"

echo ==============================================================
echo  Control de inventario por vision
echo  Python: %PY%
echo ==============================================================
echo.
echo  Desde esta PC:     http://localhost:%PUERTO%/panel/
echo  Laboratorio:       http://localhost:%PUERTO%/laboratorio/
echo  Swagger:           http://localhost:%PUERTO%/docs
echo.
echo  Desde el celular (misma red Wi-Fi), usa la IP de esta PC:
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /r /c:"IPv4"') do (
    set "IP=%%i"
    set "IP=!IP: =!"
    echo     http://!IP!:%PUERTO%/escaner/
)
echo.
echo  Si el celular no conecta, abre el puerto en el firewall (consola como administrador):
echo    netsh advfirewall firewall add rule name="Inventario vision" dir=in action=allow protocol=TCP localport=%PUERTO% profile=private
echo  y verifica que la red Wi-Fi este marcada como "Privada".
echo.
echo  Ctrl+C para detener.
echo ==============================================================

set PYTHONIOENCODING=utf-8
"%PY%" -m uvicorn src.api.main:app --host 0.0.0.0 --port %PUERTO%
set "CODIGO=%ERRORLEVEL%"
popd
exit /b %CODIGO%
