# setup_env.ps1
# Script para Windows PowerShell: crea un virtualenv y prepara el entorno.
# Uso: .\setup_env.ps1

# 1) Crear y activar virtualenv en .venv
python -m venv .venv
# Si la política de ejecución impide activar, ejecuta (una sola vez) como administrador:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

Write-Host "Activated venv: $($env:VIRTUAL_ENV)" -ForegroundColor Green

# 2) Actualizar pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel

# 3) Reinstalar numpy forzadamente (soluciona problemas tipo uninstall-no-record-file)
# Si quieres otra versión, cambia la versión aquí.
python -m pip install --force-reinstall --no-deps numpy==1.26.4

# 4) Instalar el resto de dependencias desde requirements.txt
python -m pip install -r requirements.txt

Write-Host "Entorno preparado. Activa con: .\\.venv\\Scripts\\Activate.ps1 si no está activo." -ForegroundColor Green
