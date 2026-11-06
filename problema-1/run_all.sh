#!/usr/bin/env bash
# Ejecuta en orden todo el flujo del Problema 1.
# Uso: conda activate rp-examen && ./problema-1/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Nos movemos a problema-1/ para que las rutas relativas y los imports
# (data.py, reportes.py) funcionen igual sin importar desde dónde se llame.
cd "$SCRIPT_DIR"

# Permite importar common.py (raíz) y los módulos locales desde la libreta.
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
JUPYTER="${JUPYTER:-jupyter}"

echo "[1/2] Ejecutando el módulo de descarga y lectura: data.py"
"$PYTHON" data.py

echo "[2/2] Ejecutando la libreta: 01_exploracion.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  01_exploracion.ipynb

echo "Problema 1: flujo completado."
