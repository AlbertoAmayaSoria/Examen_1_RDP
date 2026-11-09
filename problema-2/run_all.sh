#!/usr/bin/env bash
# Ejecuta en orden todo el flujo del Problema 2.
# Uso: conda activate rp-examen && ./problema-2/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
JUPYTER="${JUPYTER:-jupyter}"

echo "[1/5] Ejecutando el módulo de descarga y lectura: data.py"
"$PYTHON" data.py

echo "[2/5] Ejecutando la libreta: 01_kernels_verificacion.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  01_kernels_verificacion.ipynb

echo "[3/5] Ejecutando la libreta: 02_decisiones_p2.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  02_decisiones_p2.ipynb

echo "[4/5] Ejecutando la libreta: 03_modelos_p2.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  03_modelos_p2.ipynb

echo "[5/5] Ejecutando la libreta: 04_evaluacion_final.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  04_evaluacion_final.ipynb

echo "Problema 2: flujo completado."
