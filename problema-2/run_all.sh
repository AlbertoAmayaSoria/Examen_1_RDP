#!/usr/bin/env bash
# Ejecuta en orden todo el flujo del Problema 2.
# Uso: conda activate rp-examen && ./problema-2/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# TODO: agregar aquí los scripts .py del Problema 2 en orden de ejecución.
# TODO: agregar aquí las libretas .ipynb del Problema 2 en orden de ejecución,
#       usando:
#       jupyter nbconvert --to notebook --execute --inplace \
#         --ExecutePreprocessor.timeout=1800 <libreta>.ipynb

echo "Problema 2: TODO. Aún no hay scripts .py ni libretas que ejecutar."
