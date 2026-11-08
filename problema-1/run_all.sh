#!/usr/bin/env bash
# Ejecuta en orden todo el flujo del Problema 1.
# Uso: conda activate rp-examen && ./problema-1/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Nos movemos a problema-1/ para que las rutas relativas y los imports
# (data.py, reportes.py, evidencia_datos.py, modelos.py) funcionen igual sin
# importar desde dónde se llame.
cd "$SCRIPT_DIR"

# Permite importar common.py (raíz) y los módulos locales desde la libreta.
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
JUPYTER="${JUPYTER:-jupyter}"

echo "[1/8] Ejecutando el módulo de descarga y lectura: data.py"
"$PYTHON" data.py

echo "[2/8] Ejecutando la libreta: 01_exploracion.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  01_exploracion.ipynb

# Esta libreta solo usa el 70% de entrenamiento y guarda su evidencia en
# resultados/, incluido el split 70/30 que reutiliza 03_modelos.
echo "[3/8] Ejecutando la libreta: 02_decisiones_datos.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  02_decisiones_datos.ipynb

# Esta libreta busca hiperparámetros solo en el 70%, elige el mejor modelo por
# AUC de validación cruzada y calcula el AUC de prueba una sola vez por modelo.
echo "[4/8] Ejecutando la libreta: 03_modelos.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  03_modelos.ipynb

# Esta libreta evalúa la calibración del mejor modelo, ajusta un único
# calibrador global solo con el 70% y compara sigmoide contra isotónico.
echo "[5/8] Ejecutando la libreta: 04_calibracion.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  04_calibracion.ipynb

# Esta libreta usa las probabilidades calibradas, aplica el umbral 1/6,
# construye la tabla por grupo, verifica la identidad de FPR y hace el
# bootstrap de las diferencias entre hombres y mujeres.
echo "[6/8] Ejecutando la libreta: 05_tabla_grupos.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  05_tabla_grupos.ipynb

# Esta libreta evalúa las cinco definiciones de equidad, elige los umbrales
# por grupo con predicciones fuera de muestra del 70%, reporta en prueba sin
# afinarlos ahí, calcula el costo de equidad y el bootstrap posterior.
echo "[7/8] Ejecutando la libreta: 06_intervencion.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  06_intervencion.ipynb

# Esta libreta reentrena el mismo boosting con SEX como entrada, repite la
# tabla y el bootstrap de 05, mide asociaciones con SEX y el clasificador que
# predice SEX desde las demás variables.
echo "[8/8] Ejecutando la libreta: 07_sin_atributo.ipynb"
"$JUPYTER" nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  07_sin_atributo.ipynb

echo "Problema 1: flujo completado."
