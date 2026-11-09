#!/usr/bin/env bash
# Ejecuta el Examen 1 completo en el orden correcto:
#   1) Problema 1 (descarga, exploración, modelos, calibración, equidad)
#   2) Problema 2 (descarga, kernels, decisiones, modelos, evaluación final)
#   3) Compilación del documento LaTeX y copia de main.pdf a la raíz
#
# Uso:
#   conda activate rp-examen
#   ./run_all.sh
#
# Opciones:
#   --skip-p1    no ejecutar el Problema 1
#   --skip-p2    no ejecutar el Problema 2
#   --skip-pdf   no compilar el PDF
#   -h, --help   mostrar esta ayuda
#
# Si el entorno rp-examen no está activo pero conda existe, el script se
# re-ejecuta dentro de ese entorno automáticamente.
#
# Para el PDF se usa tectonic, latexmk o pdflatex, en ese orden. Si tectonic
# está en una ruta no estándar:  TECTONIC=/ruta/a/tectonic ./run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uso() {
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# --- 1. Asegurar el entorno conda rp-examen --------------------------------
# RP_EXAMEN_ENV_OK evita un bucle si `conda run` no llegara a definir
# CONDA_DEFAULT_ENV por alguna razón.
if [[ "${CONDA_DEFAULT_ENV:-}" != "rp-examen" && "${RP_EXAMEN_ENV_OK:-0}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "==> El entorno rp-examen no está activo; re-ejecutando con conda run."
    export RP_EXAMEN_ENV_OK=1
    exec conda run -n rp-examen --no-capture-output bash "$SCRIPT_DIR/run_all.sh" "$@"
  fi
  echo "ERROR: activa el entorno antes de correr el examen:" >&2
  echo "       conda env create -f environment.yml   # solo la primera vez" >&2
  echo "       conda activate rp-examen" >&2
  exit 1
fi

# --- 2. Leer opciones ------------------------------------------------------
RUN_P1=1
RUN_P2=1
RUN_PDF=1
for arg in "$@"; do
  case "$arg" in
    --skip-p1)  RUN_P1=0 ;;
    --skip-p2)  RUN_P2=0 ;;
    --skip-pdf) RUN_PDF=0 ;;
    -h|--help)  uso; exit 0 ;;
    *)
      echo "ERROR: opción desconocida: $arg" >&2
      uso >&2
      exit 1
      ;;
  esac
done

# --- 3. Verificar dependencias mínimas -------------------------------------
PYTHON="${PYTHON:-python}"
JUPYTER="${JUPYTER:-jupyter}"

if ! "$PYTHON" - <<'PY'
import importlib
faltan = []
for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib",
            "xlrd", "openpyxl", "sympy", "IPython"):
    try:
        importlib.import_module(mod)
    except Exception:
        faltan.append(mod)
if faltan:
    print("Faltan dependencias:", ", ".join(faltan))
    raise SystemExit(1)
PY
then
  echo "ERROR: faltan dependencias. Reinstala el entorno con:" >&2
  echo "       conda env create -f environment.yml" >&2
  exit 1
fi

if ! command -v "$JUPYTER" >/dev/null 2>&1; then
  echo "ERROR: no se encontró '$JUPYTER' en el entorno." >&2
  exit 1
fi

echo "==> Entorno: ${CONDA_DEFAULT_ENV:-?} | Python: $("$PYTHON" --version 2>&1)"

# --- 4. Problema 1 ---------------------------------------------------------
if [[ "$RUN_P1" -eq 1 ]]; then
  echo
  echo "############################################################"
  echo "# [1/3] PROBLEMA 1"
  echo "############################################################"
  "$SCRIPT_DIR/problema-1/run_all.sh"
else
  echo
  echo "==> [1/3] Problema 1 omitido (--skip-p1)."
fi

# --- 5. Problema 2 ---------------------------------------------------------
if [[ "$RUN_P2" -eq 1 ]]; then
  echo
  echo "############################################################"
  echo "# [2/3] PROBLEMA 2"
  echo "############################################################"
  "$SCRIPT_DIR/problema-2/run_all.sh"
else
  echo
  echo "==> [2/3] Problema 2 omitido (--skip-p2)."
fi

# --- 6. Documento LaTeX ----------------------------------------------------
# El PDF se compila al final porque resume los resultados de ambos problemas.
if [[ "$RUN_PDF" -eq 1 ]]; then
  echo
  echo "############################################################"
  echo "# [3/3] DOCUMENTO LaTeX"
  echo "############################################################"
  # Motor LaTeX: se puede forzar con TECTONIC=/ruta/a/tectonic. Si no, se
  # busca tectonic en el PATH y en ubicaciones típicas de instalación.
  TECTONIC_BIN="${TECTONIC:-}"
  if [[ -z "$TECTONIC_BIN" ]]; then
    for cand in tectonic "$HOME/.local/bin/tectonic" \
                "$HOME/.cargo/bin/tectonic" /usr/local/bin/tectonic \
                /opt/homebrew/bin/tectonic; do
      if command -v "$cand" >/dev/null 2>&1; then
        TECTONIC_BIN="$cand"
        break
      fi
    done
  fi

  (
    cd "$SCRIPT_DIR/entrega"
    if [[ -n "$TECTONIC_BIN" ]]; then
      echo "==> Compilando con $TECTONIC_BIN."
      "$TECTONIC_BIN" -X compile main.tex
    elif command -v latexmk >/dev/null 2>&1; then
      echo "==> Compilando con latexmk."
      latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    elif command -v pdflatex >/dev/null 2>&1; then
      echo "==> Compilando con pdflatex (dos pasadas)."
      pdflatex -interaction=nonstopmode -halt-on-error main.tex
      pdflatex -interaction=nonstopmode -halt-on-error main.tex
    else
      echo "AVISO: no hay motor LaTeX instalado (tectonic, latexmk ni pdflatex)." >&2
      echo "       Se omite el PDF; el resto del examen sí quedó ejecutado." >&2
      exit 0
    fi
    cp main.pdf "$SCRIPT_DIR/main.pdf"
  )
  echo "==> PDF copiado a: $SCRIPT_DIR/main.pdf"
else
  echo
  echo "==> [3/3] PDF omitido (--skip-pdf)."
fi

# --- 7. Resumen ------------------------------------------------------------
echo
echo "############################################################"
echo "# EXAMEN COMPLETO"
echo "############################################################"
echo "Resultados (CSV) en:  $SCRIPT_DIR/resultados/"
echo "Figuras (PNG) en:     $SCRIPT_DIR/resultados/"
[[ -f "$SCRIPT_DIR/main.pdf" ]] && echo "Documento final:      $SCRIPT_DIR/main.pdf"
echo "Listo. Revisa los resultados y haz el commit manualmente."
