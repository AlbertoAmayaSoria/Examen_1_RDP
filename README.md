# Examen 1 — Reconocimiento de Patrones

Repositorio con la resolución del primer examen de **Reconocimiento de
Patrones** de la Universidad de Sonora.

## Datos del alumno

| Campo | Valor |
|---|---|
| Nombre | Alberto Amaya |
| Expediente | `223210350` |
| `a` | `0` |
| `b` | `5` |
| Semilla aleatoria | `223210350` |

## Contenido

```text
.
├── AGENTS.md            # Contexto y reglas de trabajo del examen
├── README.md            # Este archivo
├── pyproject.toml       # Dependencias con versiones fijas
├── environment.yml      # Entorno conda rp-examen (Python 3.14)
├── common.py            # SEED y parámetros comunes
├── .gitignore           # Excluye datos, entornos y cachés
├── resultados/          # Evidencia generada (CSV) al ejecutar las libretas
├── problema-1/          # Grupos por SEXO, L = 5G, umbral del banco = 1/6
└── problema-2/          # Variable objetivo: registered
```

Los datos **no** se guardan en el repositorio. Cada problema los descarga
desde su fuente original en UCI y los deja en `data/` (ignorado por git).

## Cómo reproducir todo

El entorno conda `rp-examen` usa **Python 3.14**. Las dependencias fijas viven
en `pyproject.toml`; `environment.yml` solo aporta Python y la instalación
editable del proyecto.

```bash
# 1. Crear el entorno conda (solo la primera vez)
conda env create -f environment.yml

# 2. Activar el entorno
conda activate rp-examen

# 3. Verificar la instalación
python -c "import numpy, pandas, sklearn, scipy, matplotlib, xlrd, openpyxl, sympy; print('Dependencias OK')"
```

Para abrir los notebooks o Jupyter Lab:

```bash
jupyter lab
```

## Ejecución por problema

Cada problema tiene un `run_all.sh` que ejecuta sus scripts `.py` y luego sus
libretas en orden:

```bash
./problema-1/run_all.sh
./problema-2/run_all.sh   # por ahora solo muestra un TODO
```

En el Problema 1, `02_decisiones_datos.ipynb` escribe su evidencia en
`resultados/`: el split 70/30, las 12 combinaciones con validación cruzada, la
comparación con/sin `SEX` y las tablas de "Significado de códigos no
documentados". `03_modelos.ipynb` reutiliza ese split, busca hiperparámetros
solo en el 70% con validación cruzada de 5 particiones y AUC, calcula el AUC
de prueba una sola vez por modelo y elige el mejor modelo por AUC de
validación cruzada. `04_calibracion.ipynb` evalúa la calibración global y por
grupo, compara sigmoide contra isotónico con validación cruzada anidada dentro
del 70% y ajusta un único calibrador global. `05_tabla_grupos.ipynb` usa esas
probabilidades calibradas con el umbral `1/6`, reporta `n`, `p_g`, tasa de
rechazo, `PPV`, `FPR` y `FNR` por sexo, verifica la identidad de `FPR` y hace
un bootstrap de 2,000 réplicas para las diferencias entre grupos. Las
constantes confirmadas (`G = 1`, `L = 5`, `UMBRAL = 1/6`) viven en
`common.py`.

## Reproducibilidad

- Toda función aleatoria debe fijar la semilla `223210350`.
- Los scripts deben correr de principio a fin en un entorno limpio.
- Las descargas se hacen dentro del código; no se versionan archivos de datos.
- Cada gráfica debe incluir ejes con nombre, título e interpretación.

Consulta `AGENTS.md` para las reglas completas del examen y
`problema-1/README.md` / `problema-2/README.md` para el detalle de cada
problema.
