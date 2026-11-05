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
├── .gitignore           # Excluye datos, .venv y cachés
├── problema-1/          # Grupos por SEXO, L = 5G, umbral del banco = 1/6
└── problema-2/          # Variable objetivo: registered
```

Los datos **no** se guardan en el repositorio. Cada problema los descarga
desde su fuente original en UCI y los deja en `data/` (ignorado por git).

## Cómo reproducir todo

Requiere **Python 3.14** (el proyecto fija esa versión mínima en
`pyproject.toml`).

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 2. Actualizar pip e instalar las dependencias fijas
python -m pip install --upgrade pip
python -m pip install -e .

# 3. Verificar la instalación
python -c "import numpy, pandas, sklearn, scipy, matplotlib, xlrd, openpyxl, sympy; print('Dependencias OK')"
```

Para abrir los notebooks o Jupyter Lab:

```bash
jupyter lab
```

## Reproducibilidad

- Toda función aleatoria debe fijar la semilla `223210350`.
- Los scripts deben correr de principio a fin en un entorno limpio.
- Las descargas se hacen dentro del código; no se versionan archivos de datos.
- Cada gráfica debe incluir ejes con nombre, título e interpretación.

Consulta `AGENTS.md` para las reglas completas del examen y
`problema-1/README.md` / `problema-2/README.md` para el detalle de cada
problema.
