# Problema 1

## Objetivo

Análisis de grupos definidos por **SEXO** (hombres vs. mujeres) con:

- **`L = 5G`**
- **Umbral del banco = `1/6`**

## Estado

Descarga y exploración inicial implementadas. El modelado y las decisiones
sobre códigos raros siguen pendientes.

## Ejecución

Desde la raíz del repositorio, con el entorno conda activado:

```bash
conda activate rp-examen
./problema-1/run_all.sh
```

El script `run_all.sh` ejecuta en orden:

1. `data.py`: descarga el ZIP de UCI (si hace falta), extrae el `.xls` en
   `data/` y devuelve el DataFrame con el encabezado corregido.
2. `01_exploracion.ipynb`: libreta emparejada con `01_exploracion.py` que
   imprime los conteos y la tasa de incumplimiento por sexo.

Las funciones reutilizables viven en `data.py` y `reportes.py`; la libreta no
las duplica. La semilla común está en `common.py` (`SEED = 223210350`).

## Puntos por confirmar con el enunciado

- [ ] Dataset UCI exacto y URL/ID de descarga.
- [ ] Definición operativa de `L = 5G`.
- [ ] Definición operativa del "umbral del banco" `= 1/6`.
- [ ] Variables a utilizar y preprocesamiento esperado.
- [ ] Criterio de evaluación y gráficas solicitadas.

> Regla del examen: si alguna definición no está en el enunciado, preguntar
> antes de implementar. No inventar parámetros.

## Reproducibilidad

- Semilla: `223210350`.
- Los datos se descargan desde UCI dentro del código y se guardan en `data/`
  (ignorado por git).
- Comentarios en español explicando el porqué de cada paso.
