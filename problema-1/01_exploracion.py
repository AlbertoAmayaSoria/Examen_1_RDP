# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python (rp-examen)
#     language: python
#     name: rp-examen
# ---

# %% [markdown]
# # Problema 1 — Exploración inicial
#
# Esta libreta descarga el dataset UCI 350, lo carga con `data.py` y muestra
# los conteos solicitados. **No limpia, imputa ni recodifica valores**: los
# códigos raros de `EDUCATION`, `MARRIAGE` y `PAY_*` quedan intactos para que
# el alumno decida qué hacer con ellos.
#
# La libreta importa los módulos `.py`; no redefine funciones de descarga ni
# de reporte.

# %%
"""Configuración e importaciones de la libreta de exploración."""

from __future__ import annotations

import pandas as pd

from data import TARGET, cargar_dataframe
from reportes import (
    PAY_SOLICITADAS,
    columnas_pay,
    mostrar_codigos_no_documentados,
    mostrar_tasa_incumplimiento_por_sexo,
    mostrar_valores_y_conteos,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

# %% [markdown]
# ## Carga de datos
#
# `cargar_dataframe()` descarga el ZIP desde UCI solo si no está en `data/`,
# extrae el `.xls` y corrige la fila de encabezado desplazada. La tabla que se
# imprime solo confirma dimensiones y nombres de columnas; no transforma los
# valores originales.

# %%
df = cargar_dataframe()
print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

# El esquema original salta PAY_1. Lo hacemos explícito para no inventar una
# columna que no existe en el archivo de UCI.
faltantes = [columna for columna in PAY_SOLICITADAS if columna not in df]
if faltantes:
    print(
        "Nota: el archivo original no incluye "
        f"{', '.join(faltantes)}. Se reportan las columnas PAY_* "
        f"disponibles: {', '.join(columnas_pay(df))}."
    )

# %% [markdown]
# ## 1. Valores únicos y conteos
#
# Cada tabla muestra, para una variable, todos los códigos observados y
# cuántas filas tienen ese código. El porcentaje se calcula sobre las 30,000
# filas del dataset. Sirve para detectar categorías poco frecuentes o códigos
# que no aparecen en la documentación oficial.

# %%
mostrar_valores_y_conteos(df)

# %% [markdown]
# ## 2. Códigos fuera de la descripción oficial de UCI
#
# Esta tabla lista únicamente los códigos que el enunciado señala como no
# documentados. La columna `aparece_en_datos` indica si el código existe en el
# archivo y `conteo` cuántas veces aparece. **No se toma ninguna decisión**
# sobre imputarlos, agruparlos o eliminarlos.

# %%
mostrar_codigos_no_documentados(df)

# %% [markdown]
# ## 3. Tasa de incumplimiento por sexo
#
# La tabla agrupa por `SEX` (1 = hombre, 2 = mujer) y calcula:
#
# - `filas`: cantidad de personas de ese sexo;
# - `incumplimientos`: suma de la variable objetivo (`1` = incumplió);
# - `tasa_incumplimiento`: porcentaje de incumplimiento dentro del grupo.
#
# La comparación es descriptiva; no se ajusta por otras variables ni se
# establece causalidad.

# %%
mostrar_tasa_incumplimiento_por_sexo(df, TARGET)

# %% [markdown]
# ## Lectura de resultados
#
# Con los números anteriores se puede describir la composición de la muestra
# y comparar la tasa de incumplimiento entre hombres y mujeres. Cualquier
# decisión sobre códigos raros, imputación o modelado queda pendiente y debe
# consultarse con el alumno.
