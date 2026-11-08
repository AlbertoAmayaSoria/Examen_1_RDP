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
# # Problema 1 — Modelos finales y búsqueda de hiperparámetros
#
# Esta libreta aplica las decisiones registradas en `bitacora_notas.md`:
#
# - **Códigos raros:** agrupar `EDUCATION {0, 5, 6} -> 4` y
#   `MARRIAGE {0} -> 3`.
# - **`SEX`:** no entra como variable de entrada del modelo principal; se
#   conserva aparte solo para auditoría por grupo.
# - **`HistGradientBoostingClassifier`:** usa `PAY_*` numérico tal cual
#   (variante A).
# - **Regresión logística con escalado:** usa la variante B de `PAY_*`
#   (indicadores de estado `-2`, `-1`, `0` más meses de atraso), porque con
#   `PAY_*` numérico rinde claramente menos en validación cruzada.
#
# Constantes confirmadas en `common.py`: `G = 1`, `L = 5`,
# `UMBRAL = G / (G + L) = 1/6`.
#
# Reglas que respeta la libreta:
#
# - Usa el mismo split 70/30 guardado en `resultados/split_indices.csv`.
# - La búsqueda de hiperparámetros se hace **solo en el 70%** de
#   entrenamiento, con validación cruzada estratificada de 5 particiones y
#   AUC.
# - El AUC de prueba se calcula **una sola vez por modelo**.
# - El “mejor modelo” para los siguientes incisos se elige por AUC de
#   validación cruzada, no por el AUC de prueba.
# - No se toca el Problema 2.

# %%
"""Configuración, rutas e importaciones de la libreta."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score


def _buscar_raiz(inicio: Path) -> Path:
    """Sube desde el directorio actual hasta encontrar la raíz del repo.

    La raíz se reconoce porque contiene `common.py` y la carpeta `problema-1/`.
    Así la libreta funciona igual si se abre desde la raíz o desde
    `problema-1/`, y también cuando la ejecuta `nbconvert`.
    """
    for base in [inicio, *inicio.parents]:
        if (base / "common.py").exists() and (base / "problema-1").is_dir():
            return base
    return inicio


_AQUI = Path.cwd()
_RAIZ = _buscar_raiz(_AQUI)
_DIR_PROBLEMA = _RAIZ / "problema-1" if (_RAIZ / "problema-1").is_dir() else _RAIZ

# Permite importar common.py (raíz) y los módulos de problema-1/.
for _ruta in (_RAIZ, _DIR_PROBLEMA):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

from common import G, L, SEED, UMBRAL
from data import TARGET, cargar_dataframe
import evidencia_datos as ev
import modelos as mo

RESULTADOS = _RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print(f"SEED = {SEED}")
print(f"G = {G}, L = {L}, UMBRAL = G/(G+L) = {UMBRAL:.6f}")
print(f"Variable objetivo = '{TARGET}'")
print(f"Directorio de resultados = {RESULTADOS}")

# %% [markdown]
# ## 1. Carga de datos y split 70/30
#
# Se carga el dataset original y se leen los índices del split que ya produjo
# `02_decisiones_datos.ipynb`. Para evitar usar por accidente otro split, se
# vuelve a calcular el split con `SEED` y se comprueba que coincida
# exactamente. Las columnas de entrada excluyen `ID`, la variable objetivo y
# `SEX`; `SEX` se conserva aparte solo para auditoría.

# %%
df = cargar_dataframe()
columnas_pay = mo.obtener_columnas_pay(df)

ruta_split = RESULTADOS / "split_indices.csv"
if not ruta_split.exists():
    raise FileNotFoundError(
        f"No existe {ruta_split}. Ejecuta primero 02_decisiones_datos.ipynb."
    )

idx_train, idx_test = ev.cargar_indices(ruta_split, df, TARGET)
idx_train_esperado, idx_test_esperado = ev.dividir_estratificado(
    df, TARGET, test_size=0.30, seed=SEED
)
if not (
    np.array_equal(idx_train, idx_train_esperado)
    and np.array_equal(idx_test, idx_test_esperado)
):
    raise ValueError(
        "El split guardado no coincide con el split reproducible de SEED. "
        "Revisa resultados/split_indices.csv antes de continuar."
    )

columnas_x = mo.columnas_entrada(df, TARGET, incluir_sexo=mo.INCLUIR_SEXO)
if "SEX" in columnas_x:
    raise ValueError("SEX no debe estar en las columnas de entrada del modelo.")

X_train = df.iloc[idx_train][columnas_x].reset_index(drop=True)
y_train = df.iloc[idx_train][TARGET].reset_index(drop=True)
X_test = df.iloc[idx_test][columnas_x].reset_index(drop=True)
y_test = df.iloc[idx_test][TARGET].reset_index(drop=True)

tabla_split = pd.DataFrame(
    {
        "conjunto": ["entrenamiento (70%)", "prueba (30%)"],
        "filas": [len(idx_train), len(idx_test)],
        "tasa_incumplimiento_pct": [
            100.0 * float(y_train.mean()),
            100.0 * float(y_test.mean()),
        ],
    }
)
display(tabla_split)

print(f"Columnas de entrada ({len(columnas_x)}): {columnas_x}")
print(f"¿SEX está en las columnas de entrada? {'SEX' in columnas_x}")
print(f"Archivo de split: {ruta_split}")

# %% [markdown]
# La tabla confirma que el split conserva 21,000 filas de entrenamiento y
# 9,000 de prueba, con tasas de incumplimiento muy parecidas. También confirma
# que los índices leídos son exactamente los que produce `SEED`; por eso el
# 30% de prueba queda reservado para calcular el AUC una sola vez por modelo.
# `SEX` no aparece en las columnas de entrada: se usará después solo para
# auditar los resultados por grupo.

# %% [markdown]
# ## 2. Particiones estratificadas de validación cruzada
#
# Se crean una sola vez 5 particiones estratificadas con `SEED`. Todas las
# búsquedas de hiperparámetros usan exactamente estas mismas particiones, para
# que los AUC de los dos modelos sean comparables.

# %%
folds = ev.crear_folds(X_train, y_train, n_splits=5, seed=SEED)

filas_folds = []
for numero, (idx_fold_train, idx_fold_validacion) in enumerate(folds, start=1):
    filas_folds.append(
        {
            "particion": numero,
            "n_entrenamiento": len(idx_fold_train),
            "n_validacion": len(idx_fold_validacion),
            "tasa_entrenamiento_pct": 100.0
            * float(y_train.iloc[idx_fold_train].mean()),
            "tasa_validacion_pct": 100.0
            * float(y_train.iloc[idx_fold_validacion].mean()),
        }
    )
tabla_folds = pd.DataFrame(filas_folds)
display(tabla_folds)

# %% [markdown]
# Las 5 particiones tienen tamaños iguales y tasas de incumplimiento
# parecidas, así que la comparación entre modelos no queda sesgada por una
# partición con muchos más incumplimientos. La semilla es la misma que usa la
# libreta de evidencia, por lo que los resultados son reproducibles.

# %% [markdown]
# ## 3. Búsqueda de hiperparámetros
#
# La búsqueda se hace **solo con el 70% de entrenamiento**. Para la regresión
# logística se prueba `C` en `{0.01, 0.1, 1, 10}`. Para
# `HistGradientBoostingClassifier` se prueba `learning_rate` en `{0.03, 0.1}`,
# `max_depth` en `{3, 5, None}`, `max_leaf_nodes` en `{15, 31}` y
# `l2_regularization` en `{0, 1}`. El criterio es AUC medio de validación
# cruzada. `GridSearchCV` reentrena el mejor Pipeline con todo el 70% al
# terminar (`refit=True`).

# %%
pipeline_logistica = mo.construir_pipeline(
    variante_raros=mo.VARIANTE_RAROS,
    variante_pay=mo.VARIANTE_PAY_POR_MODELO["logistica"],
    modelo="logistica",
    columnas_pay=columnas_pay,
    incluir_sexo=mo.INCLUIR_SEXO,
    seed=SEED,
)

inicio = time.time()
busqueda_logistica = mo.buscar_hiperparametros(
    pipeline_logistica,
    mo.ESPACIOS_BUSQUEDA["logistica"],
    X_train,
    y_train,
    folds,
    n_jobs=1,
)
tiempo_logistica = time.time() - inicio

tabla_logistica = mo.tabla_busqueda(busqueda_logistica, "logistica")
display(tabla_logistica)
ruta_busqueda_logistica = RESULTADOS / "03_busqueda_logistica.csv"
tabla_logistica.to_csv(
    ruta_busqueda_logistica, index=False, float_format="%.6f"
)

print(f"Tiempo de búsqueda: {tiempo_logistica:.2f} s")
print(f"Mejores hiperparámetros: {busqueda_logistica.best_params_}")
print(f"AUC CV medio del mejor: {busqueda_logistica.best_score_:.6f}")
print(f"Guardado en: {ruta_busqueda_logistica}")

# %% [markdown]
# La tabla muestra las 4 combinaciones de `C`. El ranking se ordena por AUC
# medio de validación cruzada, no por AUC de prueba. La mejor combinación es la
# primera fila; las diferencias entre `C = 0.1` y `C = 10` son muy pequeñas, así
# que la elección es estable pero no dramática. Esta búsqueda no vio el 30% de
# prueba.

# %%
pipeline_hist_gb = mo.construir_pipeline(
    variante_raros=mo.VARIANTE_RAROS,
    variante_pay=mo.VARIANTE_PAY_POR_MODELO["hist_gb"],
    modelo="hist_gb",
    columnas_pay=columnas_pay,
    incluir_sexo=mo.INCLUIR_SEXO,
    seed=SEED,
)

inicio = time.time()
busqueda_hist_gb = mo.buscar_hiperparametros(
    pipeline_hist_gb,
    mo.ESPACIOS_BUSQUEDA["hist_gb"],
    X_train,
    y_train,
    folds,
    n_jobs=1,
)
tiempo_hist_gb = time.time() - inicio

tabla_hist_gb = mo.tabla_busqueda(busqueda_hist_gb, "hist_gb")
display(tabla_hist_gb)
ruta_busqueda_hist_gb = RESULTADOS / "03_busqueda_hist_gb.csv"
tabla_hist_gb.to_csv(
    ruta_busqueda_hist_gb, index=False, float_format="%.6f"
)

print(f"Tiempo de búsqueda: {tiempo_hist_gb:.2f} s")
print(f"Mejores hiperparámetros: {busqueda_hist_gb.best_params_}")
print(f"AUC CV medio del mejor: {busqueda_hist_gb.best_score_:.6f}")
print(f"Guardado en: {ruta_busqueda_hist_gb}")

# %% [markdown]
# La tabla muestra las 24 combinaciones de `HistGradientBoostingClassifier`.
# `max_depth = None` (mostrado como `None` en la columna `max_depth`) significa
# que el árbol no tiene un límite explícito de profundidad y se controla con
# `max_leaf_nodes`. El ranking vuelve a ser por
# AUC medio de validación cruzada. La mejor combinación aparece en la primera
# fila; las diferencias con las siguientes son del orden de milésimas, así que
# la elección debe leerse junto con la desviación estándar entre particiones.
# Esta búsqueda tampoco vio el 30% de prueba.

# %% [markdown]
# ## 4. Reentrenamiento y AUC de prueba (una sola vez por modelo)
#
# `GridSearchCV` ya reentrenó el mejor Pipeline con todo el 70% de
# entrenamiento. Aquí se usa ese Pipeline para predecir el 30% de prueba una
# sola vez y calcular su AUC. Ese AUC se reporta, pero **no** se usa para
# elegir el mejor modelo: la elección se hace por AUC de validación cruzada.

# %%
aucs_test: dict[str, float] = {}
probabilidades_test: dict[str, np.ndarray] = {}

for modelo, busqueda in [
    ("logistica", busqueda_logistica),
    ("hist_gb", busqueda_hist_gb),
]:
    mejor_pipeline = busqueda.best_estimator_
    probabilidades = mejor_pipeline.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, probabilidades)
    aucs_test[modelo] = float(auc_test)
    probabilidades_test[modelo] = probabilidades
    print(
        f"{mo.ETIQUETAS_MODELO[modelo]}: "
        f"AUC de prueba = {auc_test:.6f}"
    )

# %% [markdown]
# Los dos AUC de prueba se calcularon una sola vez, con el Pipeline ya
# reentrenado sobre el 70% completo. No se ajustó ningún hiperparámetro mirando
# estos valores. Si el modelo con mejor AUC de validación cruzada no es el de
# mejor AUC de prueba, la elección para los siguientes incisos sigue siendo el
# de validación cruzada.

# %% [markdown]
# ## 5. Resumen de modelos e hiperparámetros elegidos
#
# La tabla reúne, por modelo, la variante de códigos raros, la variante de
# `PAY_*`, los hiperparámetros elegidos, el AUC medio de validación cruzada con
# su desviación estándar y el AUC de prueba. La columna `mejor_por_auc_cv`
# marca el modelo que se usará en los siguientes incisos.

# %%
resumen = mo.resumen_modelos(
    {
        "logistica": busqueda_logistica,
        "hist_gb": busqueda_hist_gb,
    },
    aucs_test,
)
display(resumen)

ruta_resumen = RESULTADOS / "03_resumen_modelos.csv"
resumen.to_csv(ruta_resumen, index=False, float_format="%.6f")

mejor = resumen.loc[resumen["mejor_por_auc_cv"] == "sí"].iloc[0]
ruta_mejor = RESULTADOS / "03_mejor_modelo.csv"
mejor.to_frame().T.to_csv(ruta_mejor, index=False)

print(f"Resumen guardado en: {ruta_resumen}")
print(f"Mejor modelo por AUC de CV guardado en: {ruta_mejor}")
print(
    "Mejor modelo por AUC de CV: "
    f"{mejor['modelo']} "
    f"(AUC CV = {mejor['auc_cv_medio']:.6f} ± {mejor['auc_cv_std']:.6f})"
)

# %% [markdown]
# La tabla muestra que el modelo marcado con `mejor_por_auc_cv = sí` es el de
# mayor AUC medio de validación cruzada. El AUC de prueba se incluye solo como
# reporte único y no participa en la selección. La desviación estándar entre
# particiones ayuda a juzgar si las diferencias entre modelos son grandes o
# están dentro del ruido de la validación.

# %% [markdown]
# ## 6. Probabilidades de prueba para auditoría
#
# Se guardan las probabilidades de prueba de ambos modelos junto con la
# variable objetivo y `SEX`. `SEX` se incluye **solo para auditoría** por
# grupo; no fue una columna de entrada del modelo. El archivo conserva el
# índice posicional original para poder unirlo con el split guardado.

# %%
ruta_probabilidades = RESULTADOS / "03_probabilidades_test.csv"
tabla_probabilidades = mo.guardar_probabilidades_test(
    df,
    idx_test,
    TARGET,
    probabilidades_test,
    ruta_probabilidades,
)
display(tabla_probabilidades.head(10))

print(f"Probabilidades guardadas en: {ruta_probabilidades}")
print(f"Filas guardadas: {len(tabla_probabilidades)}")
print(f"Columnas: {list(tabla_probabilidades.columns)}")

# %% [markdown]
# La tabla mostrada son las primeras 10 filas de las 9,000 de prueba. El
# archivo completo tiene el índice original, el objetivo, `SEX` y las
# probabilidades de ambos modelos. `SEX` permite comparar el desempeño por
# grupo más adelante, pero no se usó para predecir. Las probabilidades se
# guardaron una sola vez, después de la búsqueda y del reentrenamiento.

# %% [markdown]
# ## 7. Cierre del paso
#
# - Se aplicaron las tres decisiones registradas: códigos raros agrupados,
#   `SEX` fuera del modelo principal y variantes de `PAY_*` por modelo.
# - La búsqueda de hiperparámetros usó solo el 70% de entrenamiento, con 5
#   particiones estratificadas y AUC.
# - El AUC de prueba se calculó una sola vez por modelo.
# - El modelo elegido para los siguientes incisos es el de mayor AUC de
#   validación cruzada, no el de mayor AUC de prueba.
# - Los resultados quedaron en `resultados/`: búsquedas, resumen, mejor modelo
#   y probabilidades de prueba.
#
# Esta libreta no pasa al Problema 2.
