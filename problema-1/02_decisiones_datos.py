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
# # Problema 1 — Evidencia para decisiones de datos
#
# Esta libreta **no decide** qué hacer con los códigos raros ni con `PAY_*`.
# Solo produce la evidencia descriptiva y las métricas de validación cruzada
# para que el alumno tome la decisión.
#
# Reglas que respeta la libreta:
#
# - El 70% de entrenamiento se usa para validar; el 30% de prueba **no se toca**.
# - La semilla común se importa desde `common.py` (`SEED = 223210350`).
# - Todas las transformaciones de columnas van dentro de un `Pipeline`.
# - No se hace búsqueda de hiperparámetros.
# - No se elige una variante como recomendación final.

# %%
"""Configuración, rutas e importaciones de la libreta."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from IPython.display import display


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

from common import SEED
from data import TARGET, cargar_dataframe
import evidencia_datos as ev

RESULTADOS = _RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)

print(f"SEED = {SEED}")
print(f"Variable objetivo = '{TARGET}'")
print(f"Directorio de resultados = {RESULTADOS}")

# %% [markdown]
# ## Carga de datos
#
# `cargar_dataframe()` descarga el archivo de UCI si no existe y corrige el
# encabezado desplazado. Aquí solo verificamos dimensiones y columnas PAY_*
# disponibles; no se imputa ni se recodifica ningún valor.

# %%
df = cargar_dataframe()
columnas_pay = ev.obtener_columnas_pay(df)

# ID es un identificador y la variable objetivo no puede ser predictora.
FEATURES = [c for c in df.columns if c not in ("ID", TARGET)]

print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Columnas PAY_* disponibles: {columnas_pay}")
print(f"Variables de entrada ({len(FEATURES)}): {FEATURES}")
print(f"Distribución del objetivo: {df[TARGET].value_counts().to_dict()}")

# %% [markdown]
# ## 1. Descriptivo de códigos raros y PAY_0
#
# Cada tabla muestra, por código: `n` (cuántas filas tienen ese código), la
# tasa de incumplimiento y la composición por sexo (`n_hombres`, `n_mujeres`,
# `pct_hombres`, `pct_mujeres`). La columna `en_descripcion_oficial` marca
# los códigos que el enunciado señala como no documentados. Después de cada
# tabla hay un comentario con su lectura.

# %%
print("EDUCATION — tabla por código")
tabla_education = ev.tabla_descriptiva(
    df, "EDUCATION", ev.CODIGOS_RAROS["EDUCATION"], TARGET
)
display(tabla_education)

# %% [markdown]
# En `EDUCATION`, los códigos 0, 5 y 6 son los no documentados. La tabla
# permite ver cuántas filas tiene cada uno y si su tasa de incumplimiento se
# separa de la de los códigos 1–4. La composición por sexo ayuda a detectar si
# esos códigos se concentran en hombres o en mujeres; con `n` pequeños, las
# tasas son más inestables.

# %%
print("EDUCATION — comparación documentados vs. no documentados")
display(
    ev.tabla_comparacion(
        df, "EDUCATION", ev.CODIGOS_RAROS["EDUCATION"], TARGET
    )
)

# %% [markdown]
# La comparación agrupa todos los códigos no documentados de `EDUCATION` en una
# sola fila y todos los documentados en otra. Si el grupo no documentado tiene
# una tasa parecida a la documentada, agruparlo o conservarlo como categoría
# propia no cambiaría mucho el patrón descriptivo; si la tasa es muy distinta,
# la decisión sí puede afectar al modelo.

# %%
print("MARRIAGE — tabla por código")
tabla_marriage = ev.tabla_descriptiva(
    df, "MARRIAGE", ev.CODIGOS_RAROS["MARRIAGE"], TARGET
)
display(tabla_marriage)

# %% [markdown]
# En `MARRIAGE`, el código no documentado es el 0. La tabla permite comparar
# su frecuencia, su tasa de incumplimiento y su mezcla por sexo contra los
# códigos 1–3. Si el código 0 tiene pocas filas, cualquier diferencia en la
# tasa debe leerse con cautela.

# %%
print("MARRIAGE — comparación documentados vs. no documentados")
display(
    ev.tabla_comparacion(
        df, "MARRIAGE", ev.CODIGOS_RAROS["MARRIAGE"], TARGET
    )
)

# %% [markdown]
# La comparación documentados vs. no documentados de `MARRIAGE` resume cuánto
# pesa el código 0 en la muestra y si su tasa de incumplimiento se aleja de la
# de los códigos oficiales. Es evidencia descriptiva; no decide si el código 0
# debe agruparse, eliminarse o mantenerse como categoría propia.

# %%
print("PAY_0 — tabla por valor")
tabla_pay_0 = ev.tabla_descriptiva(
    df, "PAY_0", ev.CODIGOS_RAROS_PAY_0, TARGET
)
display(tabla_pay_0)

# %% [markdown]
# En `PAY_0`, los valores no documentados son `-2` y `0`. La tabla muestra
# todos los valores observados, incluidos los meses de atraso documentados
# (`-1` y `1` en adelante). Así se puede comparar si los estados no
# documentados tienen una tasa de incumplimiento distinta y si su proporción
# de hombres y mujeres cambia respecto a los estados documentados.

# %%
print("PAY_0 — comparación documentados vs. no documentados")
display(
    ev.tabla_comparacion(
        df, "PAY_0", ev.CODIGOS_RAROS_PAY_0, TARGET
    )
)

# %% [markdown]
# La comparación de `PAY_0` junta los estados `-2` y `0` y los contrasta con el
# resto de los valores. Una diferencia grande de tasa sugiere que el
# tratamiento de `PAY_0` (numérico tal cual vs. indicadores de estado) puede
# importar; una diferencia pequeña sugiere que ambas variantes describen
# patrones parecidos.

# %% [markdown]
# ## 2. Split estratificado 70/30
#
# Se estratifica por la variable objetivo para conservar la proporción de
# incumplimientos. Los índices se guardan en
# `resultados/split_indices.csv`. A partir de este punto, la libreta solo usa
# el 70% de entrenamiento: el 30% de prueba no se consulta.

# %%
idx_train, idx_test = ev.dividir_estratificado(
    df, TARGET, test_size=0.30, seed=SEED
)
tabla_indices = ev.guardar_indices(
    df,
    idx_train,
    idx_test,
    TARGET,
    RESULTADOS / "split_indices.csv",
)

X_train = df.iloc[idx_train][FEATURES].reset_index(drop=True)
y_train = df.iloc[idx_train][TARGET].reset_index(drop=True)

print(f"Entrenamiento: {len(idx_train)} filas")
print(f"Prueba reservada: {len(idx_test)} filas (no se usa en esta libreta)")
print(f"Tasa de incumplimiento en entrenamiento: {y_train.mean():.4%}")
print(f"Archivo de índices: {RESULTADOS / 'split_indices.csv'}")
display(tabla_indices["conjunto"].value_counts().to_frame("filas"))

# %% [markdown]
# La tabla confirma que el 70% quedó etiquetado como `train` y el 30% como
# `test`. Esa columna `test` solo se guarda como referencia; en las secciones
# siguientes no se calcula ninguna métrica sobre ella.

# %% [markdown]
# ## 3. Significado de códigos no documentados
#
# Esta sección usa **solo el 70% de entrenamiento** (`X_train`/`y_train`).
# No recodifica, imputa ni agrupa ningún valor: solo describe qué filas tienen
# cada código y cómo se relacionan con facturación, pago, edad, límite de
# crédito y sexo. Las tasas y los intervalos se muestran en porcentaje. El
# intervalo de Wilson se usa porque es más estable con muestras chicas que la
# aproximación normal.
#
# El dataset no incluye un diccionario oficial de los códigos no documentados,
# así que estas tablas no permiten afirmar qué "significa" cada uno.

# %%
# Vista del 70% de entrenamiento con la variable objetivo incluida.
# No se recodifica nada y no se consulta el 30% de prueba.
df_train = X_train.copy()
df_train[TARGET] = y_train

# %%
print("PAY_0 — códigos -2, -1, 0, 1, 2 (solo entrenamiento)")
tabla_significado_pay_0 = ev.tabla_pay_significado(
    df_train,
    columna_pay="PAY_0",
    columna_bill="BILL_AMT1",
    columna_pay_amt="PAY_AMT1",
    objetivo=TARGET,
)
display(tabla_significado_pay_0)

ruta_significado_pay_0 = RESULTADOS / "significado_pay_0.csv"
tabla_significado_pay_0.to_csv(
    ruta_significado_pay_0, index=False, float_format="%.6f"
)
print(f"Guardado en: {ruta_significado_pay_0}")

# %% [markdown]
# **Qué sugiere.** Los códigos `-2`, `-1` y `0` tienen tasas de
# incumplimiento menores (12.9%, 17.1% y 12.5%) que `1` y `2` (33.9% y
# 69.9%). También cambia el perfil de facturación y pago: `0` tiene una
# mediana de `BILL_AMT1` de 50,094 y casi ninguna factura no positiva; `-2`
# tiene 31.7% de facturas no positivas y una mediana de 1,289; `1` tiene 46.1%
# de facturas no positivas y mediana de `PAY_AMT1` igual a 0; `2` tiene una
# mediana de factura de 41,282.5 y una razón pago/factura cercana a 0.05. Eso
# es compatible con que los códigos negativos/cero distingan estados de
# facturación o de atraso, pero es solo una lectura descriptiva.
#
# **Qué NO se puede concluir.** No hay un diccionario oficial que explique los
# códigos no documentados (`-2` y `0`), así que no se puede afirmar que `-2`
# sea "sin consumo" o que `0` sea "sin atraso" por definición. Las diferencias
# son asociaciones, no efectos causales; además, `PAY_0` y las
# facturas/pagos provienen del mismo historial, por lo que parte de la
# relación es mecánica. Los códigos `1` y `2` tienen menos filas que `0`, y
# la razón pago/factura solo usa las filas con factura positiva.

# %%
print("PAY_2 — verificación con BILL_AMT2 y PAY_AMT2 (solo entrenamiento)")
tabla_significado_pay_2 = ev.tabla_pay_significado(
    df_train,
    columna_pay="PAY_2",
    columna_bill="BILL_AMT2",
    columna_pay_amt="PAY_AMT2",
    objetivo=TARGET,
)
display(tabla_significado_pay_2)

ruta_significado_pay_2 = RESULTADOS / "significado_pay_2.csv"
tabla_significado_pay_2.to_csv(
    ruta_significado_pay_2, index=False, float_format="%.6f"
)
print(f"Guardado en: {ruta_significado_pay_2}")

# %% [markdown]
# **Qué sugiere.** La verificación con `PAY_2` repite parte del patrón:
# `-2`, `-1` y `0` tienen tasas de incumplimiento menores (18.0%, 16.4% y
# 15.7%) que `2` (55.9%); `-2` concentra 55.4% de facturas no positivas y
# mediana de factura 0, mientras que `0` tiene 1.8% de facturas no positivas y
# mediana 48,703.5. La razón pago/factura vuelve a ser alta para `-2`/`-1`
# (0.87/0.96) y baja para `0`/`2` (0.05). Que el patrón se repita en otro
# periodo hace más consistente la lectura de que los códigos negativos/cero
# se asocian a estados de facturación distintos.
#
# **Qué NO se puede concluir.** El código `1` de `PAY_2` tiene solo 19 filas,
# así que su tasa (15.8%) y su 73.7% de facturas no positivas son muy
# inestables y no permiten generalizar. Sigue sin haber un diccionario oficial
# para los códigos no documentados (`-2` y `0`): la repetición del patrón no
# prueba el significado del código. También es una asociación descriptiva, no
# causal, y `PAY_2`/`BILL_AMT2`/`PAY_AMT2` comparten el mismo historial de la
# persona.

# %%
print("EDUCATION — códigos 0 a 6 (solo entrenamiento)")
tabla_significado_education = ev.tabla_categoria_significado(
    df_train,
    columna="EDUCATION",
    codigos=[0, 1, 2, 3, 4, 5, 6],
    objetivo=TARGET,
)
display(tabla_significado_education)

ruta_significado_education = RESULTADOS / "significado_education.csv"
tabla_significado_education.to_csv(
    ruta_significado_education, index=False, float_format="%.6f"
)
print(f"Guardado en: {ruta_significado_education}")

# %% [markdown]
# **Qué sugiere.** Los códigos `0`, `4`, `5` y `6` tienen tasas de
# incumplimiento bajas (0.0%, 6.8%, 5.2% y 23.1%), pero sus muestras son
# pequeñas: 10, 74, 194 y 26 filas. Por eso sus intervalos de Wilson son
# anchos: 0–27.8%, 2.9–14.9%, 2.8–9.2% y 11.0–42.1%. Los códigos `1`, `2` y
# `3`, con miles de filas, tienen tasas de 19.2%, 23.9% y 24.9% e intervalos
# mucho más estrechos. También cambian la edad media (34.0–41.5 años), la
# mediana de `LIMIT_BAL` (80,000–220,000) y la proporción de mujeres
# (50.0–68.6%), lo que sugiere que los códigos raros no son solo "otro nivel"
# de educación, sino grupos con composición distinta.
#
# **Qué NO se puede concluir.** Con `n` tan chico, las tasas de `0`, `4`, `5`
# y `6` son compatibles con muchos valores dentro de su intervalo; no se
# puede afirmar que su tasa "real" sea menor. No hay un diccionario oficial
# que explique con precisión los códigos `0`, `5` y `6` (el `4` aparece como
# "others"), y las diferencias son asociaciones, no efectos causales. No se
# puede concluir que agrupar, eliminar o dejar como categoría propia sea
# mejor: esta tabla solo describe.

# %%
print("MARRIAGE — códigos 0 a 3 (solo entrenamiento)")
tabla_significado_marriage = ev.tabla_categoria_significado(
    df_train,
    columna="MARRIAGE",
    codigos=[0, 1, 2, 3],
    objetivo=TARGET,
)
display(tabla_significado_marriage)

ruta_significado_marriage = RESULTADOS / "significado_marriage.csv"
tabla_significado_marriage.to_csv(
    ruta_significado_marriage, index=False, float_format="%.6f"
)
print(f"Guardado en: {ruta_significado_marriage}")

# %% [markdown]
# **Qué sugiere.** El código `0` tiene 41 filas, una tasa de incumplimiento
# de 12.2% (IC 95%: 5.3–25.5%), 78.0% de mujeres, edad media 37.7 años y
# mediana de `LIMIT_BAL` de 150,000. Los códigos `1`, `2` y `3` tienen tasas
# de 23.3%, 21.0% y 27.2%; el código `3` tiene solo 250 filas y un intervalo
# de 22.1–33.0%, que se traslapa con los códigos `1` y `2`. La composición
# por sexo y el límite de crédito también cambian entre códigos.
#
# **Qué NO se puede concluir.** El código `0` tiene pocas filas y un
# intervalo ancho; no se puede afirmar que su tasa sea realmente menor que la
# de los códigos documentados. El código `3` también tiene un intervalo que
# se traslapa con `1` y `2`, así que no hay separación clara. No hay
# diccionario oficial para el `0`, y las diferencias son asociaciones, no
# causalidad. Esta tabla no decide si el `0` debe agruparse, eliminarse o
# conservarse.

# %% [markdown]
# ## 4. Variantes con validación cruzada de 5 particiones
#
# Se evalúan las 12 combinaciones: 3 variantes de códigos raros × 2 variantes
# de `PAY_*` × 2 modelos. Todas usan las mismas 5 particiones estratificadas
# del 70% de entrenamiento, generadas con `SEED`.
#
# - **Códigos raros A:** agrupa `EDUCATION {0,5,6} -> 4` y `MARRIAGE {0} -> 3`.
# - **Códigos raros B:** elimina esas filas solo del entrenamiento de cada
#   fold; la validación conserva todas las filas para que los AUC sean
#   comparables.
# - **Códigos raros C:** deja los códigos raros como categorías propias
#   (one-hot).
# - **PAY_* A:** usa los valores numéricos tal cual.
# - **PAY_* B:** crea indicadores para `-2`, `-1`, `0` y una columna de meses
#   de atraso para valores `>= 1`.
#
# Cada modelo se ajusta dentro de un `Pipeline` nuevo en cada fold. No se
# buscan hiperparámetros: se usan los valores por defecto de scikit-learn,
# salvo `max_iter` en la regresión logística para asegurar convergencia.

# %%
folds = ev.crear_folds(X_train, y_train, n_splits=5, seed=SEED)
tabla_combinaciones = ev.evaluar_combinaciones(
    X_train,
    y_train,
    folds,
    columnas_pay,
    seed=SEED,
)
display(tabla_combinaciones)

# %% [markdown]
# La tabla muestra el AUC medio y su desviación estándar para cada combinación.
# La desviación estándar se calcula sobre los 5 valores de AUC de las
# particiones (`ddof=0`). No se declara una ganadora: las diferencias entre
# variantes deben compararse con la dispersión entre folds. Una desviación
# estándar grande indica que el resultado depende de la partición; una
# diferencia pequeña entre variantes no justifica por sí sola una decisión.

# %% [markdown]
# ## 5. Comparación con y sin SEX
#
# Para continuar con la comparación solicitada se toma la combinación con el
# **mayor AUC medio** de la tabla anterior. Esto es solo un criterio empírico
# para elegir sobre qué combinación repetir el experimento; no es una
# recomendación de modelo ni una decisión sobre los datos.

# %%
mejor_idx = tabla_combinaciones["auc_medio"].idxmax()
mejor = tabla_combinaciones.loc[mejor_idx]
tabla_combinaciones["es_mejor_auc"] = False
tabla_combinaciones.loc[mejor_idx, "es_mejor_auc"] = True

print("Combinación con mayor AUC medio (solo para el paso 4):")
display(mejor.to_frame().T)

tabla_sexo, diferencia_auc = ev.evaluar_sexo(
    X_train,
    y_train,
    folds,
    mejor,
    columnas_pay,
    seed=SEED,
)
tabla_sexo["es_mejor_auc"] = False

display(tabla_sexo)
print(f"Diferencia de AUC (con SEX - sin SEX) = {diferencia_auc:.6f}")

# %% [markdown]
# La diferencia de AUC cuantifica cuánto cambia el desempeño al quitar `SEX`
# de las entradas, manteniendo la misma combinación y las mismas particiones.
# Un valor positivo indica que el AUC fue mayor al incluir `SEX`; uno negativo,
# que fue mayor al excluirlo. Esa diferencia debe compararse con la desviación
# estándar entre folds: si es mucho menor, no hay evidencia de una mejora
# clara. Esta libreta solo reporta la diferencia: no decide si `SEX` debe
# quedarse.

# %% [markdown]
# ## 6. Tabla final
#
# Se concatenan las 12 combinaciones y las 2 filas de la comparación con/sin
# `SEX`. La columna `es_mejor_auc` marca la combinación de mayor AUC medio.
# El archivo se guarda en `resultados/decisiones_datos.csv`.

# %%
tabla_final = pd.concat(
    [tabla_combinaciones, tabla_sexo],
    ignore_index=True,
    sort=False,
)
ruta_final = RESULTADOS / "decisiones_datos.csv"
tabla_final.to_csv(ruta_final, index=False, float_format="%.6f")

print(f"Tabla final guardada en: {ruta_final}")
display(tabla_final)

# %% [markdown]
# La tabla final reúne la evidencia de las 12 combinaciones y del contraste
# con/sin `SEX`. **No contiene una recomendación**: la decisión sobre qué
# variante usar, si eliminar filas o si conservar `SEX` queda en manos del
# alumno. Cualquier lectura de la tabla debe considerar el AUC medio junto con
# su desviación estándar entre folds.
