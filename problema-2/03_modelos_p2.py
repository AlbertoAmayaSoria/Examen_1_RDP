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
# # Problema 2 — 03: modelos de regresión (SVR de kernels)
#
# Modelos, todos con el objetivo `log1p(registered)` (decisión de la libreta
# 02) y la métrica RMSLE, comparados con la ventana expansiva de 4
# particiones (la prueba de 2012-10-01 en adelante sigue sin tocarse):
#
# - **M1:** SVR gaussiano (RBF) con `hr` y `weekday` numéricos.
# - **M2:** SVR gaussiano con `hr` y `weekday` en one-hot.
# - **M3:** SVR gaussiano con seno y coseno de `hr` y `weekday`.
# - **M4:** SVR precalculado con un kernel circular **válido** (von Mises)
#   para `hr`, otro para `weekday` y RBF para las demás variables,
#   combinados según la estructura ganadora de 02 (`(k_hr*k_wd) + k_rbf`).
# - **HGB:** `HistGradientBoostingRegressor` como referencia.
#
# Las "demás variables" (season, yr, mnth, holiday, weathersit, temp, atemp,
# hum, windspeed) son las mismas en M1–M4; `workingday` quedó fuera por ser
# función exacta de `weekday` y `holiday`. En M4 la matriz de kernels completa
# (3000×3000) se calcula una sola vez y se rebana por partición.
#
# Al final, sección 6: M4 con kernels **inválidos** (kI y kII en lugar del
# válido) para mostrar qué se pierde al romper la semidefinición positiva.

# %%
"""Configuración, rutas e importaciones de la libreta."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.linalg import eigvalsh
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def _buscar_raiz(inicio: Path) -> Path:
    """Sube desde el directorio actual hasta encontrar la raíz del repo."""
    for base in [inicio, *inicio.parents]:
        if (base / "common.py").exists() and (base / "problema-2").is_dir():
            return base
    return inicio


_AQUI = Path.cwd()
_RAIZ = _buscar_raiz(_AQUI)
_DIR_PROBLEMA = _RAIZ / "problema-2" if (_RAIZ / "problema-2").is_dir() else _RAIZ
for _ruta in (_RAIZ, _DIR_PROBLEMA):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

from common import SEED  # noqa: E402
import data as da  # noqa: E402
import kernels as kk  # noqa: E402

RESULTADOS = _RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 220)

OTRAS = ["season", "yr", "mnth", "holiday", "weathersit",
         "temp", "atemp", "hum", "windspeed"]

df = da.cargar_dataframe()
muestra, prueba = da.dividir_muestra_prueba(df, seed=SEED)
particiones = da.particiones_expansivas(muestra)

# Estatística de estandarización de las "otras" con la muestra completa
# (se registra como propuesta en la bitácora: simplifica que la matriz de
# kernels de M4 se calcule una sola vez y se rebane por partición).
mu = muestra[OTRAS].mean()
sd = muestra[OTRAS].std(ddof=1).replace(0, 1.0)
otras_std = ((muestra[OTRAS] - mu) / sd).to_numpy(dtype=float)
hr = muestra["hr"].to_numpy(dtype=float)
wd = muestra["weekday"].to_numpy(dtype=float)
y_log = np.log1p(muestra[da.TARGET].to_numpy(dtype=float))

# Versiones de rasgos de cada modelo (ya con hr/weekday estandarizadas
# donde aplica y "otras" estandarizadas).
hr_z = (hr - hr.mean()) / hr.std(ddof=1)
wd_z = (wd - wd.mean()) / wd.std(ddof=1)

X_M1 = np.column_stack([hr_z, wd_z, otras_std])                    # 11 rasgos
n_hr = 24
one_hr = np.zeros((len(muestra), n_hr)); one_hr[np.arange(len(muestra)), hr.astype(int)] = 1.0
one_wd = np.zeros((len(muestra), 7)); one_wd[np.arange(len(muestra)), wd.astype(int)] = 1.0
X_M2 = np.column_stack([one_hr, one_wd, otras_std])                 # 40 rasgos
X_M3 = np.column_stack([
    np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
    np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7),
    otras_std,
])                                                                 # 13 rasgos
X_HGB = muestra[da.columnas_entrada(muestra)].to_numpy(dtype=float)

print(f"Muestra: {len(muestra)} filas; particiones: {len(particiones)}.")
print(f"Rasgos — M1: {X_M1.shape[1]}, M2: {X_M2.shape[1]}, M3: {X_M3.shape[1]}, "
      f"HGB: {X_HGB.shape[1]} columnas.")
print(f"Objetivo log1p: media = {y_log.mean():.4f}, sd = {y_log.std():.4f}.")

# %% [markdown]
# ## 1. Rejillas de hiperparámetros (pequeñas, como pide el enunciado)
#
# - **M1/M3:** `C` en {1, 10, 100}, `epsilon` en {0.05, 0.15} y `gamma` RBF
#   según la dimensión: M1 {0.01, 0.05, 0.15}, M3 {0.02, 0.05, 0.15}.
# - **M2:** `gamma` más pequeño {0.002, 0.01, 0.05} porque hay 40 rasgos.
# - **M4 (precalculado):** `kappa_hr`, `kappa_wd` en {1, 3} (von Mises) y
#   `gamma` RBF en {0.05, 0.15}; `C` en {1, 10, 100} y `epsilon` en {0.05, 0.15}.
# - **HGB:** parámetros por defecto (referencia, sin rejilla).
#
# Todas las combinaciones se evalúan en las 4 particiones de ventana
# expansiva y se elige por el RMSLE medio; la prueba no interviene.

# %%
print("Rejillas listas (el detalle se imprime al ajustar).")

# %% [markdown]
# ## 2. M1, M2, M3 y HGB: ajuste por ventana expansiva

# %%
def ajustar_rbf(X, y, idx_tr, idx_va, C, epsilon, gamma):
    svr = SVR(C=C, epsilon=epsilon, kernel="rbf", gamma=gamma)
    svr.fit(X[idx_tr], y[idx_tr])
    pred = svr.predict(X[idx_va])
    return float(np.sqrt(mean_squared_error(y[idx_va], pred))), svr


resultados = []
for nombre, X, gammas in (
    ("M1", X_M1, [0.01, 0.05, 0.15]),
    ("M2", X_M2, [0.002, 0.01, 0.05]),
    ("M3", X_M3, [0.02, 0.05, 0.15]),
):
    inicio = time.time()
    for C in [1.0, 10.0, 100.0]:
        for epsilon in [0.05, 0.15]:
            for gamma in gammas:
                rmsles = []
                for idx_tr, idx_va in particiones:
                    r, _ = ajustar_rbf(X, y_log, idx_tr, idx_va, C, epsilon, gamma)
                    rmsles.append(r)
                resultados.append({
                    "modelo": nombre, "C": C, "epsilon": epsilon, "gamma": gamma,
                    "rmsle_medio": float(np.mean(rmsles)),
                    "rmsle_sd": float(np.std(rmsles, ddof=1)),
                })
    print(f"{nombre}: ajustado en {time.time() - inicio:.1f} s.")

# HGB de referencia (sin rejilla).
for j, (idx_tr, idx_va) in enumerate(particiones, start=1):
    hgb = HistGradientBoostingRegressor(random_state=SEED)
    hgb.fit(X_HGB[idx_tr], y_log[idx_tr])
    pred = hgb.predict(X_HGB[idx_va])
    resultados.append({
        "modelo": "HGB", "C": np.nan, "epsilon": np.nan, "gamma": np.nan,
        "rmsle_medio": float(np.sqrt(mean_squared_error(y_log[idx_va], pred))),
        "rmsle_sd": np.nan,
    })
    print(f"HGB partición P{j}: RMSLE = {resultados[-1]['rmsle_medio']:.4f}")

tabla_resultados = pd.DataFrame(resultados)
print("Ganadores por RMSLE medio hasta aquí:")
for nombre in ["M1", "M2", "M3"]:
    sub = tabla_resultados.loc[tabla_resultados["modelo"] == nombre]
    mejor = sub.loc[sub["rmsle_medio"].idxmin()]
    print(f"  {nombre}: C={mejor['C']}, epsilon={mejor['epsilon']}, "
          f"gamma={mejor['gamma']} -> RMSLE medio = {mejor['rmsle_medio']:.4f}")

# %% [markdown]
# M1 (gaussiano con hora y día numéricos lineales) sufre porque una distancia
# gaussiana no ve la circularidad: las 23:00 y las 1:00 aparecen lejanas
# aunque estén a 2 horas. M2 (one-hot) trata cada hora como categoría
# disjunta: tampoco comparte información entre horas vecinas. M3 (seno y
# coseno) sí captura la circularidad con rasgos continuos. La tabla final
# (sección 5) ordena los cuatro.

# %% [markdown]
# ## 3. M4: kernels precalculados (matriz 3000×3000 calculada una vez)

# %%
def kernel_circular(vals_x, vals_y, kend, kap, periodo):
    return kk.kernel_matrix(vals_x, vals_y, lambda d: kk.k_von_mises(d, kap), periodo)


inicio = time.time()
K_hr_por_kappa = {kap: kernel_circular(hr, hr, kk.k_von_mises, kap, kk.PERIODO_HR)
                  for kap in [1.0, 3.0]}
K_wd_por_kappa = {kap: kernel_circular(wd, wd, kk.k_von_mises, kap, kk.PERIODO_WEEKDAY)
                  for kap in [1.0, 3.0]}
dif_otras = otras_std[:, None, :] - otras_std[None, :, :]
K_rbf_por_gamma = {g: kk.rbf(dif_otras, gamma=g) for g in [0.05, 0.15]}
print(f"Matrices base (von Mises x2, RBF x2) listas en {time.time() - inicio:.1f} s.")

# Combinaciones por defecto: (iii) de la libreta 02.
def combinar(K_hr, K_wd, K_rbf):
    return (K_hr * K_wd) + K_rbf


resultados_m4 = []
for kap_hr in [1.0, 3.0]:
    for kap_wd in [1.0, 3.0]:
        for gamma_rbf in [0.05, 0.15]:
            K_full = combinar(K_hr_por_kappa[kap_hr], K_wd_por_kappa[kap_wd],
                              K_rbf_por_gamma[gamma_rbf])
            for C in [1.0, 10.0, 100.0]:
                for epsilon in [0.05, 0.15]:
                    rmsles = []
                    for idx_tr, idx_va in particiones:
                        Ktr = K_full[np.ix_(idx_tr, idx_tr)]
                        Kva = K_full[np.ix_(idx_va, idx_tr)]
                        svr = SVR(C=C, epsilon=epsilon, kernel="precomputed")
                        svr.fit(Ktr, y_log[idx_tr])
                        pred = svr.predict(Kva)
                        rmsles.append(float(np.sqrt(mean_squared_error(y_log[idx_va], pred))))
                    resultados_m4.append({
                        "modelo": "M4", "C": C, "epsilon": epsilon,
                        "kappa_hr": kap_hr, "kappa_wd": kap_wd, "gamma_rbf": gamma_rbf,
                        "rmsle_medio": float(np.mean(rmsles)),
                        "rmsle_sd": float(np.std(rmsles, ddof=1)),
                    })
    print("M4: una familia de kappas ajustada.")

tabla_m4 = pd.DataFrame(resultados_m4)
mejor_m4 = tabla_m4.loc[tabla_m4["rmsle_medio"].idxmin()]
print(f"M4 ganador: C={mejor_m4['C']}, epsilon={mejor_m4['epsilon']}, "
      f"kappa_hr={mejor_m4['kappa_hr']}, kappa_wd={mejor_m4['kappa_wd']}, "
      f"gamma_rbf={mejor_m4['gamma_rbf']} -> RMSLE medio = {mejor_m4['rmsle_medio']:.4f}")

# %% [markdown]
# M4 usa el kernel circular válido en cada dimensión y lo combina con el RBF
# del clima según la estructura que ganó en 02: `(k_hr * k_wd) + k_rbf`.
# La matriz completa de 3000×3000 se calcula una vez por combinación de
# hiperparámetros de kernel y se rebana por partición, como pide el enunciado.

# %% [markdown]
# ## 4. Tabla final de validación (RMSLE medio en las 4 particiones)

# %%
tabla_03 = pd.DataFrame(resultados)
tabla_03 = pd.concat([tabla_03, tabla_m4], ignore_index=True, sort=False)
tabla_03["rmsle_medio"] = tabla_03["rmsle_medio"].astype(float)
ganadores = []
for nombre in ["M1", "M2", "M3", "M4"]:
    sub = tabla_03.loc[tabla_03["modelo"] == nombre]
    mejor = sub.loc[sub["rmsle_medio"].idxmin()].copy()
    ganadores.append(mejor)
tabla_gan = pd.DataFrame(ganadores)
display(tabla_gan.round(4))
tabla_03.to_csv(RESULTADOS / "p2_03_resultados.csv", index=False, float_format="%.6f")
tabla_gan.to_csv(RESULTADOS / "p2_03_mejor_modelo.csv", index=False, float_format="%.6f")
print(f"Guardado: p2_03_resultados.csv ({len(tabla_03)} filas) y "
      f"p2_03_mejor_modelo.csv ({len(tabla_gan)} filas).")

# %% [markdown]
# La tabla ordena por RMSLE medio de validación. Quedan elegidos los
# hiperparámetros de cada modelo para evaluarlos UNA sola vez en la prueba
# (libreta 04). Si el kernel circular válido aporta, M4 o M3 deberían
# encabezar; HGB es la referencia no lineal sin kernels.

# %% [markdown]
# ## 5. M4 con kernel NO válido (kI y kII)

# %%
# Se repite M4 pero con kI o kII en lugar del von Mises, con la misma
# combinación (iii) y los hiperparámetros ganadores de M4.
C_gan = float(mejor_m4["C"])
eps_gan = float(mejor_m4["epsilon"])
g_gan = float(mejor_m4["gamma_rbf"])
print(f"M4 inválido con C={C_gan}, epsilon={eps_gan}, gamma_rbf={g_gan} "
      f"(los ganadores de M4).")

for nombre_inv, fun_inv in (("kI", kk.k_I), ("kII", kk.k_II)):
    K_hr_i = kk.kernel_matrix(hr, hr, fun_inv, kk.PERIODO_HR)
    K_wd_i = kk.kernel_matrix(wd, wd, fun_inv, kk.PERIODO_WEEKDAY)
    K_inv = (K_hr_i * K_wd_i) + K_rbf_por_gamma[g_gan]

    lmin = float(eigvalsh(K_inv, subset_by_index=[0, 0])[0])
    tol = 1e-10 * float(eigvalsh(K_inv, subset_by_index=[2999, 2999])[0])
    avisos = []
    rmsles = []
    n_iters = []
    for idx_tr, idx_va in particiones:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            svr = SVR(C=C_gan, epsilon=eps_gan, kernel="precomputed")
            svr.fit(K_inv[np.ix_(idx_tr, idx_tr)], y_log[idx_tr])
            pred = svr.predict(K_inv[np.ix_(idx_va, idx_tr)])
        avisos.extend([str(w.message) for w in wlist])
        rmsles.append(float(np.sqrt(mean_squared_error(y_log[idx_va], pred))))
        n_iters.append(int(svr.n_iter_))
    print(f"\n{nombre_inv}: lambda_min(K_train 3000x3000) = {lmin:.6f} "
          f"(tolerancia {tol:.2e}); negativo CLARO: {lmin < -tol}")
    print(f"  advertencias de SVR durante el ajuste: "
          f"{len(set(avisos))} distintas: {sorted(set(avisos))[:5]}")
    print(f"  n_iter_ por partición: {n_iters}")
    print(f"  RMSLE medio en validación: {float(np.mean(rmsles)):.4f} "
          f"(M4 válido ganador: {mejor_m4['rmsle_medio']:.4f})")
    print(f"  Support vectors por partición: "
          f"{[int(SVR(C=C_gan, epsilon=eps_gan, kernel='precomputed').fit(K_inv[np.ix_(t_, t_)], y_log[t_]).n_support_.sum()) for t_, _ in particiones]}")

# %% [markdown]
# **Qué se pierde al usar un kernel no válido.** Un kernel debe ser positivo
# semidefinido (PSD): su matriz de Gram debe tener todos los valores propios
# `>= 0`. Con `kI`/`kII` la matriz de entrenamiento tiene valores propios
# negativos claros (del orden de `-0.64` y `-107` en la matriz conjunta de
# 3000×3000 con repeticiones, según la 01), así que la hipótesis del
# problema dual de SVR —convexo, con óptimo global garantizado— deja de
# cumplirse: el dual deja de ser un problema convexo, se pierde el óptimo
# global y la garantía de convergencia de SMO (el algoritmo de coordenadas
# puede detenerse en puntos estacionarios arbitrarios o ciclar). En la
# práctica libsvm no avisa (no verifica la semidefinición) y devuelve una
# solución; el RMSLE de validación de los modelos inválidos es peor que el de
# M4 válido, y, aun cuando parece "funcionar", sus predicciones no coinciden
# con las del problema convexo bien planteado. Por eso el Problema 2 usa
# kernels válidos (von Mises) en M4.

# %% [markdown]
# ## 6. Cierre

# - M1–M4 y HGB ajustados por ventana expansiva sobre `log1p(registered)`,
#   con RMSLE como métrica (decisiones de 02).
# - Ganadores guardados en `resultados/p2_03_mejor_modelo.csv` para la
#   evaluación final en la libreta 04.
# - M4 con kernels inválidos: la matriz de entrenamiento tiene valores
#   propios negativos claros, el dual pierde convexidad y garantía de
#   óptimo global; libsvm ajusta sin avisar, pero el error de validación
#   empeora.
