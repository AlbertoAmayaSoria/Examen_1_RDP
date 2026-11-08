"""Funciones compartidas de auditoría de equidad del Problema 1.

Este módulo concentra las métricas por grupo, la ganancia del banco, la
búsqueda de umbrales por grupo y el bootstrap. Lo usan las libretas
``06_intervencion`` y ``07_sin_atributo`` para no duplicar código ni definiciones.

Convenciones:

- ``Y = 1`` significa que el cliente incumplió.
- ``rechazo = True`` significa que el banco predijo positivo (no otorgó el
  crédito).
- Se aprueba cuando ``score < umbral`` y se rechaza cuando
  ``score >= umbral``.
- La ganancia del banco es ``+G`` por cada aprobado que paga y ``-L`` por cada
  aprobado que incumple. Un rechazado aporta ``0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ``common.py`` está en la raíz del repositorio.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import G, L, SEED, UMBRAL  # noqa: E402

# Nombres de grupo usados en todas las tablas.
GRUPO_HOMBRES = "Hombres (SEX=1)"
GRUPO_MUJERES = "Mujeres (SEX=2)"
GRUPO_TODOS = "Todos"

# Definiciones de equidad que evalúa el inciso (d).
DEFINICIONES = {
    "paridad_demografica": ["tasa_rechazo"],
    "igual_fpr": ["FPR"],
    "igual_fnr": ["FNR"],
    "igual_fpr_y_fnr": ["FPR", "FNR"],
    "paridad_predictiva": ["PPV"],
}

ETIQUETAS_DEFINICION = {
    "paridad_demografica": "Paridad demográfica (igual tasa de rechazo)",
    "igual_fpr": "Igual FPR",
    "igual_fnr": "Igual FNR",
    "igual_fpr_y_fnr": "Igual FPR y FNR",
    "paridad_predictiva": "Paridad predictiva (igual PPV)",
}


def metricas_grupo(
    y: np.ndarray,
    rechazo: np.ndarray,
    nombre: str,
    ganancia_G: float = G,
    perdida_L: float = L,
    avisos: list[str] | None = None,
) -> dict[str, float | int | str]:
    """Calcula la matriz de confusión, las métricas y la ganancia de un grupo.

    ``y`` son las etiquetas reales (1 = incumplió) y ``rechazo`` indica si el
    banco rechazó al cliente. Cuando un denominador es cero, la métrica se
    devuelve como ``NaN`` y se registra un aviso legible en ``avisos``.
    """
    y = np.asarray(y, dtype=int)
    rechazo = np.asarray(rechazo, dtype=bool)
    if avisos is None:
        avisos = []

    n = int(len(y))
    tp = int(np.sum((y == 1) & rechazo))
    fn = int(np.sum((y == 1) & ~rechazo))
    fp = int(np.sum((y == 0) & rechazo))
    tn = int(np.sum((y == 0) & ~rechazo))

    positivos = tp + fn
    negativos = fp + tn
    rechazados = tp + fp

    p_g = positivos / n if n > 0 else np.nan
    tasa_rechazo = rechazados / n if n > 0 else np.nan
    ppv = tp / rechazados if rechazados > 0 else np.nan
    fpr = fp / negativos if negativos > 0 else np.nan
    fnr = fn / positivos if positivos > 0 else np.nan
    ganancia = ganancia_G * tn - perdida_L * fn
    ganancia_por_solicitante = ganancia / n if n > 0 else np.nan

    if n == 0:
        avisos.append(f"{nombre}: n = 0; p_g y tasa de rechazo no definidas.")
    if rechazados == 0:
        avisos.append(f"{nombre}: TP + FP = 0; PPV no definida.")
    if negativos == 0:
        avisos.append(f"{nombre}: FP + TN = 0; FPR no definida.")
    if positivos == 0:
        avisos.append(f"{nombre}: TP + FN = 0; FNR no definida.")

    return {
        "grupo": nombre,
        "n": n,
        "p_g": float(p_g),
        "tasa_rechazo": float(tasa_rechazo),
        "PPV": float(ppv),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "ganancia": float(ganancia),
        "ganancia_por_solicitante": float(ganancia_por_solicitante),
    }


def tabla_grupos(
    y: np.ndarray,
    probabilidad: np.ndarray,
    sexo: np.ndarray,
    umbral: float = UMBRAL,
    ganancia_G: float = G,
    perdida_L: float = L,
) -> tuple[pd.DataFrame, list[str]]:
    """Tabla por grupo y total con un umbral único para todos."""
    avisos: list[str] = []
    rechazo = np.asarray(probabilidad) >= umbral
    filas = []
    for valor, nombre in ((1, GRUPO_HOMBRES), (2, GRUPO_MUJERES)):
        mascara = sexo == valor
        filas.append(
            metricas_grupo(
                y[mascara],
                rechazo[mascara],
                nombre,
                ganancia_G,
                perdida_L,
                avisos,
            )
        )
    filas.append(
        metricas_grupo(
            y, rechazo, GRUPO_TODOS, ganancia_G, perdida_L, avisos
        )
    )
    return pd.DataFrame(filas), avisos


def tabla_grupos_umbrales(
    y: np.ndarray,
    probabilidad: np.ndarray,
    sexo: np.ndarray,
    umbral_hombres: float,
    umbral_mujeres: float,
    ganancia_G: float = G,
    perdida_L: float = L,
) -> tuple[pd.DataFrame, list[str]]:
    """Tabla por grupo y total usando un umbral distinto por grupo."""
    avisos: list[str] = []
    rechazo = np.where(
        sexo == 1,
        probabilidad >= umbral_hombres,
        probabilidad >= umbral_mujeres,
    )
    filas = []
    for valor, nombre, umbral in (
        (1, GRUPO_HOMBRES, umbral_hombres),
        (2, GRUPO_MUJERES, umbral_mujeres),
    ):
        mascara = sexo == valor
        fila = metricas_grupo(
            y[mascara],
            rechazo[mascara],
            nombre,
            ganancia_G,
            perdida_L,
            avisos,
        )
        fila["umbral"] = float(umbral)
        filas.append(fila)
    total = metricas_grupo(
        y, rechazo, GRUPO_TODOS, ganancia_G, perdida_L, avisos
    )
    total["umbral"] = np.nan
    filas.append(total)
    return pd.DataFrame(filas), avisos


def tabla_umbrales_por_grupo(
    y: np.ndarray,
    probabilidad: np.ndarray,
    sexo: np.ndarray,
    umbrales: np.ndarray,
    ganancia_G: float = G,
    perdida_L: float = L,
) -> pd.DataFrame:
    """Evalúa cada umbral de la cuadrícula por separado en cada grupo.

    Devuelve una fila por ``(grupo, umbral)``. Es la base para buscar el par
    ``(t_H, t_M)`` sin recalcular la matriz de confusión en cada combinación.
    """
    filas = []
    for valor, nombre in ((1, GRUPO_HOMBRES), (2, GRUPO_MUJERES)):
        mascara = sexo == valor
        y_grupo = y[mascara]
        prob_grupo = probabilidad[mascara]
        for umbral in umbrales:
            fila = metricas_grupo(
                y_grupo,
                prob_grupo >= umbral,
                nombre,
                ganancia_G,
                perdida_L,
            )
            fila["umbral"] = float(umbral)
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_pares_umbrales(
    tabla_umbrales: pd.DataFrame,
    metricas_definicion: list[str],
    tolerancia: float = 0.01,
) -> pd.DataFrame:
    """Combina los umbrales de ambos grupos y evalúa la restricción de equidad.

    La tabla resultante tiene una fila por cada par ``(t_H, t_M)`` de la
    cuadrícula, con la ganancia total y la diferencia de cada métrica. La
    columna ``cumple`` indica si todas las diferencias de la definición están
    dentro de ``tolerancia``.
    """
    hombres = tabla_umbrales[
        tabla_umbrales["grupo"] == GRUPO_HOMBRES
    ].reset_index(drop=True)
    mujeres = tabla_umbrales[
        tabla_umbrales["grupo"] == GRUPO_MUJERES
    ].reset_index(drop=True)
    if not np.array_equal(hombres["umbral"].to_numpy(), mujeres["umbral"].to_numpy()):
        raise ValueError("Los umbrales de ambos grupos no están alineados.")

    umbrales = hombres["umbral"].to_numpy()
    ganancia = (
        hombres["ganancia"].to_numpy()[:, None]
        + mujeres["ganancia"].to_numpy()[None, :]
    )

    datos: dict[str, np.ndarray] = {
        "t_H": np.repeat(umbrales, len(umbrales)),
        "t_M": np.tile(umbrales, len(umbrales)),
        "ganancia_total": ganancia.ravel(),
    }

    max_abs_diff = np.zeros_like(ganancia)
    for metrica in metricas_definicion:
        diferencia = (
            hombres[metrica].to_numpy()[:, None]
            - mujeres[metrica].to_numpy()[None, :]
        )
        datos[f"dif_{metrica}"] = diferencia.ravel()
        max_abs_diff = np.maximum(max_abs_diff, np.abs(diferencia))

    datos["max_abs_diff"] = max_abs_diff.ravel()
    tabla = pd.DataFrame(datos)
    tabla["cumple"] = tabla["max_abs_diff"] <= tolerancia
    return tabla


def seleccionar_par(
    tabla_pares: pd.DataFrame,
) -> pd.Series | None:
    """Elige el par factible de máxima ganancia y, en empate, el más equitativo.

    El orden es: mayor ganancia total, menor diferencia máxima, menor ``t_H`` y
    menor ``t_M``. Así la elección es determinista y no depende del orden de la
    cuadrícula.
    """
    factibles = tabla_pares[tabla_pares["cumple"]].copy()
    if factibles.empty:
        return None
    factibles = factibles.sort_values(
        ["ganancia_total", "max_abs_diff", "t_H", "t_M"],
        ascending=[False, True, True, True],
    )
    return factibles.iloc[0]


def bootstrap_diferencias(
    y: np.ndarray,
    rechazo: np.ndarray,
    sexo: np.ndarray,
    metricas: tuple[str, ...] = ("FPR", "FNR"),
    n_boot: int = 2000,
    seed: int = SEED,
    ganancia_G: float = G,
    perdida_L: float = L,
) -> tuple[dict[str, np.ndarray], int]:
    """Bootstrap de las diferencias ``métrica_H - métrica_M``.

    Remuestrea todas las filas con reemplazo en cada réplica. Devuelve un
    diccionario con un arreglo por métrica y el número de réplicas inválidas
    (aquellas donde alguna métrica no quedó definida).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    arreglos = {metrica: np.empty(n_boot, dtype=float) for metrica in metricas}
    invalidas = 0

    for b in range(n_boot):
        indices = rng.integers(0, n, size=n)
        y_b = y[indices]
        sexo_b = sexo[indices]
        rechazo_b = rechazo[indices]
        metricas_h = metricas_grupo(
            y_b[sexo_b == 1],
            rechazo_b[sexo_b == 1],
            GRUPO_HOMBRES,
            ganancia_G,
            perdida_L,
        )
        metricas_m = metricas_grupo(
            y_b[sexo_b == 2],
            rechazo_b[sexo_b == 2],
            GRUPO_MUJERES,
            ganancia_G,
            perdida_L,
        )
        invalida = False
        for metrica in metricas:
            valor_h = metricas_h[metrica]
            valor_m = metricas_m[metrica]
            if np.isfinite(valor_h) and np.isfinite(valor_m):
                arreglos[metrica][b] = float(valor_h - valor_m)
            else:
                arreglos[metrica][b] = np.nan
                invalida = True
        if invalida:
            invalidas += 1

    return arreglos, invalidas


def resumen_bootstrap(
    nombre: str,
    observada: float,
    replicas: np.ndarray,
    n_boot: int,
) -> dict[str, float | int | str]:
    """Resume una diferencia bootstrap con su IC del 95% por percentiles."""
    validas = replicas[np.isfinite(replicas)]
    if len(validas) == 0:
        return {
            "diferencia": nombre,
            "observada": float(observada),
            "media_bootstrap": np.nan,
            "sd_bootstrap": np.nan,
            "ic95_inferior": np.nan,
            "ic95_superior": np.nan,
            "contiene_cero": "no evaluable",
            "replicas_validas": 0,
            "replicas_invalidas": int(n_boot),
        }
    inferior, superior = np.percentile(validas, [2.5, 97.5])
    return {
        "diferencia": nombre,
        "observada": float(observada),
        "media_bootstrap": float(validas.mean()),
        "sd_bootstrap": float(validas.std(ddof=1)),
        "ic95_inferior": float(inferior),
        "ic95_superior": float(superior),
        "contiene_cero": "sí" if inferior <= 0.0 <= superior else "no",
        "replicas_validas": int(len(validas)),
        "replicas_invalidas": int(n_boot - len(validas)),
    }


def diferencia_observada(
    tabla: pd.DataFrame, metrica: str
) -> float:
    """Devuelve ``métrica_H - métrica_M`` de una tabla por grupo."""
    hombres = tabla.loc[tabla["grupo"] == GRUPO_HOMBRES, metrica].iloc[0]
    mujeres = tabla.loc[tabla["grupo"] == GRUPO_MUJERES, metrica].iloc[0]
    return float(hombres - mujeres)
