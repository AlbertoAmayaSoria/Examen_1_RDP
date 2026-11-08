"""Evidencia para las decisiones de datos del Problema 1.

Este módulo no elige variantes ni limpia el dataset: solo calcula tablas
descriptivas, incluidas las de códigos no documentados, y métricas de
validación cruzada para que el alumno decida.

Todas las transformaciones de columnas viven dentro de un ``Pipeline`` de
scikit-learn. La única excepción es la variante B de códigos raros, donde se
eliminan filas del conjunto de entrenamiento de cada partición (nunca del
conjunto de validación), porque un ``Pipeline`` no puede cambiar el número de
filas de ``y``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ``common.py`` está en la raíz del repositorio y ``modelos.py`` en esta
# carpeta. Agregamos ambas rutas para que el módulo funcione igual desde un
# script, una libreta o nbconvert.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DIR_PROBLEMA = Path(__file__).resolve().parent
for _ruta in (_REPO_ROOT, _DIR_PROBLEMA):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

from common import SEED  # noqa: E402  (import después de ajustar sys.path)
from modelos import (  # noqa: E402
    CODIGOS_RAROS,
    ETIQUETAS_MODELO,
    ETIQUETAS_PAY,
    ETIQUETAS_RAROS,
    construir_pipeline,
    obtener_columnas_pay,
)

# Códigos de PAY_0 que el enunciado marca como no documentados.
CODIGOS_RAROS_PAY_0 = [-2, 0]

# Valor crítico de la normal estándar para un intervalo de confianza del 95%.
Z_95 = 1.959963984540054


def dividir_estratificado(
    df: pd.DataFrame, objetivo: str, test_size: float = 0.30, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve índices posicionales de entrenamiento y prueba (70/30)."""
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        stratify=df[objetivo].to_numpy(),
        random_state=seed,
    )
    return np.sort(idx_train), np.sort(idx_test)


def guardar_indices(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    objetivo: str,
    ruta: Path,
) -> pd.DataFrame:
    """Guarda en CSV el índice posicional, su conjunto y la clase objetivo."""
    conjunto = np.full(len(df), "test", dtype=object)
    conjunto[idx_train] = "train"
    tabla = pd.DataFrame(
        {
            "indice": np.arange(len(df)),
            "conjunto": conjunto,
            "objetivo": df[objetivo].to_numpy(),
        }
    )
    tabla.to_csv(ruta, index=False)
    return tabla


def cargar_indices(
    ruta: Path, df: pd.DataFrame, objetivo: str
) -> tuple[np.ndarray, np.ndarray]:
    """Carga y valida los índices del split 70/30 guardado en CSV.

    Verifica que el archivo cubra todas las filas, que no haya traslape entre
    entrenamiento y prueba y que la variable objetivo coincida con el
    DataFrame original. Así la libreta de modelos no puede usar por accidente
    un split distinto al de la evidencia.
    """
    tabla = pd.read_csv(ruta)
    columnas_requeridas = {"indice", "conjunto", "objetivo"}
    if not columnas_requeridas.issubset(tabla.columns):
        raise ValueError(
            f"{ruta} debe contener las columnas {sorted(columnas_requeridas)}; "
            f"encontradas: {list(tabla.columns)}"
        )
    if len(tabla) != len(df):
        raise ValueError(
            f"{ruta} tiene {len(tabla)} filas y el DataFrame tiene {len(df)}."
        )
    if not np.array_equal(tabla["indice"].to_numpy(), np.arange(len(df))):
        raise ValueError(f"La columna 'indice' de {ruta} no es 0..n-1.")
    if set(tabla["conjunto"].unique()) - {"train", "test"}:
        raise ValueError(
            f"La columna 'conjunto' de {ruta} solo puede tener 'train' o "
            f"'test'; encontrados: {sorted(tabla['conjunto'].unique())}"
        )
    if not np.array_equal(
        tabla["objetivo"].to_numpy(), df[objetivo].to_numpy()
    ):
        raise ValueError(
            f"La columna 'objetivo' de {ruta} no coincide con '{objetivo}' "
            "del DataFrame."
        )

    idx_train = np.sort(
        tabla.loc[tabla["conjunto"] == "train", "indice"].to_numpy()
    )
    idx_test = np.sort(
        tabla.loc[tabla["conjunto"] == "test", "indice"].to_numpy()
    )
    if len(idx_train) + len(idx_test) != len(df):
        raise ValueError(
            f"El split de {ruta} no cubre todas las filas: "
            f"{len(idx_train)} train + {len(idx_test)} test != {len(df)}."
        )
    if np.intersect1d(idx_train, idx_test).size:
        raise ValueError(f"El split de {ruta} tiene índices repetidos.")
    return idx_train, idx_test


def _resumen_grupo(
    grupo: pd.DataFrame, objetivo: str
) -> dict[str, float | int]:
    """Calcula n, tasa de incumplimiento y composición por SEX de un grupo."""
    n = len(grupo)
    n_hombres = int((grupo["SEX"] == 1).sum())
    n_mujeres = int((grupo["SEX"] == 2).sum())
    return {
        "n": n,
        "n_hombres": n_hombres,
        "n_mujeres": n_mujeres,
        "pct_hombres": 100.0 * n_hombres / n if n else 0.0,
        "pct_mujeres": 100.0 * n_mujeres / n if n else 0.0,
        "tasa_incumplimiento": 100.0 * grupo[objetivo].mean() if n else 0.0,
    }


def tabla_descriptiva(
    df: pd.DataFrame,
    columna: str,
    codigos_no_documentados: list[int],
    objetivo: str,
) -> pd.DataFrame:
    """Tabla por cada valor observado de una variable categórica/ordinal."""
    datos = df[df[columna].notna()]
    filas = []
    for codigo in sorted(datos[columna].unique()):
        grupo = datos[datos[columna] == codigo]
        fila = {
            "variable": columna,
            "codigo": int(codigo),
            "en_descripcion_oficial": (
                "no" if int(codigo) in codigos_no_documentados else "sí"
            ),
        }
        fila.update(_resumen_grupo(grupo, objetivo))
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_comparacion(
    df: pd.DataFrame,
    columna: str,
    codigos_no_documentados: list[int],
    objetivo: str,
) -> pd.DataFrame:
    """Compara el agregado documentado contra el no documentado."""
    es_no_documentado = df[columna].isin(codigos_no_documentados)
    filas = []
    for valor, etiqueta in [
        (False, "documentado"),
        (True, "no documentado"),
    ]:
        grupo = df[es_no_documentado == valor]
        fila = {"variable": columna, "grupo": etiqueta}
        fila.update(_resumen_grupo(grupo, objetivo))
        filas.append(fila)
    return pd.DataFrame(filas)


def intervalo_wilson(
    conteo: int, n: int, z: float = Z_95
) -> tuple[float, float]:
    """Intervalo de confianza de Wilson al 95%, devuelto en porcentaje.

    Se usa Wilson porque se comporta mejor que la aproximación normal cuando
    ``n`` es pequeño o la proporción está cerca de 0 o de 1, que es justo el
    caso de varios códigos raros.
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = conteo / n
    denominador = 1.0 + z * z / n
    centro = (p + z * z / (2.0 * n)) / denominador
    margen = (
        z
        * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
        / denominador
    )
    return (
        100.0 * max(0.0, centro - margen),
        100.0 * min(1.0, centro + margen),
    )


def tabla_pay_significado(
    df: pd.DataFrame,
    columna_pay: str,
    columna_bill: str,
    columna_pay_amt: str,
    objetivo: str,
    codigos: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> pd.DataFrame:
    """Describe cada código de PAY_* sin recodificarlo.

    Para cada código se reporta el tamaño de muestra, la tasa de
    incumplimiento, la proporción de facturas no positivas, las medianas de
    factura y pago, y la mediana de la razón pago/factura. La razón solo se
    calcula cuando la factura es estrictamente positiva, porque dividir entre
    cero o entre saldos negativos no tiene una lectura directa.
    """
    filas = []
    for codigo in codigos:
        grupo = df[df[columna_pay] == codigo]
        n = len(grupo)
        bill = grupo[columna_bill]
        pay_amt = grupo[columna_pay_amt]
        positivos = bill > 0
        ratio = pay_amt[positivos] / bill[positivos]
        filas.append(
            {
                "variable_pay": columna_pay,
                "codigo": codigo,
                "n": n,
                "tasa_incumplimiento": (
                    100.0 * float(grupo[objetivo].mean()) if n else np.nan
                ),
                "pct_bill_no_positivo": (
                    100.0 * float((bill <= 0).mean()) if n else np.nan
                ),
                "mediana_bill": float(bill.median()) if n else np.nan,
                "mediana_pay_amt": float(pay_amt.median()) if n else np.nan,
                "n_bill_positivo": int(positivos.sum()),
                "mediana_ratio_pay_amt_bill": (
                    float(ratio.median()) if positivos.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(filas)


def tabla_categoria_significado(
    df: pd.DataFrame,
    columna: str,
    codigos: list[int],
    objetivo: str,
) -> pd.DataFrame:
    """Describe EDUCATION o MARRIAGE con IC de Wilson y variables de contexto.

    No recodifica ni agrupa códigos: cada fila corresponde a un código tal
    como aparece en el entrenamiento.
    """
    filas = []
    for codigo in codigos:
        grupo = df[df[columna] == codigo]
        n = len(grupo)
        incumplimientos = int(grupo[objetivo].sum()) if n else 0
        if n:
            tasa = 100.0 * incumplimientos / n
            ic_inferior, ic_superior = intervalo_wilson(incumplimientos, n)
            edad_media = float(grupo["AGE"].mean())
            limit_bal_mediana = float(grupo["LIMIT_BAL"].median())
            n_mujeres = int((grupo["SEX"] == 2).sum())
            pct_mujeres = 100.0 * n_mujeres / n
        else:
            tasa = ic_inferior = ic_superior = np.nan
            edad_media = limit_bal_mediana = np.nan
            n_mujeres = 0
            pct_mujeres = np.nan
        filas.append(
            {
                "variable": columna,
                "codigo": codigo,
                "n": n,
                "incumplimientos": incumplimientos,
                "tasa_incumplimiento": tasa,
                "ic95_inferior": ic_inferior,
                "ic95_superior": ic_superior,
                "edad_media": edad_media,
                "limit_bal_mediana": limit_bal_mediana,
                "n_mujeres": n_mujeres,
                "pct_mujeres": pct_mujeres,
            }
        )
    return pd.DataFrame(filas)


def _mascara_codigos_raros(X: pd.DataFrame) -> np.ndarray:
    """Marca filas con algún código raro en EDUCATION o MARRIAGE."""
    return (
        X["EDUCATION"].isin(CODIGOS_RAROS["EDUCATION"])
        | X["MARRIAGE"].isin(CODIGOS_RAROS["MARRIAGE"])
    ).to_numpy()


def evaluar_combinacion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    variante_raros: str,
    variante_pay: str,
    modelo: str,
    columnas_pay: list[str],
    incluir_sexo: bool = True,
    seed: int = SEED,
) -> dict[str, float]:
    """Evalúa una combinación con las mismas particiones para todas las filas.

    En la variante B de códigos raros se filtran filas solo del entrenamiento
    de cada fold; la validación conserva todas las filas del fold para que los
    AUC sean comparables con las demás variantes.
    """
    mascara_raros = _mascara_codigos_raros(X_train)
    aucs: list[float] = []
    tamanos_entrenamiento: list[int] = []

    for idx_train_fold, idx_validacion in folds:
        idx_train_fold = np.asarray(idx_train_fold)

        if variante_raros == "B":
            idx_train_fold = idx_train_fold[~mascara_raros[idx_train_fold]]

        pipeline = construir_pipeline(
            variante_raros=variante_raros,
            variante_pay=variante_pay,
            modelo=modelo,
            columnas_pay=columnas_pay,
            incluir_sexo=incluir_sexo,
            seed=seed,
        )
        pipeline.fit(
            X_train.iloc[idx_train_fold],
            y_train.iloc[idx_train_fold],
        )
        probabilidades = pipeline.predict_proba(
            X_train.iloc[idx_validacion]
        )[:, 1]
        aucs.append(
            roc_auc_score(y_train.iloc[idx_validacion], probabilidades)
        )
        tamanos_entrenamiento.append(len(idx_train_fold))

    return {
        "auc_medio": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_train_promedio": float(np.mean(tamanos_entrenamiento)),
    }


def crear_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Crea una sola vez las particiones estratificadas para todas las tablas."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X_train, y_train))


def evaluar_combinaciones(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columnas_pay: list[str],
    seed: int = SEED,
) -> pd.DataFrame:
    """Evalúa las 3 variantes de códigos raros x 2 de PAY_* x 2 modelos."""
    filas = []
    for variante_raros in ["A", "B", "C"]:
        for variante_pay in ["A", "B"]:
            for modelo in ["logistica", "hist_gb"]:
                metricas = evaluar_combinacion(
                    X_train,
                    y_train,
                    folds,
                    variante_raros,
                    variante_pay,
                    modelo,
                    columnas_pay,
                    incluir_sexo=True,
                    seed=seed,
                )
                filas.append(
                    {
                        "experimento": "combinaciones",
                        "variante_raros_id": variante_raros,
                        "variante_raros": ETIQUETAS_RAROS[variante_raros],
                        "variante_pay_id": variante_pay,
                        "variante_pay": ETIQUETAS_PAY[variante_pay],
                        "modelo_id": modelo,
                        "modelo": ETIQUETAS_MODELO[modelo],
                        "con_sexo": "sí",
                        **metricas,
                    }
                )
    return pd.DataFrame(filas)


def evaluar_sexo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    mejor: pd.Series,
    columnas_pay: list[str],
    seed: int = SEED,
) -> tuple[pd.DataFrame, float]:
    """Repite la mejor combinación con y sin SEX y calcula la diferencia."""
    filas = []
    for incluir_sexo in [True, False]:
        metricas = evaluar_combinacion(
            X_train,
            y_train,
            folds,
            mejor["variante_raros_id"],
            mejor["variante_pay_id"],
            mejor["modelo_id"],
            columnas_pay,
            incluir_sexo=incluir_sexo,
            seed=seed,
        )
        filas.append(
            {
                "experimento": "sexo",
                "variante_raros_id": mejor["variante_raros_id"],
                "variante_raros": mejor["variante_raros"],
                "variante_pay_id": mejor["variante_pay_id"],
                "variante_pay": mejor["variante_pay"],
                "modelo_id": mejor["modelo_id"],
                "modelo": mejor["modelo"],
                "con_sexo": "sí" if incluir_sexo else "no",
                **metricas,
            }
        )
    tabla = pd.DataFrame(filas)
    auc_con = float(tabla.loc[tabla["con_sexo"] == "sí", "auc_medio"].iloc[0])
    auc_sin = float(tabla.loc[tabla["con_sexo"] == "no", "auc_medio"].iloc[0])
    diferencia = auc_con - auc_sin
    tabla["diferencia_auc_con_menos_sin"] = diferencia
    return tabla, diferencia
