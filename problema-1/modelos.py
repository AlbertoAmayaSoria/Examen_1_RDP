"""Pipelines y búsqueda de hiperparámetros del Problema 1.

Este módulo centraliza el preprocesamiento que antes vivía en
``evidencia_datos.py``. Así la libreta de evidencia (``02_decisiones_datos``)
y la de modelos (``03_modelos``) usan exactamente las mismas transformaciones,
sin duplicar funciones.

Decisiones del alumno que se aplican aquí:

- Códigos raros agrupados: ``EDUCATION {0, 5, 6} -> 4`` y
  ``MARRIAGE {0} -> 3``.
- ``SEX`` no entra como variable de entrada; se conserva aparte para
  auditoría por grupo.
- ``HistGradientBoostingClassifier`` usa ``PAY_*`` numérico tal cual.
- La regresión logística con escalado usa la variante B de ``PAY_*``:
  indicadores para ``-2``, ``-1`` y ``0``, más meses de atraso para ``>= 1``.

Ninguna transformación se ajusta fuera del ``Pipeline``: el escalado, el
one-hot y la codificación de ``PAY_*`` se aprenden solo con el entrenamiento
de cada partición o del conjunto de entrenamiento final.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ``common.py`` está en la raíz del repositorio. Agregamos esa ruta para que
# el módulo funcione igual desde un script, una libreta o nbconvert.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import SEED  # noqa: E402  (import después de ajustar sys.path)

# Códigos que el enunciado marca como no documentados.
CODIGOS_RAROS = {
    "EDUCATION": [0, 5, 6],
    "MARRIAGE": [0],
}

# Columnas numéricas que no son PAY_*.
COLUMNAS_NUMERICAS_BASE = [
    "LIMIT_BAL",
    "AGE",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
]

ETIQUETAS_RAROS = {
    "A": "A: agrupar códigos raros",
    "B": "B: eliminar filas raras del entrenamiento",
    "C": "C: one-hot de categorías propias",
}
ETIQUETAS_PAY = {
    "A": "A: PAY_* numérico tal cual",
    "B": "B: estados (-2, -1, 0) + meses de atraso",
}
ETIQUETAS_MODELO = {
    "logistica": "Regresión logística con escalado",
    "hist_gb": "HistGradientBoostingClassifier",
}

# Decisiones del alumno, centralizadas para que las libretas no las repitan.
VARIANTE_RAROS = "A"
VARIANTE_PAY_POR_MODELO = {
    "logistica": "B",
    "hist_gb": "A",
}
INCLUIR_SEXO = False

# Cuadrículas pequeñas solicitadas en el examen. Las claves llevan el prefijo
# ``modelo__`` porque así se llama el paso del clasificador dentro del
# Pipeline.
ESPACIOS_BUSQUEDA = {
    "logistica": {
        "modelo__C": [0.01, 0.1, 1, 10],
    },
    "hist_gb": {
        "modelo__learning_rate": [0.03, 0.1],
        "modelo__max_depth": [3, 5, None],
        "modelo__max_leaf_nodes": [15, 31],
        "modelo__l2_regularization": [0, 1],
    },
}


class AgruparCodigosRaros(BaseEstimator, TransformerMixin):
    """Agrupa EDUCATION {0,5,6}->4 y MARRIAGE {0}->3 dentro del Pipeline."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "AgruparCodigosRaros":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.loc[X["EDUCATION"].isin(CODIGOS_RAROS["EDUCATION"]), "EDUCATION"] = 4
        X.loc[X["MARRIAGE"].isin(CODIGOS_RAROS["MARRIAGE"]), "MARRIAGE"] = 3
        return X


def columnas_pay_b(columnas_pay: list[str]) -> list[str]:
    """Nombres de las columnas derivadas de la variante B de PAY_*."""
    derivadas = []
    for columna in columnas_pay:
        derivadas.extend(
            [
                f"{columna}_estado_m2",
                f"{columna}_estado_m1",
                f"{columna}_estado_0",
                f"{columna}_meses_atraso",
            ]
        )
    return derivadas


class CodificarPayB(BaseEstimator, TransformerMixin):
    """Convierte cada PAY_* en indicadores de estado y meses de atraso.

    -2, -1 y 0 se representan con indicadores binarios.
    Los valores >= 1 se conservan como meses de atraso en una columna numérica.
    Las columnas PAY_* originales se eliminan para no duplicar información.
    """

    def __init__(self, columnas_pay: list[str]):
        self.columnas_pay = columnas_pay

    def fit(self, X: pd.DataFrame, y: object = None) -> "CodificarPayB":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for columna in self.columnas_pay:
            valores = X[columna]
            X[f"{columna}_estado_m2"] = (valores == -2).astype("int8")
            X[f"{columna}_estado_m1"] = (valores == -1).astype("int8")
            X[f"{columna}_estado_0"] = (valores == 0).astype("int8")
            X[f"{columna}_meses_atraso"] = valores.where(valores >= 1, 0)
        return X.drop(columns=self.columnas_pay)


def obtener_columnas_pay(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas PAY_* que existen realmente en el dataset."""
    return [f"PAY_{i}" for i in range(7) if f"PAY_{i}" in df.columns]


def columnas_entrada(
    df: pd.DataFrame,
    objetivo: str,
    incluir_sexo: bool = INCLUIR_SEXO,
) -> list[str]:
    """Columnas predictoras, sin ID, sin objetivo y, por decisión, sin SEX.

    ``SEX`` se conserva en el DataFrame original para auditoría, pero no se
    pasa al modelo principal. Por eso esta función la excluye de la lista de
    entrada cuando ``incluir_sexo=False``.
    """
    excluidas = {"ID", objetivo}
    if not incluir_sexo:
        excluidas.add("SEX")
    return [columna for columna in df.columns if columna not in excluidas]


def construir_pipeline(
    variante_raros: str,
    variante_pay: str,
    modelo: str,
    columnas_pay: list[str],
    incluir_sexo: bool = INCLUIR_SEXO,
    seed: int = SEED,
) -> Pipeline:
    """Construye un Pipeline nuevo con todo el preprocesamiento dentro."""
    if variante_pay == "A":
        columnas_numericas = [*COLUMNAS_NUMERICAS_BASE, *columnas_pay]
    elif variante_pay == "B":
        columnas_numericas = [
            *COLUMNAS_NUMERICAS_BASE,
            *columnas_pay_b(columnas_pay),
        ]
    else:
        raise ValueError(f"Variante de PAY_* desconocida: {variante_pay}")

    columnas_categoricas = ["EDUCATION", "MARRIAGE"]
    if incluir_sexo:
        columnas_categoricas = ["SEX", *columnas_categoricas]

    escalado = "passthrough"
    if modelo == "logistica":
        escalado = StandardScaler()

    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                columnas_categoricas,
            ),
            ("numericas", escalado, columnas_numericas),
        ],
        remainder="drop",
    )

    pasos: list[tuple[str, object]] = []
    if variante_raros == "A":
        pasos.append(("agrupar_codigos_raros", AgruparCodigosRaros()))
    if variante_pay == "B":
        pasos.append(("codificar_pay_b", CodificarPayB(columnas_pay)))
    pasos.append(("preprocesamiento", preprocesamiento))

    if modelo == "logistica":
        clasificador = LogisticRegression(max_iter=2000, random_state=seed)
    elif modelo == "hist_gb":
        clasificador = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Modelo desconocido: {modelo}")
    pasos.append(("modelo", clasificador))

    return Pipeline(pasos)


def construir_pipeline_desde_mejor(
    mejor: pd.Series,
    columnas_pay: list[str],
    seed: int = SEED,
    incluir_sexo: bool = False,
) -> Pipeline:
    """Reconstruye el Pipeline del mejor modelo guardado por ``03_modelos``.

    ``mejor`` es una fila de ``resultados/03_resumen_modelos.csv``. La columna
    ``hiperparametros`` guarda un diccionario en texto; aquí se convierte a un
    diccionario real y se le agrega el prefijo ``modelo__`` que exige el
    ``Pipeline``. Así la libreta de calibración no repite la lógica de
    reconstrucción ni corre el riesgo de usar otros hiperparámetros.

    ``incluir_sexo`` permite construir una variante de auditoría que sí usa
    ``SEX`` como entrada. El modelo principal del examen lo deja fuera; esta
    opción existe solo para el contraste del inciso (e).
    """
    parametros = ast.literal_eval(mejor["hiperparametros"])
    pipeline = construir_pipeline(
        variante_raros=str(mejor["variante_raros_id"]),
        variante_pay=str(mejor["variante_pay_id"]),
        modelo=str(mejor["modelo_id"]),
        columnas_pay=columnas_pay,
        incluir_sexo=incluir_sexo,
        seed=seed,
    )
    pipeline.set_params(
        **{f"modelo__{clave}": valor for clave, valor in parametros.items()}
    )
    return pipeline


def buscar_hiperparametros(
    pipeline: Pipeline,
    parametros: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_jobs: int = 1,
) -> GridSearchCV:
    """Busca hiperparámetros con AUC y las mismas particiones de validación.

    ``refit=True`` reentrena el mejor Pipeline con todo el 70% de
    entrenamiento. El AUC de prueba no se usa aquí: la búsqueda solo ve las
    particiones de validación cruzada del 70%.
    """
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring="roc_auc",
        cv=folds,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    busqueda.fit(X, y)
    return busqueda


def tabla_busqueda(busqueda: GridSearchCV, modelo: str) -> pd.DataFrame:
    """Convierte ``cv_results_`` en una tabla ordenada por ranking de AUC."""
    resultados = pd.DataFrame(busqueda.cv_results_)
    parametros = resultados["params"].apply(pd.Series)
    parametros.columns = [
        columna.replace("modelo__", "") for columna in parametros.columns
    ]
    # ``max_depth=None`` se convierte en NaN al construir el DataFrame porque
    # la columna queda en tipo flotante. Lo mostramos como el texto "None"
    # para que la tabla no parezca tener un dato faltante.
    if "max_depth" in parametros.columns:
        parametros["max_depth"] = parametros["max_depth"].map(
            lambda valor: "None" if pd.isna(valor) else int(valor)
        )
    tabla = pd.concat(
        [
            parametros,
            resultados[
                ["mean_test_score", "std_test_score", "rank_test_score"]
            ].rename(
                columns={
                    "mean_test_score": "auc_cv_medio",
                    "std_test_score": "auc_cv_std",
                    "rank_test_score": "ranking",
                }
            ),
        ],
        axis=1,
    )
    tabla.insert(0, "modelo", ETIQUETAS_MODELO[modelo])
    return tabla.sort_values("ranking").reset_index(drop=True)


def resumen_modelos(
    busquedas: dict[str, GridSearchCV],
    aucs_test: dict[str, float],
) -> pd.DataFrame:
    """Resume hiperparámetros elegidos, AUC de CV y AUC de prueba.

    El modelo marcado en ``mejor_por_auc_cv`` es el de mayor AUC medio de
    validación cruzada. El AUC de prueba se reporta una sola vez, pero no
    participa en esa elección.
    """
    filas = []
    for modelo, busqueda in busquedas.items():
        mejores = {
            clave.replace("modelo__", ""): valor
            for clave, valor in busqueda.best_params_.items()
        }
        filas.append(
            {
                "modelo_id": modelo,
                "modelo": ETIQUETAS_MODELO[modelo],
                "variante_raros_id": VARIANTE_RAROS,
                "variante_pay_id": VARIANTE_PAY_POR_MODELO[modelo],
                "incluye_sexo": "no",
                "hiperparametros": str(mejores),
                "auc_cv_medio": float(busqueda.best_score_),
                "auc_cv_std": float(
                    busqueda.cv_results_["std_test_score"][busqueda.best_index_]
                ),
                "auc_test": float(aucs_test[modelo]),
            }
        )

    tabla = pd.DataFrame(filas)
    indice_mejor = tabla["auc_cv_medio"].idxmax()
    tabla["mejor_por_auc_cv"] = "no"
    tabla.loc[indice_mejor, "mejor_por_auc_cv"] = "sí"
    return tabla


def guardar_probabilidades_test(
    df: pd.DataFrame,
    idx_test: np.ndarray,
    objetivo: str,
    probabilidades: dict[str, np.ndarray],
    ruta: Path,
) -> pd.DataFrame:
    """Guarda probabilidades de prueba, objetivo y SEX solo para auditoría."""
    tabla = pd.DataFrame(
        {
            "indice": idx_test,
            "objetivo": df.iloc[idx_test][objetivo].to_numpy(),
            "SEX": df.iloc[idx_test]["SEX"].to_numpy(),
        }
    )
    for modelo, valores in probabilidades.items():
        tabla[f"prob_{modelo}"] = valores
    tabla.to_csv(ruta, index=False, float_format="%.8f")
    return tabla
