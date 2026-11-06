"""Funciones de reporte para la exploración del Problema 1.

Este módulo no limpia ni transforma los datos: solo imprime conteos y tasas.
Así la libreta se mantiene enfocada en la exploración y las funciones
reutilizables viven en un archivo .py normal.
"""

from __future__ import annotations

import pandas as pd

# Variables categóricas/ordinales que pide revisar el enunciado.
VARIABLES_BASE = ["SEX", "EDUCATION", "MARRIAGE"]

# El archivo original no trae PAY_1; solo se reportan las columnas existentes.
PAY_SOLICITADAS = [f"PAY_{i}" for i in range(7)]

# Códigos que no aparecen en la descripción oficial de la página de UCI 350.
CODIGOS_NO_DOCUMENTADOS_BASE = {
    "EDUCATION": [0, 5, 6],
    "MARRIAGE": [0],
}

ETIQUETA_SEXO = {1: "hombre", 2: "mujer"}


def porcentaje(conteo: int, total: int) -> float:
    """Calcula un porcentaje seguro aunque el total sea cero."""
    return 100.0 * conteo / total if total else 0.0


def columnas_pay(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas PAY_* solicitadas que realmente existen."""
    return [columna for columna in PAY_SOLICITADAS if columna in df.columns]


def codigos_no_documentados(df: pd.DataFrame) -> dict[str, list[int]]:
    """Construye el mapa de códigos raros usando las columnas PAY_* presentes."""
    codigos = dict(CODIGOS_NO_DOCUMENTADOS_BASE)
    for columna in columnas_pay(df):
        codigos[columna] = [-2, 0]
    return codigos


def mostrar_valores_y_conteos(df: pd.DataFrame) -> None:
    """Imprime valores únicos y conteos de las variables solicitadas."""
    print("\n" + "=" * 72)
    print("1. VALORES ÚNICOS Y CONTEOS")
    print("=" * 72)

    columnas = [*VARIABLES_BASE, *columnas_pay(df)]
    total = len(df)
    for columna in columnas:
        conteos = (
            df[columna]
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis(columna)
            .reset_index(name="conteo")
        )
        conteos["porcentaje"] = conteos["conteo"].map(
            lambda n: porcentaje(int(n), total)
        )
        print(f"\n{columna}  (total de filas: {total})")
        print(
            conteos.to_string(
                index=False,
                formatters={"porcentaje": lambda v: f"{v:.2f}%"},
            )
        )


def mostrar_codigos_no_documentados(df: pd.DataFrame) -> None:
    """Reporta los códigos raros y si están presentes en los datos."""
    print("\n" + "=" * 72)
    print("2. CÓDIGOS QUE NO APARECEN EN LA DESCRIPCIÓN OFICIAL DE UCI")
    print("=" * 72)
    print(
        "Solo se reportan conteos; no se imputan ni se recodifican valores.\n"
    )

    total = len(df)
    filas = []
    for columna, codigos in codigos_no_documentados(df).items():
        for codigo in codigos:
            conteo = int((df[columna] == codigo).sum())
            filas.append(
                {
                    "variable": columna,
                    "codigo": codigo,
                    "conteo": conteo,
                    "porcentaje": porcentaje(conteo, total),
                    "aparece_en_datos": "sí" if conteo > 0 else "no",
                }
            )

    tabla = pd.DataFrame(filas)
    print(
        tabla.to_string(
            index=False,
            formatters={"porcentaje": lambda v: f"{v:.2f}%"},
        )
    )


def mostrar_tasa_incumplimiento_por_sexo(
    df: pd.DataFrame, objetivo: str
) -> None:
    """Calcula la tasa de incumplimiento para hombres y mujeres."""
    print("\n" + "=" * 72)
    print("3. TASA DE INCUMPLIMIENTO POR SEXO")
    print("=" * 72)
    print("SEX: 1 = hombre, 2 = mujer")
    print(f"Variable objetivo: '{objetivo}' (1 = incumplimiento, 0 = no)\n")

    valores_objetivo = set(df[objetivo].dropna().unique())
    if not valores_objetivo.issubset({0, 1}):
        print(
            "ADVERTENCIA: la variable objetivo contiene valores distintos "
            f"de 0/1: {sorted(valores_objetivo)}"
        )

    resumen = (
        df.groupby("SEX", dropna=False)[objetivo]
        .agg(filas="count", incumplimientos="sum", tasa="mean")
        .reset_index()
    )
    resumen["sexo"] = resumen["SEX"].map(ETIQUETA_SEXO).fillna("desconocido")
    resumen["tasa_incumplimiento"] = resumen["tasa"].map(
        lambda v: f"{100.0 * v:.2f}%"
    )

    print(
        resumen[
            ["SEX", "sexo", "filas", "incumplimientos", "tasa_incumplimiento"]
        ].to_string(index=False)
    )
