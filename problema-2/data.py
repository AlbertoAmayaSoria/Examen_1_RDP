"""Descarga y lectura del dataset UCI Bike Sharing (ID 275).

Este módulo se encarga únicamente de:
1. descargar el ZIP oficial desde UCI si no existe localmente,
2. extraer el ``hour.csv`` en ``data/`` (directorio ignorado por git),
3. leerlo con pandas,
4. dividir en prueba (2012-10-01 a 2012-12-31, que NO se toca hasta la
   libreta final) y, del resto, una muestra aleatoria de 3000 horas con
   semilla ``SEED``, ordenada por (dteday, hr).

Las columnas ``casual`` y ``cnt`` se excluyen de la lista de entradas porque
``registered`` es el objetivo y ``casual + registered = cnt`` (usar ``casual``
sería fuga de información). No se imputa, elimina ni recodifica nada más.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# URL oficial del dataset en el repositorio UCI.
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

# La raíz del repositorio es el directorio padre de problema-2/.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "bike+sharing+dataset.zip"
HORA_CSV = DATA_DIR / "hour.csv"
README_TXT = DATA_DIR / "Readme.txt"

TARGET = "registered"
# Columnas que nunca serán entradas: el objetivo y su suma con casual.
EXCLUIR_ENTRADAS = {"casual", "cnt", "instant", "dteday"}

# Prueba: 2012-10-01 a 2012-12-31 (no se toca hasta la libreta final).
INICIO_PRUEBA = "2012-10-01"
FIN_PRUEBA = "2012-12-31"
# Tamaño de la muestra de entrenamiento aleatoria.
N_MUESTRA = 3000

# UCI puede rechazar peticiones sin User-Agent; usamos uno explícito.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def descargar_dataset(force: bool = False) -> Path:
    """Descarga y extrae el ``hour.csv`` y el ``Readme.txt`` si hace falta.

    La descarga se hace a un archivo temporal y luego se renombra, para no
    dejar un ZIP incompleto si la conexión falla. Devuelve la ruta del CSV.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not HORA_CSV.exists():
        if force or not ZIP_PATH.exists():
            tmp_zip = ZIP_PATH.with_suffix(".zip.part")
            peticion = Request(URL, headers=_HEADERS)
            try:
                with urlopen(peticion, timeout=120) as respuesta:
                    with tmp_zip.open("wb") as destino:
                        shutil.copyfileobj(respuesta, destino)
                tmp_zip.replace(ZIP_PATH)
            except Exception:
                tmp_zip.unlink(missing_ok=True)
                raise
        with zipfile.ZipFile(ZIP_PATH) as zf:
            nombres = zf.namelist()
            for buscado, destino in (("hour.csv", HORA_CSV), ("Readme.txt", README_TXT)):
                candidatos = [n for n in nombres if n.lower().endswith(buscado)]
                if candidatos:
                    destino.write_bytes(zf.read(candidatos[0]))
    return HORA_CSV


def cargar_dataframe(force_download: bool = False) -> pd.DataFrame:
    """Lee el ``hour.csv`` original sin modificar valores."""
    ruta = descargar_dataset(force=force_download)
    df = pd.read_csv(ruta, parse_dates=["dteday"])
    return df


def dividir_muestra_prueba(
    df: pd.DataFrame, seed: int, n_muestra: int = N_MUESTRA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (muestra de 3000, prueba) con el corte temporal fijado.

    La prueba es 2012-10-01 a 2012-12-31. Del resto se eligen ``n_muestra``
    filas al azar con ``seed`` y se ordenan por (dteday, hr): así la ventana
    expansiva de validación siempre entrena con horas anteriores a las de
    validación.
    """
    en_prueba = (df["dteday"] >= INICIO_PRUEBA) & (df["dteday"] <= FIN_PRUEBA)
    resto = df.loc[~en_prueba].reset_index(drop=True)
    prueba = df.loc[en_prueba].sort_values(["dteday", "hr"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    elegidas = rng.choice(len(resto), size=n_muestra, replace=False)
    muestra = (
        resto.iloc[elegidas]
        .sort_values(["dteday", "hr"], kind="stable")
        .reset_index(drop=True)
    )
    return muestra, prueba


def columnas_entrada(df: pd.DataFrame) -> list[str]:
    """Columnas candidatas a entrar, sin casual ni cnt ni identificadores."""
    return [c for c in df.columns if c not in EXCLUIR_ENTRADAS and c != TARGET]


if __name__ == "__main__":
    df = cargar_dataframe()
    print(f"Archivo: {HORA_CSV}")
    print(f"Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
    print(f"Columnas: {list(df.columns)}")
    print(df.head(3).to_string())


def particiones_expansivas(
    muestra: pd.DataFrame, n_bloques: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Particiones de ventana expansiva sobre la muestra ordenada por tiempo.

    Divide la muestra ordenada en ``n_bloques`` bloques consecutivos de igual
    tamaño y devuelve 4 particiones: entrena bloques [0..1]→valida [2],
    [0..2]→[3], [0..3]→[4], [0..4]→[5]. En todas, el entrenamiento usa solo
    horas anteriores a las de validación.
    """
    assert len(muestra) % n_bloques == 0
    bloque = len(muestra) // n_bloques
    particiones = []
    for i in range(n_bloques - 2):
        fin_train = (i + 2) * bloque
        particiones.append(
            (np.arange(0, fin_train), np.arange(fin_train, fin_train + bloque))
        )
    return particiones
