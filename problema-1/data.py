"""Descarga y lectura del dataset UCI 350 (Default of Credit Card Clients).

Este módulo se encarga únicamente de:
1. descargar el ZIP original desde UCI si no existe localmente,
2. extraer el archivo .xls en ``data/`` (directorio ignorado por git),
3. leerlo con pandas usando el motor ``xlrd``,
4. corregir el encabezado desplazado que trae el archivo original.

No se imputan, eliminan ni recodifican valores. En particular, los códigos
raros de EDUCATION, MARRIAGE y PAY_* se dejan intactos para que el análisis
posterior decida qué hacer con ellos.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

# URL oficial del dataset en el repositorio UCI.
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# La raíz del repositorio es el directorio padre de problema-1/.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "default+of+credit+card+clients.zip"
XLS_PATH = DATA_DIR / "default of credit card clients.xls"

# Nombre exacto de la columna objetivo en el archivo original.
TARGET = "default payment next month"

# UCI puede rechazar peticiones sin User-Agent; usamos uno explícito.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _normalizar_nombre(valor: object) -> str:
    """Normaliza un texto para comparar encabezados sin depender de espacios."""
    return " ".join(str(valor).split()).casefold()


def descargar_dataset(force: bool = False) -> Path:
    """Descarga y extrae el .xls si hace falta; devuelve la ruta del Excel.

    ``force=True`` vuelve a descargar y extraer aunque los archivos existan.
    La descarga se hace a un archivo temporal y luego se renombra, para no
    dejar un ZIP incompleto si la conexión falla.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if force or not XLS_PATH.exists():
        if force or not ZIP_PATH.exists():
            tmp_zip = ZIP_PATH.with_suffix(".zip.part")
            peticion = Request(URL, headers=_HEADERS)
            try:
                with urlopen(peticion, timeout=120) as respuesta:
                    with tmp_zip.open("wb") as destino:
                        shutil.copyfileobj(respuesta, destino)
                tmp_zip.replace(ZIP_PATH)
            except Exception:
                # Si falla la descarga, no dejamos archivos parciales.
                tmp_zip.unlink(missing_ok=True)
                raise

        # Extraemos únicamente el .xls y lo escribimos con el nombre esperado.
        with zipfile.ZipFile(ZIP_PATH) as zf:
            candidatos = [n for n in zf.namelist() if n.lower().endswith(".xls")]
            if not candidatos:
                raise FileNotFoundError(
                    f"El ZIP {ZIP_PATH} no contiene ningún archivo .xls"
                )
            XLS_PATH.write_bytes(zf.read(candidatos[0]))

    return XLS_PATH


def _detectar_fila_encabezado(vista: pd.DataFrame) -> int:
    """Encuentra la fila (0-based) que contiene la columna objetivo.

    El archivo de UCI trae una primera fila basura con etiquetas X1...Y y
    los nombres reales en la siguiente fila. En lugar de fijar ``header=1``
    a ciegas, buscamos la fila que contiene el nombre de la columna objetivo.
    """
    objetivo = _normalizar_nombre(TARGET)
    for indice, fila in vista.iterrows():
        nombres = {
            _normalizar_nombre(valor)
            for valor in fila.to_numpy()
            if pd.notna(valor)
        }
        if objetivo in nombres:
            return int(indice)
    raise ValueError(
        f"No se encontró la columna objetivo '{TARGET}' en las primeras filas "
        f"de {XLS_PATH.name}."
    )


def cargar_dataframe(force_download: bool = False) -> pd.DataFrame:
    """Devuelve el DataFrame con encabezados correctos y valores originales.

    Solo se corrige la estructura del encabezado y se limpian espacios en los
    nombres de columnas. Los valores de las variables no se modifican.
    """
    xls_path = descargar_dataset(force=force_download)

    # Leemos una vista corta solo para localizar el encabezado real.
    vista = pd.read_excel(xls_path, engine="xlrd", header=None, nrows=20)
    fila_encabezado = _detectar_fila_encabezado(vista)

    # Volvemos a leer el archivo completo usando la fila detectada como header.
    df = pd.read_excel(xls_path, engine="xlrd", header=fila_encabezado)

    # Quitamos espacios accidentales en los nombres; no tocamos los valores.
    df.columns = [str(columna).strip() for columna in df.columns]

    if TARGET not in df.columns:
        raise KeyError(
            f"El DataFrame no contiene la columna objetivo '{TARGET}'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    datos = cargar_dataframe()
    print(f"Archivo: {XLS_PATH}")
    print(f"Dimensiones: {datos.shape[0]} filas x {datos.shape[1]} columnas")
    print(f"Columna objetivo: {TARGET}")
    print("Primeras columnas:", list(datos.columns[:8]))
    print(datos.head(3).to_string(index=False))
