"""Parámetros comunes del examen.

La semilla vive aquí y solo aquí. Cualquier módulo o libreta que use
aleatoriedad debe importar ``SEED`` desde este archivo, para evitar semillas
dispersas y garantizar que todo el repositorio sea reproducible.
"""

from __future__ import annotations

# Parámetros personales del examen (no modificar).
EXPEDIENTE = 223210350
A = 0
B = 5
SEED = 223210350

# Parámetros del Problema 1 ya confirmados con el enunciado.
# L = (3 + B mod 3) * G = (3 + 5 mod 3) * 1 = 5, es decir, L = 5G.
G = 1
L = (3 + B % 3) * G
UMBRAL = G / (G + L)  # 1/6
