"""Kernels circulares definidos solo con NumPy.

La distancia circular ``d`` es el arco más corto entre dos posiciones de un
círculo de periodo ``P`` (24 para la hora, 7 para el día de semana), en
radianes: ``d = 2*pi*min(delta, P-delta)/P``. El rango de ``d`` es [0, pi].

Esta librería contiene:
- ``circ_dist``: distancia circular con broadcasting (arco más corto),
- los dos candidatos del examen: ``kI = exp(-d^2/2)`` y
  ``kII = max(0, 1 - d^2/4)``,
- los kernels válidos de la tabla: von Mises ``exp(kappa*(cos d - 1))``,
  ``exp(-d/ell)``, ``max(0, 1 - d/2)`` y ``pi^2/6 - pi*d/2 + d^2/4``,
- ``kernel_matrix(X, Y, k)``: matriz de kernels para conjuntos de distinto
  tamaño, lista para usar con SVR precalculado.

Nada de esto usa datos: son funciones puras de la distancia.
"""

from __future__ import annotations

import numpy as np

PERIODO_HR = 24
PERIODO_WEEKDAY = 7
PI = np.pi


def circ_dist(x: np.ndarray, y: np.ndarray, periodo: int) -> np.ndarray:
    """Distancia circular (arco más corto, en radianes) entre x e y.

    ``x`` e ``y`` pueden tener cualquier forma; el resultado hace broadcast:
    la entrada ``(m, 1)`` contra ``(1, n)`` da una matriz ``(m, n)``. La
    diferencia absoluta se reduce módulo ``periodo`` y se toma el mínimo
    entre ``delta`` y ``periodo - delta`` (borde incluido: el arco más corto).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    delta = np.abs(x - y) % float(periodo)
    delta = np.minimum(delta, float(periodo) - delta)
    return 2.0 * PI * delta / float(periodo)


def k_I(d: np.ndarray) -> np.ndarray:
    """Candidato I del examen: exp(-d^2/2) sobre la distancia circular."""
    d = np.asarray(d, dtype=float)
    return np.exp(-(d ** 2) / 2.0)


def k_II(d: np.ndarray) -> np.ndarray:
    """Candidato II del examen: max(0, 1 - d^2/4) sobre la distancia."""
    d = np.asarray(d, dtype=float)
    return np.maximum(0.0, 1.0 - (d ** 2) / 4.0)


def k_von_mises(d: np.ndarray, kappa: float) -> np.ndarray:
    """Kernel de von Mises: exp(kappa*(cos d - 1)), con kappa > 0."""
    d = np.asarray(d, dtype=float)
    if kappa <= 0:
        raise ValueError("kappa debe ser positivo.")
    return np.exp(kappa * (np.cos(d) - 1.0))


def k_exp_minus(d: np.ndarray, ell: float) -> np.ndarray:
    """Kernel exp(-d/ell), con ell > 0."""
    d = np.asarray(d, dtype=float)
    if ell <= 0:
        raise ValueError("ell debe ser positivo.")
    return np.exp(-d / ell)


def k_tri(d: np.ndarray) -> np.ndarray:
    """Kernel triangular: max(0, 1 - d/2)."""
    d = np.asarray(d, dtype=float)
    return np.maximum(0.0, 1.0 - d / 2.0)


def k_poly_circular(d: np.ndarray) -> np.ndarray:
    """Kernel pi^2/6 - pi*d/2 + d^2/4 (d en [0, pi])."""
    d = np.asarray(d, dtype=float)
    return PI ** 2 / 6.0 - PI * d / 2.0 + (d ** 2) / 4.0


def kernel_matrix(
    X: np.ndarray, Y: np.ndarray, kernel, periodo: int | None, **kwargs
) -> np.ndarray:
    """Matriz ``K[i, j] = kernel(d(X[i], Y[j]))`` para conjuntos de distinto tamaño.

    ``kernel`` recibe la matriz de distancias circulares. Si ``periodo`` es
    ``None``, ``kernel`` recibe directamente las diferencias ``X - Y`` (para
    kernels euclidianos como RBF que no usan distancia circular).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if periodo is not None:
        d = circ_dist(X[:, None], Y[None, :], periodo)
        return kernel(d, **kwargs) if kwargs else kernel(d)
    diferencia = X[:, None, :] - Y[None, :, :]
    return kernel(diferencia, **kwargs) if kwargs else kernel(diferencia)


def rbf(diferencia: np.ndarray, gamma: float) -> np.ndarray:
    """RBF sobre diferencias ya calculadas: exp(-gamma * ||x - y||^2)."""
    if gamma <= 0:
        raise ValueError("gamma debe ser positivo.")
    return np.exp(-gamma * np.sum(diferencia ** 2, axis=-1))
