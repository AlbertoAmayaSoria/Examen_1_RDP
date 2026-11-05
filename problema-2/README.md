# Problema 2

## Objetivo

Modelado de la variable objetivo **`registered`** comparando dos kernels:

- `kI = exp(-d^2 / 2)`
- `kII = max(0, 1 - d^2 / 4)`

donde `d` es la **distancia circular**.

## Estado

Pendiente de implementar.

## Puntos por confirmar con el enunciado

- [ ] Dataset UCI exacto y URL/ID de descarga.
- [ ] Unidad y periodo de la distancia circular `d`.
- [ ] Tratamiento del periodo (por ejemplo, 24 horas o 2π) y de los extremos.
- [ ] Variables predictoras y preprocesamiento esperado.
- [ ] Criterio de evaluación y gráficas solicitadas.

> Regla del examen: si alguna definición no está en el enunciado, preguntar
> antes de implementar. No inventar parámetros.

## Reproducibilidad

- Semilla: `223210350`.
- Los datos se descargan desde UCI dentro del código y se guardan en `data/`
  (ignorado por git).
- Comentarios en español explicando el porqué de cada paso.
