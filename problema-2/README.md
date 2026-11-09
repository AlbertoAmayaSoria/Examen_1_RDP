# Problema 2

## Objetivo

Modelado de `registered` (bicicletas registradas, UCI Bike Sharing ID 275)
comparando kernels circulares `kI = exp(-d^2/2)` y `kII = max(0, 1-d^2/4)`
con kernels válidos de la tabla (von Mises, `exp(-d/ell)`, `max(0,1-d/2)`,
`pi^2/6 - pi*d/2 + d^2/4`), donde `d` es la distancia circular (arco más
corto, periodo 24 para `hr` y 7 para `weekday`).

## Estado

Implementado: descarga + split temporal, verificación de kernels, decisiones
con evidencia (ventana expansiva, combinación `(k_hr*k_wd)+k_rbf`, `log1p` +
RMSLE, von Mises, variables), modelos M1–M4 + HGB + kernels inválidos, y
evaluación final única en prueba. Las decisiones "Tú decides" fueron tomadas
por el asistente y marcadas como **PROPUESTA DEL ASISTENTE, pendiente de
revisión del usuario** en `bitacora_notas.md`.

## Ejecución

```bash
conda activate rp-examen
./problema-2/run_all.sh
```

1. `data.py`: descarga el ZIP de UCI (si falta) y lee `hour.csv`.
2. `01_kernels_verificacion.ipynb`: coeficientes de Fourier y valores
   propios de los 6 kernels.
3. `02_decisiones_p2.ipynb`: evidencia (gráficas hora×workingday, semana,
   errores por franja), particiones de ventana expansiva, comparación de
   combinaciones y decisiones.
4. `03_modelos_p2.ipynb`: M1–M4 + HGB con rejillas y ventana expansiva; M4
   con kernels inválidos (kI/kII) y su análisis.
5. `04_evaluacion_final.ipynb`: evaluación única en prueba (2012-10-01 a
   2012-12-31), tabla validación/prueba y gráfica semanal.

Funciones compartidas: `data.py` (descarga y split), `kernels.py` (solo
NumPy). Semilla en `common.py` (`SEED = 223210350`).

## Resultados generados (carpeta `resultados/`, con prefijo `p2_`)

- `p2_01_coeficientes.csv`, `p2_01_resumen_coeficientes.csv`,
  `p2_01_eigenvalores.csv`, `p2_01_fourier_*.png`.
- `p2_02_particiones.csv`, `p2_02_error_por_franja.csv`,
  `p2_02_comparacion_kernels.csv`, `p2_02_registrado_por_hr.png`,
  `p2_02_registrado_por_weekday.png`.
- `p2_03_resultados.csv`, `p2_03_mejor_modelo.csv`.
- `p2_04_evaluacion_final.csv`, `p2_04_semana_prueba.png`.

## Conclusiones

- `kI` y `kII` **no son kernels**: `a_4 ≈ -3.19e-4` (kI), `a_3 ≈ -0.0712`
  (kII); λ_min(24h) = −0.00505 (kI) y −0.839 (kII). Con 7 días, kI parece
  positivo semidefinido: un conjunto finito pequeño no always detecta la
  pérdida de semidefinición.
- Combinación ganadora: `(k_hr * k_wd) + k_rbf` (RMSLE 0.556 frente a 0.623
  y 0.720); validación de ventana expansiva con 4 particiones; objetivo
  `log1p(registered)` con RMSLE (el error absoluto crece con el nivel).
- Modelos (validación → prueba, RMSLE): M4 0.439 → 0.458; HGB 0.452 → 0.457;
  M2 0.597 → 0.505; M3 0.781 → 0.743; M1 1.004 → 0.909. El kernel circular
  válido sí valió la pena frente a M1, M2 y M3, y empata/supera en
  eficiencia a HGB.
- Kernels inválidos en M4: kI λ_min = −0.247 (RMSLE 0.552) y kII λ_min =
  −45.4 (RMSLE 25.45, todos los puntos como vectores de soporte), sin
  advertencias de libsvm: el dual pierde convexidad y garantía de óptimo
  global. M3 equivale a un producto de kernels von Mises.

## Bitácora de decisiones

`bitacora_notas.md` registra las 6 decisiones (validación, combinación,
métrica/objetivo, kernel válido, variables, estandarización/HGB), cada una
marcada como PROPUESTA DEL ASISTENTE.

## Reproducibilidad

- Semilla: `223210350`.
- Los datos se descargan desde UCI dentro del código y se guardan en `data/`
  (ignorado por git).
- Comentarios en español explicando el porqué de cada paso.
