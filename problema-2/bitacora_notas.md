# Bitácora de decisiones — Problema 2

> Regla de trabajo: cada decisión del alumno se registra con **evidencia** y
> con su **razón textual**. Si la razón no está registrada, se marca
> **pendiente del usuario**. Las decisiones de este problema fueron tomadas
> por el asistente con evidencia propia, por la excepción autorizada; cada
> razón se marca como **PROPUESTA DEL ASISTENTE, pendiente de revisión del
> usuario**.

## Estado general

- **Problema:** 2 (bicicletas, UCI 275).
- **Objetivo:** `registered` ((a+b) mod 3 = 2).
- **Distancia circular:** `d` = arco más corto en radianes, periodo 24 para
  `hr` y 7 para `weekday`.
- **Decisiones registradas:** 6 (todas propuestas del asistente).
- **Semilla:** `SEED = 223210350` (de `common.py`, sin cambios).

---

## Decisión 1 — Esquema de validación (PROPUESTA DEL ASISTENTE)

> **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario.**

### Evidencia

- **Libreta:** `problema-2/02_decisiones_p2.ipynb`, sección 2.
- **Evidencia:** `resultados/p2_02_particiones.csv`.

### Opciones presentadas

- **A)** Split aleatorio 80/20 (ignora el orden temporal).
- **B)** Ventana expansiva con 4 particiones sobre la muestra ordenada por
  tiempo, entrenando siempre con horas anteriores a las de validación.
- **C)** Una sola partición temporal de validación (pocas datos de validación).

### Decisión (propuesta del asistente)

**(a) Qué decidí:** ventana expansiva con 4 particiones: bloques de 500
horas sobre la muestra ordenada por (dteday, hr); entrena 0:1000→valida
1000:1500, 0:1500→1500:2000, 0:2000→2000:2500, 0:2500→2500:3000.

**(b) Alternativa descartada:** split aleatorio (mezcla futuro en el
entrenamiento y no refleja el uso real) y un solo corte temporal (solo 500
horas de validación, alta varianza).

**(c) Por qué:** en datos de demanda temporal, validar con horas posteriores
a las de entrenamiento es el único esquema que respeta la dirección del
tiempo. Con 4 particiones se tienen 2000 horas de validación en 4 momentos
distintos y el tamaño de entrenamiento crece (1000, 1500, 2000, 2500),
reflejando cómo el modelo mejora con más datos.

**(d) Qué pierdo:** los bloques de 500 horas comparten bordes de día (la
primera partición separa horas del mismo día entre train y val) y las
particiones no son independientes (el entrenamiento se solapa); no se
evalúa extrapolación estacional completa porque la muestra termina el
2012-09-30.

---

## Decisión 2 — Combinación de kernels (PROPUESTA DEL ASISTENTE)

> **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario.**

### Evidencia

- **Libreta:** `problema-2/02_decisiones_p2.ipynb`, secciones 1 y 4.
- **Evidencia:** `resultados/p2_02_registrado_por_hr.png`,
  `p2_02_registrado_por_weekday.png`, `p2_02_comparacion_kernels.csv`
  (RMSLE medio: suma 0.720, producto 0.623, `(hr*wd)+rbf` 0.556).

### Opciones presentadas

- **(i)** `k = k_hr + k_wd + k_rbf` (aditivo).
- **(ii)** `k = k_hr * k_wd * k_rbf` (producto total).
- **(iii)** `k = (k_hr * k_wd) + k_rbf` (interacción hora×día + clima).

### Decisión (propuesta del asistente)

**(a) Qué decidí:** la opción **(iii)** `(k_hr * k_wd) + k_rbf`, con von
Mises (kappa con rejilla en 03) para hora y día y RBF para las demás
variables.

**(b) Alternativa descartada:** la suma pura (i) (fuerza la misma forma
horaria todos los días) y el producto total (ii) (obliga también al clima a
interactuar con hora y día).

**(c) Por qué:** las gráficas de registradas medias por hora separando
`workingday` muestran formas distintas (doble pico laboral vs. máximo al
mediodía en fin de semana): hay interacción hora × día. La suma ganaría si el
efecto de la hora fuera el mismo cada día. La validación lo confirma: (iii)
0.556 < (ii) 0.623 < (i) 0.720 de RMSLE medio.

**(d) Qué pierdo:** (iii) asume que el efecto del clima es aditivo sobre la
interacción hora-día (no varía su forma con la hora); un producto por
bandas horarias de clima podría capturar más, a costa de más parámetros.

---

## Decisión 3 — Objetivo logarítmico y métrica RMSLE (PROPUESTA DEL ASISTENTE)

> **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario.**

### Evidencia

- **Libreta:** `problema-2/02_decisiones_p2.ipynb`, sección 3.
- **Evidencia:** `resultados/p2_02_error_por_franja.csv` (RMSE crudo pasa de
  10.5 en madrugada a 58.2 en la tarde; RMSLE pasa de 0.87 a 0.43).

### Opciones presentadas

- **A)** Objetivo crudo `registered`, métrica RMSE/MAE global.
- **B)** Objetivo `log1p(registered)`, métrica RMSLE (error relativo).

### Decisión (propuesta del asistente)

**(a) Qué decidí:** trabajar con `log1p(registered)` y elegir por RMSLE
(RMSE y MAE se reportan en 04 como contexto).

**(b) Alternativa descartada:** el objetivo crudo con RMSE global.

**(c) Por qué:** errar 20 bicicletas a las 3:00 (nivel ~18) no cuesta lo
mismo que a las 18:00 (nivel ~188): el error absoluto crece con el nivel,
así que con RMSE una hora pico pesaría 5-6 veces más que una hora valle.
RMSLE estabiliza el error entre franjas (0.87/0.53/0.43/0.31 crudo frente a
0.44/0.46/0.42/0.28 con log1p) y en validación el modelo en log obtiene
RMSLE global 0.404 frente a 0.570 del modelo en crudo.

**(d) Qué pierdo:** RMSLE subestima relativamente los errores absolutos
grandes en horas pico; al retransformar con `expm1` hay sesgo de
retransformación (la media del log no es el log de la media) y los
predichos pueden quedar por debajo de la media real en los picos.

---

## Decisión 4 — Kernel circular válido elegido (PROPUESTA DEL ASISTENTE)

> **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario.**

### Evidencia

- **Libreta:** `problema-2/01_kernels_verificacion.ipynb`.
- **Evidencia:** `resultados/p2_01_coeficientes.csv` (los 4 válidos sin
  coeficientes negativos; mínimo ~1e-16) y `p2_01_eigenvalores.csv`
  (λ_min ≥ 0 en 24h, 7d y muestra de 3000).

### Opciones presentadas

- **Von Mises** `exp(kappa(cos d − 1))`.
- **Exponencial circular** `exp(−d/ell)`.
- **Triangular** `max(0, 1 − d/2)`.
- **Cuadrático** `π²/6 − πd/2 + d²/4`.

### Decisión (propuesta del asistente)

**(a) Qué decidí:** **von Mises** para `hr` y para `weekday` (kappas
independientes con rejilla {1, 3} en 03; ganadores kappa_hr=3, kappa_wd=1).

**(b) Alternativa descartada:** el resto de kernels válidos (decaen con
distancia absoluta y no admiten un parámetro de concentración como kappa;
el triangular y el cuadrático son menos flexibles para picos horarios).

**(c) Por qué:** von Mises es el kernel circular estándar (distribución
circular de referencia), su ancho lo controla kappa de forma interpretable
y es el que aparece en la justificación teórica (Taylor de la exponencial).
Además, M3 (seno/coseno con gaussiano) equivale a un producto de von Mises,
así que la comparación con la familia von Mises es la más limpia.

**(d) Qué pierdo:** kappa controla un solo ancho por dimensión; formas
asimétricas de la demanda intra-día (subida rápida, bajada lenta) no se
capturan con un único kappa.

---

## Decisión 5 — workingday, weathersit y yr (PROPUESTA DEL ASISTENTE)

> **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario.**

### Evidencia

- **Libreta:** `problema-2/02_decisiones_p2.ipynb`, sección 4.
- **Evidencia:** `workingday` vale 1 exactamente si `weekday` está en 1..5 y
  `holiday` es 0 (3000/3000 filas); correlaciones de `temp`/`atemp` y medias
  por nivel de `weathersit` (161.9, 141.6, 91.8, 22.0) y `yr` (117.5, 194.5).

### Opciones presentadas

- **A)** Meter `workingday`, `weathersit` y `yr` en la parte RBF tal cual.
- **B)** Sacar `workingday` (redundante), conservar `weathersit` y `yr`.

### Decisión (propuesta del asistente)

**(a) Qué decidí:** **sacar `workingday`** de la parte RBF (ya está
codificado en `weekday` + `holiday`, que sí se conservan); **conservar
`weathersit`** como ordinal numérico y **`yr`** como binaria en la parte RBF,
junto a season, mnth, temp, atemp, hum y windspeed.

**(b) Alternativa descartada:** incluir `workingday` como variable extra
(duplica la información de domingo/sábado/festivo que ya captura el kernel de
periodo 7 y la interacción hora × día).

**(c) Por qué:** `workingday` es función exacta de `weekday` y `holiday`;
`weathersit` discrimina el nivel de demanda en 4 escalones limpios y `yr`
captura el crecimiento interanual (117 a 195 bicicletas medias). Ninguna de
las dos es función de las variables circulares.

**(d) Qué pierdo:** `holiday` queda como binaria en RBF, así que el efecto
específico de festivos puntuales (Navidad) solo se captura parcialmente; un
efecto de festivo × hora exigiría interacción adicional.

---

## Decisión 6 — Estandarización de "otras" y entrada del HGB (PROPUESTA DEL ASISTENTE)

> **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario.**

### Evidencia

- **Libretas:** `problema-2/03_modelos_p2.py` (sección 1) y
  `04_evaluacion_final.py`.
- **Evidencia:** resultados de `p2_03_resultados.csv` y
  `p2_04_evaluacion_final.csv`.

### Opciones presentadas

- **A)** Estandarizar "otras" con estadísticas de cada partición de
  entrenamiento (protocolo estricto, impide precalcular una matriz única).
- **B)** Estandarizar una vez con la muestra completa de 3000 horas.
- **C)** Para el HGB de referencia: usar todas las columnas de entrada
  (incluida `workingday`) o solo las de los SVR.

### Decisión (propuesta del asistente)

**(a) Qué decidí:** la opción **B** para M1–M4 (una sola estandarización con
las 3000 horas; la matriz M4 de 3000×3000 se calcula una vez y se rebana) y
para el HGB la opción de **todas las columnas de entrada** (incluida
`workingday`), porque es la referencia sin restricción de kernels.

**(b) Alternativa descartada:** la opción A (estandarizar por partición)
because impide cumplir el requisito de precalcular la matriz única de M4;
y dar al HGB solo las variables de los SVR (limitaría la referencia).

**(c) Por qué:** la fuga por estandarizar covariables con toda la muestra es
mínima (solo traslación/escala de variables de clima y calendario, sin el
objetivo) y simplifica que M4 fuera precalculado como exige el enunciado.
El HGB, al no usar kernels, se beneficia de ver `workingday` explícito.

**(d) Qué pierdo:** un ligero optimismo en validación por la estandarización
global; el efecto es del orden del error de estimación de media y desviación
con 1000-2500 puntos, despreciable frente a las diferencias entre modelos
(RMSLE 0.44 vs 0.60-1.00).

---

## Nota sobre kernels no válidos (evidencia, no decisión)

- Los candidatos `kI` y `kII` no son kernels (coeficientes de Fourier
  negativos y valores propios negativos claros en la matriz de 3000×3000).
- M4 con `kI`/`kII` (03, sección 5): λ_min = −0.247 y −45.4 en la matriz de
  entrenamiento; libsvm no emite advertencias; RMSLE de validación 0.552
  (kI) y 25.45 (kII, con todos los puntos como vectores de soporte) frente a
  0.439 del M4 válido. El dual deja de ser convexo: se pierde el óptimo
  global y la garantía de convergencia de SMO.
