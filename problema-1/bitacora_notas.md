# Bitácora de decisiones — Problema 1

> Regla de trabajo: cada decisión se registra con **evidencia** y con su
> **razón textual**. No se parafrasean, mejoran ni completan las razones del
> alumno.

## Estado general

- **Problema:** 1 (crédito, UCI 350).
- **Evidencia producida:** `problema-1/02_decisiones_datos.ipynb` y archivos
  de `resultados/`. También: `03_modelos.ipynb`, `04_calibracion.ipynb`,
  `05_tabla_grupos.ipynb`, `06_intervencion.ipynb` y `07_sin_atributo.ipynb`.
- **Decisiones registradas:** 3 del usuario + 3 redactadas por el asistente
  y aceptadas por el alumno.
- **Decisiones con razón textual registrada:** 3 (códigos raros, `PAY_*`,
  `SEX`). Las decisiones 4–6 fueron redactadas por el asistente y aceptadas
  por el alumno.
- **Decisiones de datos sin respuesta:** 0.
- **Cierre del problema:** decisiones de datos cerradas y definiciones del
  enunciado fijadas en `common.py`.

---

## Decisión 1 — Tratamiento de códigos raros

Códigos no documentados: `EDUCATION` 0, 5, 6 y `MARRIAGE` 0.

### Evidencia

- **Archivo/libreta:** `problema-1/02_decisiones_datos.ipynb`, secciones 1 y 3.
- **Tablas:**
  - Sección 1: `tabla_education`, `tabla_marriage`,
    `tabla_comparacion` para `EDUCATION` y `MARRIAGE`.
  - Sección 3: `tabla_significado_education`,
    `tabla_significado_marriage`.
  - Sección 4: `tabla_combinaciones`.
  - Archivos: `resultados/significado_education.csv`,
    `resultados/significado_marriage.csv`,
    `resultados/decisiones_datos.csv`.
- **Gráficas:** no se produjeron gráficas en esta evidencia.

**Cifras clave (dataset completo, sección 1):**

- `EDUCATION` no documentado: `n = 345`, tasa de incumplimiento `7.54%`;
  documentado: `n = 29,655`, `22.29%`.
- `MARRIAGE` 0: `n = 54`, `9.26%`; documentado: `n = 29,946`, `22.14%`.

**Cifras clave (solo 70% de entrenamiento, sección 3):**

| Variable | Código | n | Tasa default | IC 95% Wilson |
|---|---:|---:|---:|---:|
| EDUCATION | 0 | 10 | 0.00% | 0.00–27.75% |
| EDUCATION | 4 | 74 | 6.76% | 2.92–14.86% |
| EDUCATION | 5 | 194 | 5.15% | 2.82–9.23% |
| EDUCATION | 6 | 26 | 23.08% | 11.03–42.05% |
| MARRIAGE | 0 | 41 | 12.20% | 5.32–25.54% |
| MARRIAGE | 3 | 250 | 27.20% | 22.06–33.03% |

**AUC medio por variante de códigos raros** (mismas 5 particiones; solo 70%
de entrenamiento):

| PAY_* | Modelo | Raros A | Raros B | Raros C |
|---|---|---|---:|---:|---:|
| A: numérico | Logística | 0.7267 | 0.7251 | 0.7265 |
| A: numérico | HistGB | 0.7789 | 0.7780 | 0.7788 |
| B: estados + atraso | Logística | 0.7687 | 0.7671 | 0.7686 |
| B: estados + atraso | HistGB | 0.7782 | 0.7786 | 0.7782 |

- **Rango de AUC medio de las 12 combinaciones:** `0.7251–0.7789`.
- **Mejor AUC medio:** `0.778908` (raros A, PAY_* A, HistGB), usado solo para
  el experimento de `SEX`; no es una recomendación.

### Opciones presentadas

- **A)** Agrupar `EDUCATION {0,5,6} -> 4` y `MARRIAGE {0} -> 3`.
- **B)** Eliminar esas filas solo del entrenamiento de cada fold, validando
  en las mismas particiones completas.
- **C)** Dejar los códigos raros como categorías propias (one-hot).

### Decisión del usuario (respuesta textual, sin editar)

**Respuesta textual del usuario:**

```text
"(a) Qué decidí: agrupar los códigos no documentados con la categoría
    "otros": EDUCATION 0, 5 y 6 pasan a 4, y MARRIAGE 0 pasa a 3.

(b) Alternativas que descarté: eliminar esas filas del entrenamiento, o
    dejarlas como categorías propias (one-hot).

(c) Por qué: con boosting, las tres variantes dan el mismo AUC de
    validación dentro del ruido: agrupar 0.7789, eliminar 0.7780 y
    categorías propias 0.7788, con desviación entre particiones de
    ≈0.005 a 0.006. Como el AUC no decide, elegí la más simple y la que no
    pierde filas. Descarté eliminar porque quita ≈1.3% de los datos sin
    ganancia y cambiaría la población que estoy auditando; descarté las
    categorías propias porque dejan columnas de 14 a 54 filas.

(d) Qué pierdo: al agrupar mezclo códigos que quizá significan cosas
    distintas, y que sean equivalentes a "otros" es un supuesto que no
    puedo verificar, porque la documentación de UCI no define esos
    códigos."
```

Los incisos (a)–(d) cubren, en el orden pedido, elección, alternativa
descartada, razón y qué se pierde.

---

## Decisión 2 — Tratamiento de `PAY_0..PAY_6`

### Evidencia

- **Archivo/libreta:** `problema-1/02_decisiones_datos.ipynb`, secciones 1, 3
  y 4.
- **Tablas:**
  - Sección 1: `tabla_pay_0`, `tabla_comparacion` para `PAY_0`.
  - Sección 3: `tabla_significado_pay_0`, `tabla_significado_pay_2`.
  - Sección 4: `tabla_combinaciones`.
  - Archivos: `resultados/significado_pay_0.csv`,
    `resultados/significado_pay_2.csv`,
    `resultados/decisiones_datos.csv`.
- **Gráficas:** no se produjeron gráficas en esta evidencia.

**Cifras clave de `PAY_0` (solo 70% de entrenamiento):**

| Código | n | Tasa default | % BILL_AMT1 ≤ 0 | Mediana BILL_AMT1 | Mediana PAY_AMT1 | Mediana PAY_AMT1/BILL_AMT1 |
|---:|---:|---:|---:|---:|---:|---:|
| -2 | 1,913 | 12.86% | 31.68% | 1,289 | 1,131 | 0.813 |
| -1 | 3,985 | 17.11% | 0.10% | 2,711 | 1,661 | 0.775 |
| 0 | 10,308 | 12.51% | 0.00% | 50,094 | 3,000 | 0.053 |
| 1 | 2,599 | 33.90% | 46.09% | 4,473 | 0 | 0.000 |
| 2 | 1,850 | 69.89% | 0.00% | 41,282.5 | 2,000 | 0.049 |

**Verificación con `PAY_2` (solo 70% de entrenamiento):**

| Código | n | Tasa default | % BILL_AMT2 ≤ 0 | Mediana BILL_AMT2 | Mediana PAY_AMT2 | Mediana PAY_AMT2/BILL_AMT2 |
|---:|---:|---:|---:|---:|---:|---:|
| -2 | 2,659 | 17.98% | 55.40% | 0 | 181 | 0.866 |
| -1 | 4,211 | 16.41% | 11.61% | 1,989 | 1,599 | 0.957 |
| 0 | 11,016 | 15.69% | 1.80% | 48,703.5 | 2,863.5 | 0.050 |
| 1 | 19 | 15.79% | 73.68% | 0 | 2 | 0.333 |
| 2 | 2,750 | 55.85% | 1.75% | 27,704 | 1,621.5 | 0.046 |

**AUC medio por variante de `PAY_*`** (mismas 5 particiones):

| Raros | Modelo | PAY_* A | PAY_* B |
|---|---|---:|---:|
| A | Logística | 0.7267 | 0.7687 |
| A | HistGB | 0.7789 | 0.7782 |
| B | Logística | 0.7251 | 0.7671 |
| B | HistGB | 0.7780 | 0.7786 |
| C | Logística | 0.7265 | 0.7686 |
| C | HistGB | 0.7788 | 0.7782 |

### Opciones presentadas

- **A)** Usar `PAY_0..PAY_6` numéricos tal cual.
- **B)** Crear indicadores para `-2`, `-1`, `0` y una columna de meses de
  atraso para valores `>= 1`.

### Decisión del usuario (respuesta textual, sin editar)

**Respuesta textual del usuario:**

```text
"(a) Qué decidí: tratar PAY_0..PAY_6 de forma distinta según el modelo. En
    boosting las dejo como números tal cual (codificación numérica). En
    regresión logística las separo en indicadores de estado (-2, -1, 0)
    más una variable de meses de atraso (valores >= 1).

(b) Alternativas que descarté: en boosting, la codificación por estados más
    meses de atraso; en regresión logística, la codificación numérica.

(c) Por qué: en boosting las dos codificaciones dan el mismo AUC de
    validación dentro del ruido (0.7789 con numérico contra 0.7782 con
    estados, con desviación entre particiones de ≈0.006); los árboles
    pueden cortar en -2, -1 y 0 por su cuenta, así que usé la más simple.
    En regresión logística la diferencia sí es real: el AUC pasa de 0.7267
    con numérico a 0.7687 con estados (+0.042, con desviaciones de
    0.003 a 0.007), porque -2, -1 y 0 son estados distintos y no una
    escala numérica, y la logística los trata como si lo fueran.

(d) Qué pierdo: al usar codificaciones distintas, la comparación entre
    modelos mezcla el efecto del modelo con el de la codificación. Además,
    en boosting acepto que el árbol necesite varios cortes para separar
    códigos que no tienen un orden numérico real, y no sé qué significan
    el 0 y el -2 porque la documentación de UCI no los define."
```

Los incisos (a)–(d) cubren, en el orden pedido, elección, alternativa
descartada, razón y qué se pierde.

---

## Decisión 3 — Incluir o excluir `SEX`

### Evidencia

- **Archivo/libreta:** `problema-1/02_decisiones_datos.ipynb`, sección 5.
- **Tablas:** `tabla_sexo`, `tabla_combinaciones`,
  `resultados/decisiones_datos.csv`.
- **Gráficas:** no se produjeron gráficas en esta evidencia.
- **Combinación usada:** raros A, `PAY_*` A, HistGB (la de mayor AUC medio,
  usada solo para este contraste).

**Cifras clave:**

| Entrada | AUC medio | Desviación estándar |
|---|---:|---:|
| Con `SEX` | 0.778908 | 0.006115 |
| Sin `SEX` | 0.778779 | 0.006701 |

- **Diferencia de AUC (con `SEX` − sin `SEX`):** `+0.000129`.
- La diferencia es mucho menor que la desviación estándar entre folds.

### Opciones presentadas

- **A)** Mantener `SEX` como variable de entrada.
- **B)** Excluir `SEX` de las variables de entrada.

### Decisión del usuario (respuesta textual, sin editar)

**Respuesta textual del usuario:**

```text
"(a) Qué decidí: SEX no entra como variable de entrada en el modelo
    principal. Se conserva aparte, solo como variable de auditoría para
    analizar los resultados por grupo.

(b) Alternativa que descarté: incluir SEX como entrada del modelo.

(c) Por qué: SEX quedó fuera porque la diferencia de AUC con y sin SEX
    (0.00013) es mucho menor que la variación entre particiones de
    validación cruzada (≈0.006); no aporta valor predictivo medible, así
    que no justifico usar un atributo protegido. (Con SEX: 0.77891; sin
    SEX: 0.77878, boosting, 5 particiones.)

(d) Qué pierdo: la comparación se hizo solo con boosting, no con regresión
    logística. Que SEX no mejore el AUC no significa que quitarla elimine
    las diferencias entre grupos, porque otras variables pueden funcionar
    como sustitutas; eso lo evalúo en el inciso (e) de la práctica. Además,
    para analizar equidad sigo necesitando saber el grupo de cada persona,
    así que el sexo no desaparece del análisis, solo del modelo."
```

Los incisos (a)–(d) cubren, en el orden pedido, elección, alternativa
descartada, razón y qué se pierde.

### Evidencia posterior (07)

> Nota agregada después de la decisión del usuario. No forma parte del texto
> textual registrado arriba.

- **Libreta:** `problema-1/07_sin_atributo.ipynb` (mismo split, mismos
  hiperparámetros de 03, misma calibración isotónica global, umbral `1/6`).
- **Incluir `SEX` como entrada no elimina las diferencias; las amplía.**
  - La brecha de FPR pasa de `0.042289` (modelo sin `SEX`) a `0.082428`
    (con `SEX`): **+0.040**, IC95% pareado `[0.031, 0.048]` (no contiene el
    cero).
  - La brecha de FNR pasa de `-0.023922` a `-0.053067`: **-0.029**, IC95%
    pareado `[-0.042, -0.017]` (no contiene el cero).
  - **ΔAUC = -0.0003** (0.779162 con `SEX` − 0.779463 sin `SEX`).
  - **Δganancia = +35 unidades de G** (2019 − 1984), IC95% pareado
    `[-36, 100]` (contiene el cero, dentro del ruido).
- **Lectura:** confirmar la advertencia del inciso (d) de la decisión del
  usuario: quitar `SEX` del modelo no elimina las diferencias entre grupos
  (otras variables predicen `SEX` con AUC 0.650929 en CV y 0.649996 en
  prueba), y meterla como entrada las agrava. La decisión de dejar `SEX`
  fuera del modelo principal se mantiene.

---

## Cierre del Problema 1

**Estado:** las 3 decisiones de datos tienen respuesta textual registrada y
las definiciones del enunciado usadas en el análisis están fijadas en
`common.py`.

### Decisiones registradas (3)

1. **Tratamiento de códigos raros** (`EDUCATION` 0/5/6, `MARRIAGE` 0):
   registrada; respuesta textual en la sección “Decisión 1”.
2. **Tratamiento de `PAY_0..PAY_6`:** registrada; respuesta textual en la
   sección “Decisión 2”.
3. **Incluir o excluir `SEX`:** registrada; respuesta textual en la sección
   “Decisión 3”.

### Estado de definiciones

- **Decisiones de datos con razón textual:** 3 de 3 registradas.
- **Definiciones del enunciado usadas:** `L = 5G` y umbral del banco `= 1/6`
  (fijadas en `common.py`); el criterio de evaluación y el formato de
  entrega se documentan en este repositorio.

---

## Otras definiciones del enunciado

Estas no son decisiones de datos producidas por esta libreta, sino
definiciones del enunciado que el análisis ya utiliza:

- `L = 5G` (fijado en `common.py`).
- Umbral del banco `= 1/6` (fijado en `common.py`).
- Criterio de evaluación/validación: validación cruzada, ventana temporal y
  reporte único en prueba, descritos en las libretas 03–07.
- Formato de entrega: documentos de `entrega/` y código reproducible de
  `problema-1/`.

---

## Inciso (b) — calibración (evidencia, no decisión)

- **Libreta:** `problema-1/04_calibracion.ipynb`.
- **Evidencia:** archivos `resultados/04_*` y figuras
  `resultados/04_calibracion_*.png`.
- **Modelo calibrado:** `HistGradientBoostingClassifier`, elegido por AUC de
  validación cruzada (`0.781596 ± 0.005092`), no por AUC de prueba.
- **Calibrador global:** isotónico, elegido por menor log-loss en validación
  cruzada anidada dentro del 70% (`0.426337` frente a `0.426558` del
  sigmoide).
- **Decisión registrada:** no calibrar por grupo (ver Decisión 4). En
  `04_calibracion` se aplicó un único calibrador global; la razón y la
  evidencia están en la Decisión 4.

---

## Inciso (c) — tabla por grupo, identidad y bootstrap (evidencia)

- **Libreta:** `problema-1/05_tabla_grupos.ipynb`.
- **Evidencia:** archivos `resultados/05_*` y figuras
  `resultados/05_hist_*_diferencia.png`.
- **Probabilidades usadas:** `prob_calibrada` del archivo
  `resultados/04_probabilidades_calibradas_test.csv`, es decir, el calibrador
  global isotónico del inciso (b). No se usaron las probabilidades originales
  porque sí hubo calibración.
- **Umbral:** `1/6` (`G = 1`, `L = 5`); se rechaza si el puntaje es `>= 1/6`.
- **División entre cero:** no ocurrió en la tabla por grupo ni en las 2,000
  réplicas bootstrap. Todos los denominadores fueron positivos.
- **Identidad verificada:** `FPR = p/(1-p) · (1-PPV)/PPV · (1-FNR)` con error
  absoluto de `5.55e-17` en hombres y `0.0` en mujeres (ruido de punto
  flotante).
- **Bootstrap:** 2,000 réplicas de todo el conjunto de prueba con `SEED`.
  - `FPR_H − FPR_M = 0.042289`, IC 95% `[0.020183, 0.064465]`; **no** contiene
    el cero, por lo que la diferencia de `FPR` parece real.
  - `FNR_H − FNR_M = -0.023922`, IC 95% `[-0.064162, 0.015438]`; **sí**
    contiene el cero, por lo que la diferencia de `FNR` puede ser ruido.
- **Decisiones nuevas:** ninguna. Este inciso es descriptivo y de auditoría;
  no se calibró ni se ajustó nada por grupo.

---

## Decisión 4 — Método de calibración

> **Decisión aceptada por el alumno.** La razón siguiente fue redactada por
> el asistente con evidencia propia.

### Evidencia

- **Libreta:** `problema-1/04_calibracion.ipynb`.
- **Evidencia:** `resultados/04_comparacion_cv.csv`,
  `resultados/04_metricas_test.csv` y `resultados/04_metodo_calibracion.csv`.
- **Criterio:** menor log-loss en validación cruzada anidada dentro del 70%
  (5 particiones externas; dentro de cada una, `CalibratedClassifierCV(cv=5)`).
  El 30% de prueba no se usó para elegir el método.

| Método | Log-loss CV | Brier CV | ECE CV |
|---|---|---:|---:|
| Sin calibrar | 0.427871 | 0.133874 | 0.014938 |
| Sigmoide | 0.426558 | 0.133503 | 0.008658 |
| Isotónico | 0.426337 | 0.133471 | 0.004595 |

- **Resultado:** se eligió el calibrador global isotónico, con `ECE = 0.012525`
  en prueba y `E[S|G=g] − p_g ≤ 0.001` por grupo según
  `resultados/04_tabla_grupos_despues.csv`.

### Opciones presentadas

- **A)** No calibrar.
- **B)** Calibración sigmoide global.
- **C)** Calibración isotónica global.
- **D)** Calibración por grupo (un calibrador por grupo).

### Decisión

**(a) Qué decidí:** usar calibración isotónica global, con un único calibrador
aplicado a todos los solicitantes. No se calibró por grupo.

**(b) Alternativas que descarté:** sin calibrar (peor log-loss y ECE),
sigmoide (técnicamente peor en log-loss, Brier y ECE que isotónica en la CV
anidada) y calibración por grupo (usaría `SEX` para calibrar, lo que se aparta
de la decisión de dejar `SEX` fuera como variable de auditoría, y
fragmentaría el entrenamiento).

**(c) Por qué:** el criterio era menor log-loss en CV anidada dentro del 70%.
El isotónico obtuvo `0.426337`, frente a `0.426558` del sigmoide y `0.427871`
sin calibrar, y ganó también en Brier y ECE. El AUC no se usó para elegir
porque la calibración busca probabilidades correctas, no reordenar casos. El
30% de prueba se evaluó una sola vez después de la elección, solo como
reporte.

**(d) Qué pierdo:** la isotónica es un escalón flexible y puede sobreajustar
con muestras pequeñas y producir probabilidades repetidas; el sigmoide es más
estable y suave. Además, un calibrador global garantiza
`P(Y=1|S=s,G=g) ≈ s` solo en promedio: no corrige por sí mismo las
diferencias de `FPR`/`FNR` entre grupos (ver inciso (b) de la teoría), que
deben tratarse a nivel de umbral.

---

## Decisión 5 — Definición de equidad

> **Decisión aceptada por el alumno.** La razón siguiente fue redactada por
> el asistente con evidencia propia.

### Evidencia

- **Libreta:** `problema-1/06_intervencion.ipynb`.
- **Evidencia:** `resultados/06_comparativa_definiciones.csv`,
  `resultados/06_pares_umbrales_igual_fpr.csv` y
  `resultados/06_ganancia_vs_diferencia.png`.
- **Bootstrap de 05 (`05_bootstrap_ic95.csv`):** `FPR_H − FPR_M = 0.042289`,
  IC 95% `[0.020183, 0.064465]` (no contiene el cero);
  `FNR_H − FNR_M = -0.023922`, IC 95% `[-0.064162, 0.015438]` (sí contiene el
  cero).

### Opciones presentadas

- **Paridad demográfica:** misma tasa de rechazo por grupo.
- **Igual FPR:** mismo porcentaje de no incumplidores rechazados.
- **Igual FNR:** mismo porcentaje de incumplidores aprobados.
- **Igual FPR y FNR:** ambas diferencias dentro de la tolerancia.
- **Paridad predictiva:** mismo valor predictivo positivo (PPV).

### Decisión

**(a) Qué decidí:** elegir **igual FPR** como definición de equidad.

**(b) Alternativas que descarté:** paridad demográfica (iguala tasas de
rechazo pero deja diferencias de errores de 1.76 y 1.26 puntos
porcentuales en OOF) y paridad predictiva (iguala la PPV pero deja diferencias
de FPR de 10.96 puntos y de FNR de -7.51 en OOF). Igual FNR e igual FPR y
FNR no se descartaron por malas, sino porque en esta cuadrícula eligen el
mismo par que igual FPR, así que no cambian el resultado.

**(c) Por qué:** es la única diferencia entre grupos que el bootstrap de 05
muestra como real: el intervalo de FPR no contiene el cero y el de FNR sí. El
banco debería corregir primero la disparidad que la evidencia respalda. Además,
igual FPR es la violación más visible del principio de tratar igual a quien
paga: rechazar a 36.74% de los hombres que no incumplieron frente a
32.51% de las mujeres (prueba).

**(d) Qué pierdo:** al igualar FPR se despriorizan otras definiciones. La
tolerancia (0.01) y la cuadrícula (0.05–0.60, paso 0.01) son
convenciones elegidas para el análisis; otra tolerancia podría dar otro par.
La igualdad de FPR tampoco garantiza igualdad de FNR o de PPV, y su
cumplimiento debe vigilarse en carteras nuevas.

---

## Decisión 6 — Selección de umbrales

> **Decisión aceptada por el alumno.** La razón siguiente fue redactada por
> el asistente con evidencia propia.

### Evidencia

- **Libreta:** `problema-1/06_intervencion.ipynb`.
- **Evidencia:** `resultados/06_umbrales_elegidos.csv`,
  `resultados/06_top_pares_igual_fpr.csv`,
  `resultados/06_costo_equidad.csv` y `resultados/06_bootstrap_ic95.csv`.
- **Umbrales elegidos:** t_H = 0.19, t_M = 0.18, con ganancia OOF de 4697
  unidades de G y |FPR_H − FPR_M| OOF = 0.008511 ≤ 0.01.
- **Costo en prueba:** ganancia 1977 frente a 1984 de la línea base:
  pérdida de 7 unidades de G; no se distingue de cero: pérdida de 7 unidades
  de G, IC95% [-100, 116] (0.35% de la ganancia base, 0.000778 por
  solicitante como magnitud puntual, no como conclusión).
- **Bootstrap posterior:** FPR_H − FPR_M = 0.009493, IC 95%
  [-0.012185, 0.031146] (contiene el cero); diferencia de ganancia -7, IC
  [-116, 100.025] (contiene el cero); pérdida (base − intervención) = 7, IC
  [-100, 116] (contiene el cero).

### Opciones presentadas

- **Umbral único 1/6 para todos** (línea base).
- **Par de umbrales por grupo** (t_H, t_M) elegido en la cuadrícula con
  predicciones fuera de muestra del 70%.

### Decisión

**(a) Qué decidí:** usar dos umbrales, t_H = 0.19 y t_M = 0.18, elegidos
con predicciones fuera de muestra (`cross_val_predict`, cv=5) sobre el 70%.
El 30% de prueba se usó solo para reportar el resultado, sin afinar los
umbrales.

**(b) Alternativas que descarté:** mantener el umbral único 1/6 (conserva
una diferencia de FPR de 4.23 puntos porcentuales que el bootstrap de 05
señaló como real) y elegir los umbrales sobre la prueba (sería usar los datos
de evaluación para tomar la decisión, lo que invalida la estimación del costo
y de la equidad resultante).

**(c) Por qué:** con igual FPR y tolerancia 0.01, el par (0.19, 0.18) es el
que maximiza la ganancia OOF entre los factibles (4697 unidades de G,
frente a 4745 de la línea base). En prueba el costo no se distingue de
cero: pérdida de 7 unidades de G, IC95% [-100, 116] (0.35% y 0.000778 por
solicitante solo como magnitud puntual), así que no es distinguible del ruido
con esta muestra. Los desempates se resolvieron de forma determinista: menor
diferencia máxima, luego menor t_H y menor t_M.

**(d) Qué pierdo:** usar umbrales distintos por grupo es una política explícita
de trato diferenciado por atributo protegido y requiere justificación jurídica y
de gobernanza; no basta con que sea técnicamente barata. Además, el par se
eligió en una sola cuadrícula y tolerancia: su estabilidad ante otras
tolerancias o carteras nuevas debe vigilarse. La intervención no elimina la
información de `SEX`: sigue siendo necesaria para auditar y para aplicar el
umbral correcto.

---

## Nota sobre el recuadro (e) de la teoría

> **Decisión aceptada por el alumno.**

- El recuadro "Tú decides" de `entrega/p1_teoria.tex` (inciso (e)) propone
  igual FPR con (t_H, t_M) = (0.19, 0.18), acepta umbrales distintos por grupo
  con justificación y mantiene SEX fuera del modelo principal, en consistencia
  con la Decisión 3.
- No introduce decisiones nuevas más allá de las Decisiones 5 y 6; resume los
  resultados de `06_intervencion.ipynb` y `07_sin_atributo.ipynb` y los
  conecta con la imposibilidad del inciso (b).
- El recuadro se defiende oralmente con la evidencia de las Decisiones 5 y 6.
