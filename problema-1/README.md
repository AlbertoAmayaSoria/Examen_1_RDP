# Problema 1

## Objetivo

Análisis de grupos definidos por **SEXO** (hombres vs. mujeres) con:

- **`L = 5G`** (`G = 1`, `L = 5`)
- **Umbral del banco = `1/6`** (`G/(G+L)`)

La pregunta central es si el modelo de riesgo crediticio trata distinto a
hombres y mujeres, qué definición de equidad conviene corregir, cuánto cuesta
corregirla y si usar el sexo como variable de entrada resolvería la
disparidad. El flujo va de los datos a la teoría: las libretas producen
tablas y gráficas, y los documentos de `entrega/` desarrollan los incisos
(a)--(e).

## Estado

Descarga, exploración, decisiones de datos, modelos, calibración, tabla por
grupo, intervención por umbrales y contraste con `SEX` como entrada
implementados. **Las decisiones sobre códigos raros, `PAY_*` y `SEX` están
registradas textualmente en `bitacora_notas.md` y aplicadas en `03_modelos`.**
Las constantes están en `common.py`: `G = 1`, `L = 5`, `UMBRAL = 1/6`,
`SEED = 223210350`. El 30% de prueba se usa una sola vez por modelo para
reportar su AUC; la selección del mejor modelo y del método de calibración se
hace con validación cruzada dentro del 70%. La intervención por umbrales de
`06_intervencion` elige los umbrales con predicciones fuera de muestra del
70% y reporta en prueba sin afinarlos ahí.

## Ejecución

Desde la raíz del repositorio, con el entorno conda activado:

```bash
conda activate rp-examen
./problema-1/run_all.sh
```

El script `run_all.sh` ejecuta en orden (8 pasos):

1. `data.py`: descarga el ZIP de UCI (si hace falta), extrae el `.xls` en
   `data/` y devuelve el DataFrame con el encabezado corregido.
2. `01_exploracion.ipynb`: libreta emparejada con `01_exploracion.py` que
   imprime los conteos y la tasa de incumplimiento por sexo.
3. `02_decisiones_datos.ipynb`: evidencia para decidir códigos raros, `PAY_*`
   y `SEX`; guarda el split 70/30 en `resultados/split_indices.csv`.
4. `03_modelos.ipynb`: aplica las tres decisiones, busca hiperparámetros solo
   en el 70% (CV de 5 particiones, AUC), reentrena, calcula el AUC de prueba
   una sola vez por modelo y elige el mejor por AUC de validación cruzada.
5. `04_calibracion.ipynb`: evalúa calibración global y por grupo, compara
   sigmoide contra isotónico con validación cruzada anidada dentro del 70%,
   ajusta un único calibrador global y guarda las probabilidades calibradas.
6. `05_tabla_grupos.ipynb`: usa las probabilidades calibradas, aplica el
   umbral `1/6`, calcula `n`, `p_g`, tasa de rechazo, `PPV`, `FPR` y `FNR` por
   grupo, verifica la identidad de `FPR` y hace bootstrap de 2,000 réplicas.
7. `06_intervencion.ipynb`: evalúa las cinco definiciones de equidad, elige
   los umbrales por grupo con predicciones fuera de muestra (`cv=5`) sobre el
   70%, reporta en prueba, calcula el costo de equidad y el bootstrap posterior.
8. `07_sin_atributo.ipynb`: reentrena el mismo boosting **con `SEX` como
   entrada**, repite tabla y bootstrap de 05, mide asociaciones con `SEX` y un
   clasificador que predice `SEX` desde las demás variables.

Las funciones reutilizables viven en `data.py`, `reportes.py`,
`evidencia_datos.py`, `modelos.py` y `equidad.py`. La semilla común está en
`common.py` (`SEED = 223210350`).

## Resultados generados

Además de las tablas de `02_decisiones_datos`, `03_modelos`, `04_calibracion`
y `05_tabla_grupos` (ver historial de este README), las libretas recientes
escriben:

La libreta `06_intervencion` escribe en `resultados/`:

- `06_tabla_base_test.csv` y `06_tabla_base_oof.csv`: línea base con umbral
  único `1/6` en prueba y en OOF (con ganancia).
- `06_comparativa_definiciones.csv`: las cinco definiciones con umbrales,
  ganancia OOF y prueba, y diferencias de `FPR`, `FNR`, `PPV` y rechazo.
- `06_pares_umbrales_igual_fpr.csv`: los 3,136 pares `(t_H, t_M)` de la
  cuadrícula con ganancia, diferencias y marca de factibilidad.
- `06_top_pares_igual_fpr.csv`: los 10 mejores pares factibles de igual `FPR`.
- `06_umbrales_elegidos.csv`: el par elegido (`t_H = 0.19`, `t_M = 0.18`),
  su ganancia OOF y prueba y el costo.
- `06_antes_despues.csv`: tabla antes/después por grupo y total.
- `06_costo_equidad.csv`: ganancia perdida en unidades de `G` y en porcentaje,
  en prueba y en OOF.
- `06_bootstrap_ic95.csv`, `06_bootstrap_replicas.csv`,
  `06_bootstrap_ganancia.csv`: bootstrap de 2,000 réplicas de la diferencia de
  `FPR` y `FNR` después de intervenir y de la diferencia de ganancia.
- `06_ganancia_vs_diferencia.png`: ganancia OOF contra `|dif_FPR|`, con la
  línea base y el par elegido.
- `06_metricas_vs_umbral.png`: `FPR` y `FNR` por grupo frente al umbral.
- `06_hist_fpr_despues.png` y `06_hist_ganancia_despues.png`: histogramas
  bootstrap posteriores.
- `06_resumen_definiciones.csv`: resumen de las cinco definiciones.

La libreta `07_sin_atributo` escribe además:

- `07_tabla_grupos.csv`: tabla por grupo del modelo con `SEX` como entrada.
- `07_bootstrap_ic95.csv` y `07_bootstrap_replicas.csv`: bootstrap de la
  diferencia de `FPR` y `FNR` para ese modelo.
- `07_comparacion_modelos.csv`: comparación directa sin `SEX` contra con `SEX`.
- `07_asociaciones_sexo.csv`: asociación de cada variable con `SEX`
  (V de Cramér o correlación punto-biserial), ordenada.
- `07_asociaciones_sexo.png`: gráfica de las asociaciones más fuertes.
- `07_clasificador_sexo.csv`: AUC en CV y en prueba del clasificador que
  predice `SEX`.

`resultados/` se crea en tiempo de ejecución; no contiene datos descargados.
Nota: la regla `*.csv` de `.gitignore` hace que estos archivos no se
versionen por defecto.

## Documentos de entrega

En `entrega/`:

- `p1_teoria.tex`: incisos (a)--(e) en prosa, con demostraciones.
- `p1_bitacora.tex`: las seis decisiones con evidencia.
- `p1_memorandum.tex`: nota de una página para el consejo del banco.
- `declaracion_ia.tex`: declaración de uso de asistentes de IA.

## Bitácora de decisiones

`bitacora_notas.md` registra, para cada decisión:

- la evidencia que la sustenta (libreta o archivo, tablas y cifras clave),
- la respuesta textual del alumno, sin parafrasear ni completar, o marcada
  como **PROPUESTA DEL ASISTENTE, pendiente de revisión del usuario** cuando
  la redactó el asistente.

Las 3 decisiones de datos tienen respuesta textual del usuario. Las
decisiones 4 (calibración), 5 (definición de equidad), 6 (umbrales) y el
recuadro (e) de la teoría son propuestas del asistente pendientes de revisión.

## Conclusiones (práctica)

- **Línea base:** con umbral `1/6` e isotónica global, la ganancia de prueba
  es `1984` unidades de `G` (`0.220444` por solicitante). Los hombres tienen
  `FPR = 0.367406` y las mujeres `0.325117`; el bootstrap de 05 muestra que
  esa brecha de `0.042289` es real (`IC95 [0.020183, 0.064465]`), mientras que
  la brecha de `FNR` (`-0.023922`, `IC95 [-0.064162, 0.015438]`) no se
  distingue del ruido.
- **Definición elegida: igual FPR.** En la cuadrícula `0.05`--`0.60`, paso
  `0.01`, tolerancia `0.01`, el par de máxima ganancia OOF es
  `(t_H, t_M) = (0.19, 0.18)`. Igual FNR e igual FPR y FNR eligen el mismo
  par; paridad demográfica (`0.18, 0.17`) y paridad predictiva (`0.16, 0.18`)
  dejan diferencias de errores mayores.
- **Costo de ser justo:** la ganancia baja de `1984` a `1977` unidades de `G`
  en prueba: `7` unidades, `0.35%` de la ganancia base, `0.000778` por
  solicitante. La diferencia de `FPR` baja a `0.009493` (IC95 que contiene el
  cero) y el IC bootstrap de la ganancia contiene el cero.
- **Con `SEX` como entrada:** el AUC es prácticamente el mismo (`0.779162`
  frente a `0.779463`), la ganancia sube a `2019` (`+35`, `+1.76%`), pero la
  brecha de `FPR` se duplica (`0.082428`, IC95 sin cero) y la de `FNR` se
  vuelve significativa (`-0.053067`). Quitar `SEX` no ciega al modelo: otras
  variables predicen `SEX` con AUC `0.650929` en CV y `0.649996` en prueba.
- **Por qué:** con tasas base distintas (`p_H = 0.232980 ≠ p_M = 0.213442`)
  no se pueden igualar a la vez PPV, FNR y FPR sin caer en casos degenerados.
  El modelo principal sigue sin `SEX`; la equidad se atiende con umbrales y
  auditoría, no con el atributo como entrada.

## Puntos por confirmar con el enunciado

- [x] Dataset UCI exacto y URL/ID de descarga (UCI 350).
- [x] Definición operativa de `L = 5G` (`G = 1`, `L = 5`).
- [x] Definición operativa del "umbral del banco" `= 1/6`.
- [x] Definiciones de equidad y cuadrícula de umbrales para los incisos
      (d) y (e).
- [ ] Formato final de entrega y número de página de los documentos LaTeX.

## Reproducibilidad

- Semilla: `223210350`.
- Los datos se descargan desde UCI dentro del código y se guardan en `data/`
  (ignorado por git).
- Comentarios en español explicando el porqué de cada paso.
