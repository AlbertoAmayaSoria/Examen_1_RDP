# AGENTS.md — Contexto del Examen 1 de Reconocimiento de Patrones

Este archivo es la fuente de contexto para cualquier agente que trabaje en este
repositorio. **Léelo completo antes de escribir código o tomar decisiones.**
Si algo del enunciado no está aquí, no lo inventes: pregunta.

---

## 1. Contexto académico

- **Materia:** Reconocimiento de Patrones.
- **Institución:** Universidad de Sonora.
- **Evaluación:** Examen 1 (nivel intermedio).
- **Entregable:** código reproducible, comentado en español, con gráficas
  explicadas y defendible oralmente.

---

## 2. Parámetros personales (fijos, no modificar)

| Parámetro | Valor |
|---|---|
| Expediente | `223210350` |
| `a` | `0` |
| `b` | `5` |
| Semilla aleatoria | `223210350` |

- La semilla `223210350` debe usarse en **todo** proceso aleatorio.
- `a` y `b` deben quedar como constantes explícitas en el código, no
  hardcodeadas de forma dispersa.

---

## 3. Problema 1

- **Criterio de agrupamiento:** por `SEXO` (hombres vs. mujeres).
- **Parámetro `L`:** `5G`.
- **Umbral del banco:** `1/6`.

> No reinterpretar `L = 5G` ni el "umbral del banco" sin confirmarlo con el
> enunciado. Si el código requiere una definición que no está en este archivo,
> preguntar antes de implementar.

---

## 4. Problema 2

- **Variable objetivo:** `registered`.
- **Candidatos de kernel:**
  - `kI = exp(-d^2 / 2)`
  - `kII = max(0, 1 - d^2 / 4)`
- **Distancia `d`:** distancia circular.

> La definición exacta de la distancia circular (unidades, periodo, tratamiento
> de extremos) debe tomarse del enunciado. Si no está disponible, preguntar.

---

## 5. Reglas de trabajo (obligatorias)

1. **Reproducibilidad total.** El código debe correr de principio a fin en un
   entorno limpio y usar la semilla `223210350` en todo lo aleatorio. Nada de
   resultados que dependan de estado previo, rutas locales o datos ya
   descargados a mano.
2. **Datos desde la fuente original.** Los datos se descargan desde UCI dentro
   del propio código. **Nunca** se suben al repositorio (agregar `data/` al
   `.gitignore`).
3. **Comentarios en español.** Cada paso debe explicar **qué** se hace y
   **por qué** se hace, no solo repetir la sintaxis.
4. **Gráficas completas.** Toda gráfica debe llevar:
   - ejes con nombre,
   - título,
   - un comentario o texto que explique qué significa y qué conclusión se
     obtiene de ella.
5. **"Tú decides" = decide el alumno.** Donde el examen diga "Tú decides",
   **no elegir por cuenta propia**. Presentar 2 o 3 alternativas con sus
   costos/ventajas y **esperar la respuesta** antes de continuar.
6. **Resumen de 5 líneas después de cada paso.** El alumno debe poder
   defender el trabajo oralmente; cada paso termina con un resumen breve
   (ver plantilla en la sección 8).

---

## 6. Datos y reproducibilidad

- Fuente: **UCI Machine Learning Repository**.
- El nombre exacto del dataset, su URL/ID y el esquema de columnas deben
  confirmarse con el enunciado antes de implementar la descarga.
- Los archivos descargados van en `data/` (ignorado por git) o en un
  directorio temporal; el repositorio solo contiene código, reportes y
  figuras que sí deban entregarse.
- Fijar semilla también para particiones, inicializaciones, barajados y
  cualquier muestreo.
- Evitar rutas absolutas y dependencias del directorio actual.
- Registrar versiones de dependencias para que el entorno limpio sea
  replicable.

Estructura del repositorio (la raíz del repo es esta carpeta):

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml        # dependencias con versiones fijas
├── .gitignore
├── data/                 # descargas, no versionado
├── problema-1/
│   └── README.md
└── problema-2/
    └── README.md
```

---

## 7. Decisiones pendientes ("Tú decides")

Registrar aquí cada decisión que el examen deje abierta. **No avanzar** hasta
tener la elección del alumno.

| Fecha | Paso | Alternativas presentadas | Costos/ventajas | Elección del alumno |
|---|---|---|---|---|
| — | — | — | — | — |

También quedan por confirmar con el enunciado, si no aparecen en él:

- [ ] Dataset UCI exacto del Problema 1 y del Problema 2.
- [ ] Definición operativa de `L = 5G`.
- [ ] Definición operativa del "umbral del banco" `= 1/6`.
- [ ] Unidad y periodo de la distancia circular `d`.
- [ ] Criterio de evaluación/validación esperado.
- [ ] Formato de entrega y lenguaje/entorno exigido.

---

## 8. Plantilla de resumen de 5 líneas por paso

Después de cada paso, entregar exactamente este bloque (una línea por punto):

1. **Qué hice:** …
2. **Por qué:** …
3. **Resultado:** …
4. **Supuestos/riesgos:** …
5. **Siguiente paso:** …

---

## 9. Checklist antes de entregar

- [ ] El flujo corre completo desde cero en un entorno limpio.
- [ ] La semilla `223210350` está aplicada en todo lo aleatorio.
- [ ] Los datos se descargan desde UCI y no están versionados.
- [ ] Todo el código está comentado en español, explicando el porqué.
- [ ] Todas las gráficas tienen ejes, título e interpretación.
- [ ] Las decisiones "Tú decides" fueron consultadas y registradas.
- [ ] Cada paso tiene su resumen de 5 líneas para la defensa oral.
- [ ] No se modificaron los parámetros personales (`a`, `b`, semilla, expediente).
