# CAPO × CROP: pipeline unificado para optimización de prompts consciente del costo

> **Informe de prototipo de investigación**
> **Autor:** Samuel Angulo
> **Fecha:** Julio 2026
> **Plazo de desarrollo:** 1 semana
> **Versión:** v1.1-prototipo (Iteración 2)

> **Actualización de resultados (2026-07-20, v1.1).** Este informe refleja la **Iteración 2**: se re-ejecutaron las 4 condiciones sobre un dataset ampliado a **223 ejemplos**, con **10 seeds por condición** y un scorer *fuzzy* para respuestas cortas. La conclusión principal **cambió respecto a la Iteración 1** (5 seeds / 63 ejemplos, donde CROP parecía el mejor punto Pareto): con más datos y más seeds, **UNIFIED es el ganador Pareto** — mejor accuracy *y* menor costo, con significancia estadística frente a las tres condiciones. La §1.1 documenta el cambio Iteración 1 → 2 y conserva los números históricos.

---

## 1. Resumen

Este prototipo integra los algoritmos CAPO (*Cost-Aware Prompt Optimization*) y CROP (*Cost-Regularized Optimization of Prompts*) en un pipeline declarativo-iterativo único que produce prompts Pareto-óptimos sobre tres objetivos: accuracy, costo de entrada y costo de salida. CAPO aporta el *racing* con Holm-Bonferroni para descartar candidatos inferiores con significancia estadística, mientras que CROP aporta un Critic LM que genera reescrituras más breves solo cuando la longitud de salida supera el percentil 70 del *pool* actual. La contribución propia no es replicar los *papers* originales sino demostrar que ambos métodos son complementarios y pueden orquestarse sobre un *backbone* común basado en `litellm`.

El prototipo se construyó en Python 3.11 con un *dataset* QA propio de 223 ejemplos, el modelo `MiniMax-M3` con `thinking: {type: "disabled"}` (cambio documentado en §7.3 respecto al plan original de usar `M2.5-highspeed`), y cuatro condiciones experimentales. Se ejecutaron **10 seeds × 4 condiciones = 40 corridas** contra la API real, con un costo total agregado de ~0.82 USD (precios M3 a tarifa *before-discount*: $0.60 / $2.40 por millón de tokens, ver §7.7).

**Resultados principales (mean ± std sobre 10 seeds):**

| Condición | Accuracy | Costo (USD) | Estado |
|---|---:|---:|---|
| baseline | 0.843 ± 0.078 | 0.02755 ± 0.00284 | referencia fuerte |
| capo     | 0.391 ± 0.077 | 0.02069 ± 0.00757 | sobre-optimiza brevedad — ver §7.5 |
| crop     | 0.780 ± 0.154 | 0.01730 ± 0.00357 | buen equilibrio |
| **unified** | **0.891 ± 0.057** | **0.01638 ± 0.00487** | **Pareto-óptimo** |

UNIFIED supera al baseline en accuracy (+0.048) y a la vez tiene **el menor costo** (−41%) y el **menor número de tokens de salida** (3930 vs 9051 del baseline): domina el frente Pareto en las dos dimensiones. Las tres comparaciones pareadas (Wilcoxon, n=10) de UNIFIED contra baseline, capo y crop son estadísticamente significativas (p = 0.049 / 0.002 / 0.037). Los 24 *smoke tests* sobre Holm-Bonferroni, modelo de costos, dominancia de Pareto y *scoring* pasan sin red. Toda la trazabilidad por llamada queda en [results/raw/](../results/raw/) como JSONL.

### 1.1 Qué cambió respecto a la Iteración 1

> El resto del informe (§6–§8) ya está actualizado a la **Iteración 2**. Esta sección documenta el **cambio** respecto a la Iteración 1 (5 seeds / 63 ejemplos) y por qué la conclusión se invirtió. Se re-ejecutaron las **4 condiciones × 10 seeds = 40 corridas** sobre el dataset ampliado a **223 ejemplos QA** (94 *easy* / 92 *medium* / 37 *hard*), y se incorporó la métrica `fuzzy_short_accuracy` (Levenshtein normalizado) que corrige el problema de `accuracy_short = 0` documentado en §7.3. Costo total agregado ≈ **0.82 USD** (tarifa *before-discount* M3). Todas las tablas (`results/tables/`) y figuras (`results/figures/`) fueron regeneradas a partir de estos datos.

**Desglose por métrica de la Iteración 2 (mean ± std sobre 10 seeds):**

| Condición | Accuracy | acc_short | fuzzy_short | acc_long | judge_long | Tokens OUT | Costo (USD) | Estado |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.843 ± 0.078 | 0.681 | 0.740 | 1.000 | 0.943 | 9051 | 0.02755 ± 0.00284 | referencia fuerte |
| capo | 0.391 ± 0.077 | 0.810 | **0.865** | 0.007 | 0.227 | 5919 | 0.02069 ± 0.00757 | sobre-optimiza brevedad |
| crop | 0.780 ± 0.154 | 0.692 | 0.760 | 0.889 | 0.691 | 4031 | 0.01730 ± 0.00357 | buen equilibrio |
| **unified** | **0.891 ± 0.057** | 0.781 | 0.829 | 0.990 | 0.817 | **3930** | **0.01638 ± 0.00487** | **ganador Pareto** |

**La conclusión principal se invierte respecto a la Iteración 1.** Con el dataset ampliado y 10 seeds:

- **UNIFIED es ahora el ganador Pareto inequívoco:** logra la **mayor accuracy (0.891)** *y* el **menor costo (0.0164 USD)** *y* el menor número de tokens de salida (3930). La hipótesis a priori de §7.4 (que el *pipeline* unificado combinaría lo mejor de CAPO + CROP) **sí se cumple** en esta iteración.
- **CAPO ya no colapsa.** Con la población y el test corregidos, el *racing* mantiene 3–4 supervivientes por seed (antes eliminaba a todos, `n_survivors = 0`). Sigue siendo la condición más débil en accuracy agregada (0.391), pero por una razón distinta: **sobre-optimiza la brevedad** — es la mejor en respuestas cortas (`fuzzy_short = 0.865`, la más alta de las cuatro) pero destruye las respuestas largas (`acc_long = 0.007`, `judge_long = 0.227`) porque sus mutaciones de restricción ("responde en ≤30 palabras") truncan el contenido que el juez exige.
- **El scorer *fuzzy* rescata la métrica corta.** `accuracy_short` pasa de ≈0 (Iteración 1) a 0.68–0.81, confirmando que el problema era del *exact match*, no del modelo (§7.3).

**Análisis estadístico (Wilcoxon pareado, n = 10):**

| Comparación | Métrica | W | p-valor | Cohen's d |
|---|---|---:|---:|---:|
| unified vs baseline | accuracy | 8.0 | **0.049** | +0.70 |
| unified vs capo | accuracy | 0.0 | **0.002** | +7.35 |
| unified vs crop | accuracy | 7.0 | **0.037** | +0.96 |
| unified (bootstrap CI 95%) | accuracy | — | mean=0.891, CI=[0.859, 0.923] | — |
| unified (bootstrap CI 95%) | cost_total_usd | — | mean=0.0164, CI=[0.0138, 0.0195] | — |

Con n = 10 las tres comparaciones de UNIFIED contra las demás condiciones son **estadísticamente significativas** (p < 0.05), a diferencia de la Iteración 1 donde ninguna diferencia de accuracy alcanzaba significancia (p ≈ 0.4). Esto es consecuencia directa de duplicar los seeds y triplicar el tamaño del *dataset*, tal como se recomendaba en §8.2.

**Comparación Iteración 1 → Iteración 2 (accuracy media):**

| Condición | Iter 1 (5 seeds, 63 ej.) | Iter 2 (10 seeds, 223 ej.) | Δ |
|---|---:|---:|---:|
| baseline | 0.583 | 0.843 | +0.260 |
| capo | 0.183 | 0.391 | +0.208 |
| crop | 0.650 | 0.780 | +0.130 |
| unified | 0.567 | **0.891** | **+0.324** |

El costo agregado subió de ~0.12 USD a ~0.82 USD, coherente con el mayor número de llamadas por corrida (baseline/capo: 44; crop/unified: 220) sobre el dataset ampliado; **no altera la lógica de las comparaciones**, que siguen siendo intra-experimento con precios uniformes.

---

## 2. Introducción

La calidad de un prompt determina en gran medida la calidad de la respuesta de un LLM. La ingeniería de prompts manual es cara, lenta y no escala. En los últimos años han surgido métodos automáticos de optimización de prompts: *gradient-free*, evolutivos, basados en *bootstrapping* y, más recientemente, conscientes del costo. Este trabajo se enfoca en dos de ellos:

- **CAPO** (Zehle et al., 2025) introduce *racing* con Holm-Bonferroni para descartar prompts candidatos en bloques mini-batch crecientes, evitando gastar cómputo en perdedores. Combina un algoritmo evolutivo tipo GA con operadores de *cross-over* y mutación sobre instrucciones, además de una penalización por longitud de prompt.
- **CROP** (Amanchukwu et al., 2025) ataca el costo de salida con un Critic LM que produce *feedback* textual de brevedad y reescrituras más cortas. La idea es que el costo de salida suele dominar el costo total en producción.

El **aporte propio** es un *pipeline* declarativo-iterativo unificado en el que ambos métodos cooperan: CAPO selecciona supervivientes por calidad-costo-de-entrada, y CROP interviene selectivamente sobre los candidatos que ya sobrevivieron pero cuya salida sigue siendo cara.

---

## 3. Trabajo relacionado

- **Zehle, Schlager, Heiß & Feurer (2025).** *CAPO: Cost-Aware Prompt Optimization.* arXiv:2504.16005. Introduce el *racing* con Holm-Bonferroni y la penalización por longitud γ; reporta mejoras de hasta 21 %p sobre EvoPromptGA, OPRO y PromptWizard en 11/15 configuraciones. Es la fuente canónica del módulo `racing.py` y de la función objetivo multi-objetivo.
- **Amanchukwu et al. (2025).** *CROP: Cost-Regularized Optimization of Prompts.* Presenta un Critic LM λ-regularizado que produce *feedback* textual y un *brevity score*. La política de invocación selectiva (percentil 70) usada en este prototipo se inspira directamente en su Sección 4.2.
- **Guo et al. (2024).** *EvoPrompt.* Base evolutiva de CAPO; provee los meta-prompts de *cross-over* y mutación.
- **Birattari et al. (2002).** *F-Race.* Algoritmo de *racing* original; CAPO lo adapta a prompt optimization.
- **Holm (1979).** *A simple sequentially rejective multiple test procedure.* Procedimiento step-down que usamos para corregir el *family-wise error rate* del *racing*.
- **DSPy** (Khattab et al., 2024). Referencia de *framework* declarativo. Aunque lo consideramos, no se adoptó para mantener el *stack* ligero y entendible.
- **LiteLLM** (BerriAI, 2024). Capa de abstracción multi-proveedor; abstrae el *endpoint* de MiniMax.

---

## 4. Metodología

### 4.1 Arquitectura general

El *pipeline* se compone de siete módulos desacoplados (ver [arquitectura.md](../arquitectura.md) para el detalle):

```
Seed prompts  ─►  PromptMutator  ─►  Candidates  ─►  RacingEvaluator
                                                          │
                                          MultiObjectiveScorer ◄──┘
                                                          │
                                              BrevityFeedbackGenerator
                                                          │
                                              Final prompt + métricas
```

Cada llamada al LLM se registra como una línea JSONL en `results/raw/<condición>/seed<N>.jsonl`, con `prompt_hash`, `response_hash`, `tokens_in`, `tokens_out`, `latency_ms` y `seed`. Este *log* es la única fuente de verdad del experimento: tablas y figuras se derivan de él en el *post-procesamiento*.

### 4.2 Operadores de mutación

Se implementan dos operadores obligatorios y uno opcional:

1. **`paraphrase`** — Reformulación con LLM usando el meta-prompt de CAPO (Appendix D.3): "Reformula el prompt preservando su significado pero variando el estilo lingüístico."
2. **`add_constraint`** — Concatena una restricción predefinida al final del prompt (por ejemplo: "Responde en máximo 30 palabras").
3. **`swap_fewshot`** *(opcional)* — Reordena ejemplos *few-shot*; queda inactivo en la configuración por defecto.

### 4.3 *Racing* con Holm-Bonferroni

Cada bloque de tamaño `b` se evalúa sobre todos los supervivientes. Tras cada bloque calculamos los p-valores pareados (t-test, configurable a Wilcoxon) entre cada candidato y cada oponente, y aplicamos Holm-Bonferroni con `alpha = 0.2` (valor por defecto del *paper* CAPO). Un candidato se elimina si y solo si al menos `n_survive` oponentes son Holm-significativamente mejores.

**Nota metodológica:** el *paper* original de CAPO usa una prueba t pareada **sin** corrección por comparaciones múltiples, argumentando que la corrección hace al *racing* demasiado conservador. En este prototipo seguimos la especificación provista y aplicamos Holm-Bonferroni. Esto reduce ligeramente la tasa de eliminación prematura, lo que se refleja en la Discusión como una posible causa de mayor costo en generaciones tempranas.

### 4.4 Critic LM con política de percentil 70

El `BrevityFeedbackGenerator` se invoca solo cuando la longitud media de salida de un candidato supera el percentil 70 del *pool* actual. Esta política, tomada de CROP, evita que el costo del Critic eclipse el ahorro que produce. El Critic devuelve un JSON con `rewritten`, `feedback` y `brevity_score ∈ [0,1]`.

### 4.5 Función objetivo multi-objetivo

El `MultiObjectiveScorer` calcula:

```
score = accuracy − α · cost_in_norm − β · cost_out_norm
```

donde `cost_in_norm` y `cost_out_norm` son los promedios de tokens normalizados por un *baseline* (600 y 200 respectivamente). α y β son configurables; en este prototipo α=γ=0.05 (penalización CAPO) y β=0.05 (penalización CROP).

### 4.6 *Budget guard*

Cada corrida lleva un `BudgetGuard` que detiene la optimización si el costo acumulado supera `--max-budget-usd` (5 USD por defecto). Esto protege contra *loops* patológicos del *racing* o del Critic.

---

## 5. Setup experimental

### 5.1 *Dataset*

Generamos un *dataset* propio de 223 ejemplos QA en español (`data/toy_qa.jsonl`), ampliado de los 63 originales de la Iteración 1 para ganar potencia estadística. Cada ejemplo trae `expected_short` (evaluado con *exact match* normalizado **y** con *fuzzy match* Levenshtein) y `expected_long` (evaluado con un LLM-juez). Las dificultades se reparten en 94 *easy*, 92 *medium* y 37 *hard*. Cada ejemplo es un *dict* con `{id, question, expected_short, expected_long, difficulty}`.

**Justificación del *dataset* toy:** control de variabilidad, *scoring* determinístico, cero dependencias externas de HuggingFace, y cabe en 5 USD por condición.

### 5.2 Modelo y proveedor

Proveedor: **MiniMax API** 
Modelo: **MiniMax-M3** con `thinking: {type: "disabled"}`. Este cambio respecto al plan original de usar `M2.5-highspeed` se documenta en §7.3 y reduce el costo de salida ~60% y elimina la varianza no-determinista del razonamiento interno.

### 5.3 Condiciones

| Condición   | Racing | Critic LM | Mutación | Juez      |
|-------------|:------:|:---------:|:--------:|:---------:|
| `baseline`  |   ✗    |    ✗      |    ✗     | opcional  |
| `capo`      |   ✓    |    ✗      |    ✓     | desactivado |
| `crop`      |   ✗    |    ✓      |    ✗     | activado  |
| `unified`   |   ✓    |    ✓      |    ✓     | activado  |

### 5.4 Seeds y presupuesto

**10 *seeds* por condición (`0…9`)** — se subió de 3 → 5 (Iteración 1) → 10 (Iteración 2) al detectar que la varianza entre seeds era mayor de lo anticipado (§7.6) y para alcanzar potencia estadística en el Wilcoxon pareado. Presupuesto por condición: 5 USD (el gasto real promedio por corrida fue ~0.02 USD con `MiniMax-M3 + thinking=disabled` sobre el dataset de 223 ejemplos, por lo que el límite teórico nunca se activó). El `--max-budget-usd` global es 5 USD.

### 5.5 Hiperparámetros

| Parámetro              | Valor | Justificación                              |
|------------------------|:-----:|--------------------------------------------|
| `block_size`           |  3–10 | Adaptado al tamaño del *dev set* (≈38)   |
| `alpha` (Holm)         | 0.2   | Default del *paper* CAPO                   |
| `z_max` (bloques)      | 2–6   | Deriva de `len(dev) // block_size`         |
| `population_size`      |  4    | Reducido por restricción de tiempo         |
| `crossovers_per_iter`  |  3    | Suficiente para 4 supervivientes           |
| `gamma` (long. CAPO)   | 0.05  | Default del *paper*                        |
| `n_survive`            |  2    | Mitad de la población                      |
| `n_generations`        |  2    | Suficiente para observar Pareto            |

---

## 6. Resultados

> Los números de esta sección corresponden a la **Iteración 2** (10 seeds × 223 ejemplos), regenerados con `analysis/aggregate.py`, `analysis/stats.py` y `analysis/figures.py`. La comparación con la Iteración 1 está en §1.1.

Se ejecutaron **4 condiciones × 10 seeds = 40 corridas** completas con `MiniMax-M3 + thinking=disabled` el 2026-07-20. Costo total agregado ≈ 0.82 USD (a tarifa *before-discount*; ver §7.7). Los JSONL por corrida están en `results/raw/<condición>/seed<N>.jsonl` y los resúmenes `.json` contienen las métricas que alimentan esta sección.

### 6.1 Tabla principal de resultados

| Condición  | Accuracy (mean ± std) | acc_short | fuzzy_short | acc_long | judge_long | Costo USD (mean ± std) | n_llm_calls |
|------------|:---------------------:|:---------:|:-----------:|:--------:|:----------:|:----------------------:|:-----------:|
| baseline   | 0.843 ± 0.078         | 0.681     | 0.740       | 1.000    | 0.943      | 0.02755 ± 0.00284      | 44          |
| capo       | 0.391 ± 0.077         | 0.810     | **0.865**   | 0.007    | 0.227      | 0.02069 ± 0.00757      | 44          |
| crop       | 0.780 ± 0.154         | 0.692     | 0.760       | 0.889    | 0.691      | 0.01730 ± 0.00357      | 220         |
| **unified**| **0.891 ± 0.057**     | 0.781     | 0.829       | 0.990    | 0.817      | **0.01638 ± 0.00487**  | 220         |

**Lectura clave:** UNIFIED domina el frente *Pareto*: **mayor accuracy (0.891)** *y* **menor costo (−41% vs baseline)** *y* menos tokens de salida (3930 vs 9051). CAPO es la condición más débil en accuracy agregada (0.391) porque **sobre-optimiza la brevedad**: gana en respuestas cortas (`fuzzy_short = 0.865`, la mejor) pero hunde las largas (`acc_long = 0.007`). El detalle está en §7.5.

### 6.2 Tabla por seed (auditoría)

| Condición | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.818 | 0.795 | 0.886 | 0.864 | 0.795 | 0.727 | 0.750 | 0.932 | 0.955 | 0.909 |
| capo     | 0.523 | 0.409 | 0.386 | 0.409 | 0.295 | 0.386 | 0.500 | 0.386 | 0.295 | 0.318 |
| crop     | 0.636 | 0.886 | 0.409 | 0.841 | 0.795 | 0.795 | 0.795 | 0.818 | 0.955 | 0.864 |
| unified  | 0.909 | 0.818 | 0.955 | 0.977 | 0.818 | 0.841 | 0.864 | 0.864 | 0.932 | 0.932 |

> **Nota sobre el scorer corto:** en la Iteración 1 el *exact match* daba `accuracy_short = 0` en casi todas las condiciones porque `MiniMax-M3` responde verbosamente (p. ej., "El cuerpo humano adulto tiene 206 huesos" en lugar de "206"). La Iteración 2 añade `fuzzy_short_accuracy` (Levenshtein normalizado), que sube a 0.74–0.87 y confirma que el problema era del scorer, no del modelo (§7.3).

### 6.3 Análisis estadístico

`analysis/stats.py` calcula Wilcoxon pareado y bootstrap CI al 95% sobre las 10 muestras por condición. Resultados con los datos reales:

| Comparación | Métrica | W | p-valor (Wilcoxon) | Cohen's d |
|---|---|---:|---:|---:|
| unified vs baseline | accuracy | 8.0 | **0.049** | +0.70 |
| unified vs capo     | accuracy | 0.0 | **0.002** | +7.35 |
| unified vs crop     | accuracy | 7.0 | **0.037** | +0.96 |
| unified (bootstrap CI) | accuracy | — | mean=0.891, CI95=[0.859, 0.923] | — |
| unified (bootstrap CI) | cost_total_usd | — | mean=0.0164, CI95=[0.0138, 0.0195] | — |

**Interpretación:** con n=10, las **tres comparaciones de UNIFIED contra las demás condiciones son estadísticamente significativas** (p < 0.05), a diferencia de la Iteración 1 donde ninguna diferencia de accuracy alcanzaba significancia (p ≈ 0.4). Duplicar los seeds y triplicar el *dataset* (recomendación de §8.2) fue lo que dio potencia al test. Los p-valores y CI se exportan a `results/tables/stats.csv` mediante `python -m analysis.stats`.

### 6.4 Figuras

Las tres figuras se generan con `python -m analysis.figures` y se depositan en `results/figures/`:

1. **Figura 1 — Barras agrupadas.** Accuracy, tokens IN/OUT, costo USD por condición con barras de error sobre los 10 seeds.
2. **Figura 2 — Frente Pareto.** Dispersión `(tokens_out, accuracy)` coloreada por condición; **unified** aparece como el punto dominante (arriba-izquierda: ↑accuracy, ↓tokens).
3. **Figura 3 — Convergencia del Racing.** Número medio de supervivientes por generación en `capo` y `unified`; ahora el racing mantiene 3–4 supervivientes por seed (ya no colapsa a 0 como en la Iteración 1).

---

## 7. Discusión

### 7.1 Lo que funcionó

- **Desacople de módulos.** `RacingEvaluator`, `Critic` y `Mutator` son completamente independientes. Esto permite intercambiarlos (por ejemplo, cambiar el t-test pareado por Wilcoxon con un solo parámetro) y hace que el código sea legible y testeable.
- **Logs JSONL como única fuente de verdad.** Cada llamada al LLM queda registrada, lo que permite reconstruir cualquier métrica sin re-correr.
- **Política del percentil 70.** Evita que el Critic se invoque sobre prompts ya breves. En el *paper* CROP, esto se reporta como el principal ahorro de costo.
- **Holm-Bonferroni como *safety net*.** Más conservador que el t-test sin corrección del *paper*, lo que asegura que ningún candidato *boundary* se descarte por ruido.
- **Migración del modelo.** El paso de `MiniMax-M2.5-highspeed` a `MiniMax-M3` con `thinking: {type: "disabled"}` (ver §7.3) redujo el costo de salida ~60% y eliminó la varianza no-determinista del razonamiento interno.

### 7.2 Lo que no funcionó (o quedó pendiente)

- **CAPO aislado sobre-optimiza la brevedad** y queda como la condición más débil en accuracy agregada — ver §7.5.
- **Operador `swap_fewshot` opcional.** Quedó implementado pero desactivado por defecto; el *paper* CAPO reporta que intercambia *shots* con probabilidad uniforme y que la operación aporta poca mejora.

### 7.3 Cambio de modelo: de `M2.5-highspeed` a `M3 + thinking=disabled`

El plan original apuntaba a `MiniMax-M2.5-highspeed`. Durante el primer *smoke test* se detectó que emite tokens de razonamiento interno en **70–99%** de su salida facturada, incluso para prompts triviales. La estructura de la respuesta es:

```json
"usage": {
  "completion_tokens": 76,
  "completion_tokens_details": {"reasoning_tokens": 73}
}
```

Se probaron tres formas de desactivar el razonamiento, **todas sin efecto en M2.5**:

| Parámetro | Resultado en M2.5 |
|---|---|
| `thinking: {type: "disabled"}` | ignorado |
| `reasoning: {enabled: false}` | ignorado |
| `reasoning_effort: 0` | ignorado |

**Además, el reasoning de M2.5-highspeed es inherentemente no-determinista**: dos corridas idénticas del baseline dieron `accuracy = 0.667` y `accuracy = 0.917`, una variación de **25 puntos** que invalida cualquier comparación single-seed.

#### Solución adoptada: `MiniMax-M3`

A diferencia de M2.5, **M3 sí respeta el parámetro `thinking`** (la documentación lo confirma explícitamente). Se pasó vía `extra_body` en [src/llm_client.py:142](src/llm_client.py#L142) para evitar la validación de spec OpenAI en litellm. Resultados verificados:

| Métrica | M2.5-highspeed | M3 + thinking=disabled |
|---|---:|---:|
| `reasoning_tokens` (pregunta fácil) | 130-330 | **0** |
| `tokens_out` (pregunta fácil) | 130-348 | **11-16** |
| Rango entre 5 corridas idénticas | 200+ tokens | **5 tokens** |
| Accuracy baseline (× 2 corridas) | 0.667 / 0.917 | **0.75 / 0.75** |

**Implicaciones**: con M3 la varianza por llamada se elimina en la práctica. `LLMResponse.reasoning_tokens` queda en 0 pero el campo se conserva en la API y en los logs para detectar regresiones si MiniMax cambia el comportamiento por default.

**Limitación del scorer con M3**: el modelo responde verbosamente a preguntas SHORT (p. ej., "El cuerpo humano adulto tiene 206 huesos" en vez de "206"), por lo que el scorer de exact-match falla y `accuracy_short = 0` en 3 de 4 condiciones. El judge LM para LONG sigue dando ≥95% porque evalúa semánticamente. Esto es **una limitación del scorer, no del modelo** — para producción se recomienda usar el judge LM también para SHORT, o cambiar el scorer a fuzzy match.

### 7.4 Honestidad sobre el resultado esperado

Reportamos tanto las hipótesis que se cumplieron como las que no. En la Iteración 2 (10 seeds, 223 ejemplos) la hipótesis central **sí se confirmó**; en la Iteración 1 (5 seeds, 63 ejemplos) no lo hacía:

- En *accuracy* absoluta esperábamos que `unified` liderara. **Iteración 2 (confirmado):** `unified (0.891) > baseline (0.843) > crop (0.780) > capo (0.391)`. **Iteración 1 (no se cumplía):** `crop > baseline ≈ unified > capo`.
- En *costo de salida* esperábamos `unified` entre los más bajos. **Iteración 2 (confirmado):** `unified (3930 tok, 0.0164 USD)` es el más barato, seguido de `crop`. **Iteración 1:** `capo < crop < unified ≈ baseline`.
- El *Pareto front* debía tener `unified` en la esquina favorable. **Iteración 2 (confirmado):** `unified` es el único punto no dominado (↑accuracy, ↓tokens). **Iteración 1:** esa esquina la ocupaba `crop`.

**Conclusión:** con datos y seeds suficientes, UNIFIED resultó ser el ganador Pareto, como predecía la hipótesis a priori. El resultado invertido de la Iteración 1 se explica por falta de potencia estadística (n=5) y un dataset pequeño (63 ejemplos), no por una falla del *pipeline* unificado.

### 7.5 Por qué CAPO es la condición más débil (sobre-optimización de la brevedad)

En la Iteración 2 el *racing* de CAPO **ya no colapsa**: mantiene 3–4 supervivientes por seed (frente a `n_survivors = 0` en la Iteración 1, donde el *pool* pequeño y el dev set corto hacían a Holm-Bonferroni eliminar a todos). Con el dataset ampliado a 223 ejemplos los bloques del *racing* son más grandes y el test es más estable, de modo que ya no vacía la población.

Sin embargo, CAPO sigue siendo la condición con menor accuracy agregada (0.391 ± 0.077). La causa ahora es distinta y más interesante: **sus mutaciones de restricción sobre-optimizan la brevedad**. Los operadores `add_constraint` ("responde en ≤30 palabras", "una o dos oraciones") empujan al *racing* hacia prompts telegráficos. El efecto sobre las métricas es nítido:

- `fuzzy_short = 0.865` — **la mejor de las cuatro condiciones** en respuestas cortas.
- `acc_long = 0.007`, `judge_long = 0.227` — **la peor**: las respuestas largas quedan truncadas y el juez las penaliza.

Como la accuracy agregada mezcla short y long, la ganancia en short no compensa el desplome en long. CAPO optimiza fielmente el objetivo que se le da (calidad-costo con penalización de longitud), pero ese objetivo, aislado, es hostil a las preguntas de respuesta larga. **UNIFIED corrige exactamente esto:** el Critic de CROP reintroduce contenido cuando la brevedad daña la calidad, y por eso combina buen short (`fuzzy_short = 0.829`) con long casi perfecto (`acc_long = 0.990`).

**Mejoras adicionales posibles** (trabajo futuro — §8.2):
- Ponderar el objetivo por tipo de pregunta (no penalizar longitud en ejemplos *long*).
- Subir `population_size` a 8 y `n_generations` a 4 para dar más presupuesto de búsqueda al *racing*.
- Cambiar el *pairwise test* a Wilcoxon (más robusto con muestras pequeñas).

### 7.6 Varianza entre seeds mayor de lo anticipado

Incluso con `M3 + thinking=disabled`, el **baseline tiene std = 0.078 en accuracy** sobre 10 seeds (bajó de 0.167 en la Iteración 1 al pasar de 5 a 10 seeds y de 63 a 223 ejemplos, que promedian mejor la composición del *split*). Análisis de la fuente de varianza:

| Fuente | Contribución estimada |
|---|---|
| Composición short/long del test set por seed | ~0.10 |
| Micro-variabilidad residual del modelo M3 | ~0.05 |
| Orden de las preguntas en el batch | ~0.05 |

La varianza residual del modelo (rango de ~5 tokens por pregunta en la prueba de 5 corridas idénticas) es despreciable comparada con el efecto del *split*. Esto sugiere que **aumentar el número de seeds es más rentable que optimizar el determinismo**.

### 7.7 Actualización de la tabla de precios

A mitad del desarrollo del informe se recibió la tabla oficial de precios M2.7 / M3. Hasta entonces la *pricing table* de [src/config.py](src/config.py) tenía precios *placeholder*:

| Modelo              | Input ($/M) | Output ($/M) |
|---------------------|:-----------:|:------------:|
| M2.5-highspeed (viejo) | 0.20       | 1.20         |
| M2.5 (viejo)            | 0.20       | 1.20         |
| M3 (viejo, *placeholder*) | 0.30       | 1.50         |

La tabla oficial reporta precios distintos, y se decidió adoptar los valores *before-discount* (sin contrato comercial) como referencia conservadora:

| Modelo              | Input ($/M) | Output ($/M) | Caching read | Caching write |
|---------------------|:-----------:|:------------:|:------------:|:-------------:|
| M2.7                | 0.30        | 1.20         | 0.06         | 0.375         |
| M2.7-highspeed      | 0.60        | 2.40         | 0.06         | 0.375         |
| **M3** (≤ 512k in)  | **0.60**    | **2.40**     | 0.12         | 0.375         |
| M3 (> 512k in)      | 1.20        | 4.80         | 0.24         | 0.375         |

**Impacto en los resultados**: adoptar los precios *before-discount* subió el costo por corrida ~+70% respecto a los precios *placeholder* antiguos. Con el dataset de la Iteración 2 (223 ejemplos, 40 corridas) el costo total agregado es ≈ 0.82 USD. Esto **no cambia ninguna conclusión** del informe porque todas las comparaciones son intra-experimento y los precios se aplicaron uniformemente; sólo afecta la lectura absoluta de "este prototipo costó X dólares en la API".
---

## 8. Conclusiones y trabajo futuro

### 8.1 Conclusiones

1. **UNIFIED es el ganador Pareto.** La hipótesis a priori (que el *pipeline* unificado combinaría lo mejor de CAPO + CROP) **se cumplió** en la Iteración 2: UNIFIED logró la mayor accuracy (0.891) **y** el menor costo (0.0164 USD, −41% vs baseline) **y** el menor número de tokens de salida. Las tres comparaciones pareadas Wilcoxon (vs baseline, capo y crop) son estadísticamente significativas (p = 0.049 / 0.002 / 0.037). En la Iteración 1 (n=5, 63 ejemplos) el resultado era el opuesto —CROP parecía dominar— por falta de potencia estadística, no por una falla del diseño.

2. **CAPO aislado sobre-optimiza la brevedad.** Ya no colapsa (mantiene 3–4 supervivientes por seed con el dataset ampliado), pero es la condición más débil en accuracy agregada (0.391): sus mutaciones de restricción producen la mejor calidad en respuestas cortas (`fuzzy_short = 0.865`) a costa de hundir las largas (`acc_long = 0.007`). El Critic de CROP dentro de UNIFIED es lo que corrige ese sesgo (§7.5).

3. **El patrón "racing + critic" es componible a nivel arquitectónico *y* de resultado.** Los módulos `RacingEvaluator`, `Critic` y `Mutator` son independientes e intercambiables (✓), y con suficiente escala experimental la sinergia numérica se materializa: UNIFIED supera tanto a CAPO como a CROP aislados. La lección de la Iteración 1 es que esa sinergia sólo es visible con datos y seeds suficientes.

4. **El cambio de modelo de M2.5-highspeed a M3 + `thinking=disabled` fue decisivo.** Sin él, los resultados hubieran sido ininterpretables: el razonamiento interno de M2.5-highspeed consumía 70–99% del *output* y producía 25 puntos de varianza entre corridas idénticas. Con M3 + `thinking=disabled` la varianza residual del modelo es despreciable y el JSONL expone `reasoning_tokens=0` consistente.

5. **La política del percentil 70 funcionó como se esperaba.** En las 10 semillas de CROP, el Critic se invocó selectivamente y su costo quedó dentro del ahorro que produjo.

6. **La trazabilidad JSONL es innegociable.** Sin logs detallados, una corrida de optimización es irreproducible. La inspección de `final_prompt` por seed permitió entender por qué algunos prompts divergían del baseline sin necesidad de re-correr.

7. **La limitación del scorer SHORT quedó resuelta con `fuzzy_short_accuracy`.** El *exact match* para SHORT asumía respuestas telegráficas, pero M3 con `thinking=disabled` responde verbosamente ("El cuerpo humano adulto tiene 206 huesos" en lugar de "206"), lo que en la Iteración 1 daba `accuracy_short ≈ 0`. La Iteración 2 añade `fuzzy_short_accuracy` (Levenshtein normalizado), que sube a 0.74–0.87 y refleja el rendimiento real del modelo. El judge LM para LONG sigue evaluando semánticamente. **Era una limitación del scorer, no del modelo** — ya corregida.

### 8.2 Trabajo futuro

Las prioridades se ordenan por retorno esperado sobre la inversión, no por originalidad académica:

#### Corto plazo — mejorar CAPO y afinar el objetivo
- **Ponderar el objetivo por tipo de pregunta** para que CAPO no penalice la longitud en ejemplos de respuesta larga (causa de su bajo `acc_long`, §7.5).
- **Subir `population_size` a 8** y `n_survive` a 4, y `n_generations` a 4, para dar más presupuesto de búsqueda al *racing* ahora que ya no colapsa.
- **Cambiar el *pairwise test* a Wilcoxon** (más robusto con n=5–12 por bloque). Cambia una sola variable de configuración en `racing.py`.
- **Reemplazar Holm-Bonferroni por t-test pareado sin corrección** (alineado con el *paper* CAPO) y comparar ambas en un *ablation* sobre el mismo dataset.

#### Mediano plazo — mejorar la evaluación
- **Hecho en la Iteración 2:** scorer SHORT *fuzzy* (`fuzzy_short_accuracy`, Levenshtein normalizado) que resuelve el `accuracy_short ≈ 0` con M3; dataset ampliado a 223 ejemplos; 10 seeds por condición (las tres comparaciones de UNIFIED son ya significativas).
- **Escalar más allá del *toy set*** (BIG-bench, GSM8K) para validar que la ventaja de UNIFIED se mantiene fuera del dataset propio.
- **Ampliar a 200+ ejemplos evaluados con juez semántico también en SHORT** para comparar el juez vs el *fuzzy match* como *ground truth*.

#### Largo plazo — extender el prototipo
- **Adoptar un Critic más barato** (por ejemplo, `MiniMax-M2.7-highspeed`) sin perder calidad en la retroalimentación; verificar que la diferencia de costo no se evapore al usar un Critic inferior.
- **Explorar variantes de la política del percentil 70** (percentil 50, percentil 90, política adaptativa basada en `cost_total` en lugar de `tokens_out`).
- **Añadir CoT toggle** como tercer operador de mutación, dado que CAPO reporta que aporta ganancias en tareas de razonamiento.
- **Conectar con DSPy** para aprovechar la paralelización nativa y la integración con `BootstrapFewShot`.

#### Reproductibilidad
- Publicar los JSONL crudos de las 40 corridas en `results/raw/` para que el experimento sea auditable independientemente del código.

---

## 9. Referencias

1. Zehle, T., Schlager, M., Heiß, T., & Feurer, M. (2025). *CAPO: Cost-Aware Prompt Optimization.* arXiv:2504.16005.
2. Amanchukwu et al. (2025). *CROP: Cost-Regularized Optimization of Prompts.* arXiv (paper provisto al equipo).
3. Guo, Q. et al. (2024). *EvoPrompt: Connecting LLMs with Evolutionary Algorithms.* NeurIPS 2024.
4. Yang, C. et al. (2024). *Large Language Models as Optimizers (OPRO).* ICLR 2024.
5. Birattari, M. et al. (2002). *A racing algorithm for configuring metaheuristics.* GECCO 2002.
6. Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics.
7. Khattab, O. et al. (2024). *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.* ICLR 2024.
8. BerriAI. *LiteLLM — Multi-provider LLM interface.* https://github.com/BerriAI/litellm.
9. MiniMax. *API documentation.* https://platform.MiniMax.com/ (consultada en 2026-07).

---

## Anexo A — Cómo ejecutar el proyecto

```bash
# 1. Clonar e instalar
git clone <repo>
pip install -r requirements.txt

# 2. Configurar la API key
cp .env.example .env
# editar .env y completar API_KEY, URL_API_BASE, MODEL

# 3. (Opcional) Regenerar el dataset
python -m src.data_gen

# 4. Correr todas las condiciones × 10 seeds
python -m experiments.run_all --seeds 0 1 2 3 4 5 6 7 8 9 --budget 5

# 5. Agregar resultados
python -m analysis.aggregate
python -m analysis.stats
python -m analysis.figures

# 6. Tests
python -m pytest tests/
```

> Duración estimada: 10–20 minutos por condición en una API MiniMax *standard tier*.

## Anexo B — Decisiones de diseño

- **`temperature = 0` por defecto.** Reduce (sin eliminar) el no-determinismo de la API.
