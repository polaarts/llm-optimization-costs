# CAPO × CROP — Presentación

> 10 slides para defensa del prototipo. Markdown listo para convertir a Reveal.js, Marp o `pandoc -t beamer`.

---

## Slide 1 — Título

**CAPO × CROP: pipeline unificado para optimización de prompts consciente del costo**

- Autor: Samuel Angulo
- Tipo: prototipo de investigación (1 semana)
- Stack: Python · LiteLLM · MiniMax API
- Repositorio: `capo-crop-unified/`

---

## Slide 2 — Problema

- La calidad de un prompt determina la calidad de la respuesta de un LLM.
- Optimizar prompts manualmente es caro, lento y no escala.
- Los métodos automáticos existentes **ignoran el costo** de la API o lo abordan de forma parcial.
- CAPO optimiza **costo de entrada**; CROP optimiza **costo de salida**.
- **Pregunta:** ¿se pueden combinar en un solo *pipeline*?

---

## Slide 3 — CAPO en 1 minuto

- *Cost-Aware Prompt Optimization* (Zehle et al., 2025).
- Algoritmo evolutivo (GA) sobre instrucciones y *few-shots*.
- **Racing con Holm-Bonferroni** descarta perdedores en bloques mini-batch crecientes → ahorra ~44 % de las evaluaciones.
- Penalización por longitud de prompt γ = 0.05.
- Reporta hasta +21 %p de accuracy vs. EvoPromptGA, OPRO, PromptWizard en 11/15 configs.

---

## Slide 4 — CROP en 1 minuto

- *Cost-Regularized Optimization of Prompts* (Amanchukwu et al., 2025).
- **Critic LM** λ-regularizado que produce *feedback* textual + reescritura breve.
- Política de invocación: solo se llama al Critic si la salida supera el **percentil 70** del *pool*.
- Ataca directamente el costo de salida, donde CAPO no llega.

---

## Slide 5 — Nuestra propuesta

```
     Seed prompts
         │
   PromptMutator (paraphrase | add_constraint)
         │
     Candidates
         │
   RacingEvaluator  ◄──  Holm-Bonferroni
         │                    ▲
   MultiObjectiveScorer ─────┘
         │
   BrevityFeedbackGenerator (sólo si cost_out > P70)
         │
     Final prompt Pareto-óptimo
```

Aporte: un solo *backbone* declarativo que selecciona supervivientes con CAPO y aplica *feedback* de brevedad con CROP.

---

## Slide 6 — Setup experimental

- **Dataset toy**: 63 ejemplos QA en español (30 cortos, 33 largos), 3 niveles de dificultad.
- **Modelo**: MiniMax API (mismo modelo para objetivo y Critic).
- **Condiciones**: `baseline` · `capo` · `crop` · `unified`.
- **Seeds**: 10 (`0…9`) — Iteración 2 amplió de 3→10 seeds y el dataset de 63→223 ejemplos.
- **Presupuesto**: 5 USD por condición (gasto real agregado ≈ 0.82 USD en las 40 corridas).
- **Hiperparámetros**: α=0.2, γ=0.05, block_size=3–10, generations=2, population=4.
- **Tests**: 24/24 pasan (`pytest tests/`).

---

## Slide 7 — Resultados: tabla

> **Resultados vigentes (Iteración 2 — 10 seeds × 223 ejemplos).**

| Condición  | Accuracy (mean ± std) | Costo USD (mean ± std) | Estado |
|------------|:--------------------:|:----------------------:|:-------|
| baseline   | 0.843 ± 0.078        | 0.02755 ± 0.00284      | referencia fuerte |
| capo       | 0.391 ± 0.077        | 0.02069 ± 0.00757      | sobre-optimiza brevedad |
| crop       | 0.780 ± 0.154        | 0.01730 ± 0.00357      | buen equilibrio |
| **unified**| **0.891 ± 0.057**    | **0.01638 ± 0.00487**  | **ganador Pareto** |

**UNIFIED gana en las dos dimensiones**: mayor accuracy *y* menor costo. Wilcoxon pareado (n=10): unified > baseline (p=0.049), > capo (p=0.002), > crop (p=0.037) — las tres significativas.

> La hipótesis a priori (`unified` combina lo mejor de CAPO + CROP y domina el frente Pareto) **se confirma** en esta iteración. En la Iteración 1 (5 seeds, 63 ej.) no se cumplía y CROP quedaba como mejor punto Pareto.

---

## Slide 8 — Resultados: frente Pareto

- **Figura 2** del informe: dispersión `(tokens_out, accuracy)` coloreada por condición.
- Los puntos que dominan a los demás forman el *frente*.
- `unified` se ubica en la esquina dominante (↑accuracy = 0.891, ↓tokens_out = 3930): **es el único punto no dominado**.
- `crop` queda cerca en costo (tokens_out ≈ 4030) pero con menor accuracy (0.780).
- `capo` cae abajo (accuracy 0.391) pese a pocos tokens; `baseline` queda a la derecha (tokens_out ≈ 9050, accuracy alta 0.843).

---

## Slide 9 — Discusión y limitaciones

**Lo que aprendimos**

- El *racing* con Holm-Bonferroni es más conservador que el t-test pareado del *paper* original; en Holm se eliminan menos candidatos por bloque.
- El desacople Racing ↔ Critic permite intercambiarlos sin reescribir nada.
- La política del percentil 70 evita que el Critic gaste más de lo que ahorra.

**Limitaciones honestas**

- *Dataset* toy de 223 ejemplos, 10 *seeds* → potencia razonable (Wilcoxon significativo), pero aún es un *toy set* (no BIG-bench/GSM8K).
- CAPO sobre-optimiza la brevedad: gana en respuestas cortas (`fuzzy_short = 0.865`) pero destruye las largas (`acc_long ≈ 0`).
- API no determinista (`temperature = 0` no garantiza igualdad bit-a-bit).
- Sin *ablation* exhaustiva (`capo-sin-Racing`, `crop-sin-Critic`).
- Precios M3 confirmados (before-discount 0.60/2.40 $/M); costo agregado de las 40 corridas ≈ 0.82 USD.

---

## Slide 10 — Conclusiones + repo

**Conclusiones**

1. **UNIFIED es el ganador Pareto** (Iteración 2): mayor accuracy (0.891) *y* menor costo (0.0164 USD), significativo vs. las tres condiciones. CAPO y CROP **son componibles** y su combinación supera a cada parte aislada.
2. CAPO aislado sobre-optimiza la brevedad: gana en respuestas cortas pero colapsa en las largas. El Critic de CROP dentro de UNIFIED corrige ese sesgo.
3. La política del percentil 70 es **indispensable** para que el Critic pague su costo.
4. Los logs JSONL como única fuente de verdad hacen el experimento **reproducible** sin un tracker externo.

**Trabajo futuro**

- Reemplazar Holm por la t-test pareada del *paper* CAPO y comparar.
- Probar un Critic más barato (`MiniMax-M2.7-highspeed`).
- Escalar más allá del *toy set* (BIG-bench, GSM8K) para validar la ventaja de UNIFIED.
- Integrar con DSPy para paralelizar el *racing*.

**Cómo correr en 10 minutos**

```bash
git clone <repo> && cd capo-crop-unified
pip install -r requirements.txt
cp .env.example .env       # editar y agregar API_KEY
python -m experiments.run_all --seeds 0 1 2 3 4 5 6 7 8 9
python -m analysis.aggregate && python -m analysis.stats && python -m analysis.figures
```

¡Gracias! ¿Preguntas?
