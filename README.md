# CAPO × CROP — pipeline unificado de optimización de prompts

Prototipo de investigación que integra dos métodos conscientes del costo:

- **CAPO** (*Cost-Aware Prompt Optimization* — Zehle et al., 2025): optimiza el costo de entrada con un algoritmo de *racing* y Holm-Bonferroni.
- **CROP** (*Cost-Regularized Optimization of Prompts* — Amanchukwu et al., 2025): optimiza el costo de salida con un Critic LM que produce *feedback* textual de brevedad.

El aporte propio es un *pipeline* declarativo-iterativo único que produce prompts Pareto-óptimos sobre `(accuracy, cost_in, cost_out)`.

---

## 1. Quickstart (5–10 minutos)

```bash
# 1. Clonar e instalar dependencias
git clone <repo>
pip install -r requirements.txt

# 2. Configurar credenciales
cp .env.example .env
# editar .env y completar:
#   API_KEY=<tu-key>
#   URL_API_BASE=https://api.minimax.io/v1
#   MODEL=MiniMax-M2.5-highspeed

# 3. (Opcional) Regenerar el dataset toy (223 ejemplos QA en español)
python -m src.data_gen

# 4. Correr las 4 condiciones × 10 seeds
python -m experiments.run_all --seeds 0 1 2 3 4 5 6 7 8 9 --budget 5

# 5. Agregar, analizar y graficar
python -m analysis.aggregate
python -m analysis.stats
python -m analysis.figures

# 6. Tests
python -m pytest tests/ -v
```

Si solo quieres probar la integración sin gastar API, todos los *scripts* admiten `--condition <nombre>` para correr una condición puntual.

---

## 2. Decisiones de diseño

| Decisión | Por qué |
|---|---|
| **Sin MLflow / W&B / DVC** | Para 1 semana la infraestructura mata. JSONL + CSV basta. |
| **Dataset toy propio (223 ejemplos)** | Control de variabilidad, *scoring* determinístico, cero dependencias de HuggingFace. Iteración 2 amplió de 63 → 223 filas para alcanzar potencia estadística. |
| **MiniMax como objetivo Y Critic** | Una sola API key, una sola cuenta de costos. Configurable vía `MODEL`. |
| **2 operadores de mutación** (`paraphrase`, `add_constraint`) | Cubre los efectos más reportados; `swap_fewshot` queda como opcional. |
| **Wilcoxon + Holm-Bonferroni** | Wilcoxon es robusto con muestras pequeñas (5–12 ítems por bloque). Holm protege contra eliminación prematura. La corrección es configurable (`--correction holm\|none\|bonferroni`) para reproducir el paper CAPO. |
| **Política percentil 70 del Critic** | Invoca al Critic solo cuando la salida supera el P70 del *pool* (sigue a CROP). |
| **`temperature = 0`** por defecto | Reduce (sin eliminar) el no-determinismo de la API. |
| **Mock LLM para tests** | `_enable_mock()` activa respuestas deterministas; los 33 *smoke tests* pasan sin red. |
| **Fuzzy match para SHORT** | `rapidfuzz.fuzz.token_set_ratio` reconoce respuestas que contienen el token esperado aunque estén envueltas en prosa (p. ej. "El cuerpo humano adulto tiene 206 huesos" → esperado "206"). Umbral por defecto: 0.85. |

---

## 3. Estructura del repositorio

```
capo-crop-unified/
├── README.md                 ← este archivo
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── toy_qa.jsonl          ← 223 ejemplos QA en español
├── src/
│   ├── config.py             ← carga .env y expone Settings
│   ├── data_gen.py           ← genera el dataset toy
│   ├── llm_client.py         ← wrapper LiteLLM + retries + JSONL logging
│   ├── cost_model.py         ← tokens → USD
│   ├── mutator.py            ← paraphrase, add_constraint, swap_fewshot
│   ├── racing.py             ← RacingEvaluator + Holm-Bonferroni
│   ├── scorer.py             ← MultiObjectiveScorer + LLM-as-judge
│   ├── critic.py             ← BrevityFeedbackGenerator (Critic LM)
│   ├── pipeline.py           ← run_baseline / run_capo / run_crop / run_unified
│   └── utils/
│       ├── seeds.py          ← set_seed reproducible
│       ├── stats.py          ← Wilcoxon, bootstrap, Cohen's d
│       └── logging.py        ← JSONLLogger
├── experiments/
│   ├── run_baseline.py
│   ├── run_capo.py
│   ├── run_crop.py
│   ├── run_unified.py
│   └── run_all.py            ← corre las 4 condiciones × N seeds
├── analysis/
│   ├── aggregate.py          ← consolida JSONL → CSV
│   ├── stats.py              ← Wilcoxon pareado + bootstrap CI
│   └── figures.py            ← genera las 3 figuras
├── tests/
│   ├── test_racing.py        ← invariantes Holm + no-false-elimination
│   ├── test_cost_model.py
│   ├── test_pareto.py
│   ├── test_scorer.py
│   └── test_pipeline.py      ← smoke tests con LLM mock
├── reports/
│   ├── informe.md            ← fuente markdown
│   └── informe.pdf           ← versión PDF
├── presentation/
│   └── slides.md             ← 10 slides para defensa
└── results/
    ├── raw/                  ← JSONL por condición/seed
    ├── tables/               ← CSV (summary, long, stats)
    └── figures/              ← PNG de las 3 figuras
```

---

## 4. Hiperparámetros por defecto

| Parámetro | Valor | Origen |
|---|---|---|
| `alpha` (Holm) | 0.2 | Default CAPO |
| `gamma` (long. CAPO) | 0.05 | Default CAPO |
| `block_size` | 3–30 | Adaptado a `len(dev)` |
| `z_max` | 2–10 | Deriva de `len(dev) // block_size` |
| `population_size` | 8 | antes 4 |
| `crossovers_per_iter` | 3 | Suficiente para 4 supervivientes |
| `n_survive` | 4 | Mitad de la población (mínimo 2) |
| `n_generations` | 4 | antes 2 |
| `pairwise_test` | `wilcoxon` | antes `ttest`. Más robusto con n pequeño. |
| `correction` | `holm` | Holm-Bonferroni (default). `none` reproduce CAPO paper, `bonferroni` es single-step. |
| `fuzzy_threshold` | 0.85 | `rapidfuzz.token_set_ratio` mínimo para considerar `correct` en SHORT. |
| `beta` (long. CROP) | 0.05 | Default CROP |

Todos los hiperparámetros son configurables desde `config.py` o como flags de los *scripts* de `experiments/`.

---

## 5. Cómo correr una condición puntual

```bash
# Baseline
python -m experiments.run_baseline --seed 0 --budget 5

# CAPO (defaults: generations=4, population=8, wilcoxon, holm)
python -m experiments.run_capo --seed 0 --budget 5 --generations 4 --population 8

# CROP
python -m experiments.run_crop --seed 0 --budget 5 --iterations 2

# Unified
python -m experiments.run_unified --seed 0 --budget 5 --generations 4 --population 8

# Ablación Holm vs t-test pareado sin corrección (paper CAPO)
python -m experiments.run_capo --seed 0 --correction none --pairwise-test ttest
```

Cada *script* escribe:

- `results/raw/<condición>/seed<N>.jsonl` — log JSONL con cada llamada al LLM.
- `results/raw/<condición>/seed<N>.json` — resumen de métricas (accuracy, costo, prompt final, `pairwise_test`, `correction`).

---

## 6. Cómo regenerar entregables

```bash
# CSV consolidados
python -m analysis.aggregate         # → results/tables/summary.csv, long.csv
python -m analysis.stats             # → results/tables/stats.csv

# Figuras
python -m analysis.figures           # → results/figures/figure{1,2,3}_*.png

# PDF del informe (requiere pandoc + Chrome)
pandoc reports/informe.md -o reports/informe.html --standalone
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless --disable-gpu --no-sandbox \
    --print-to-pdf=reports/informe.pdf reports/informe.html
```

---

## 7. Limitaciones conocidas

- **API no determinista.** `temperature = 0` no garantiza igualdad bit-a-bit. Los números pueden variar entre corridas; el Wilcoxon pareado mitiga esto.
- **Dataset toy.** 223 ejemplos no captura la complejidad de BIG-bench ni GSM8K. Las conclusiones se re-evaluarán a mayor escala en el siguiente hito.
- **Default de Holm-Bonferroni diverge del paper CAPO.** El paper usa t-test pareado sin corrección. El default actual sigue siendo Holm (más conservador) pero el flag `--correction none` reproduce exactamente el comportamiento del paper.
- **10 *seeds* por defecto.** Suficiente para potencia razonable en Wilcoxon pareado (con n=5 era de ≈0.4 para efectos d≈0.4). Las corridas individuales pueden usar `--seeds N` para reducir si el tiempo aprieta.
- **Default apunta a `api.minimax.io`.** Si tu contrato está en `api.minimaxi.com` (plan legado), override `URL_API_BASE` en `.env`.
- **Reasoning tokens de MiniMax-M2.5-highspeed.** El modelo emite tokens de razonamiento interno en **70–99%** de la salida facturada, incluso para prompts triviales. Los parámetros `thinking: {type: disabled}` (M3), `reasoning: {enabled: false}` y `reasoning_effort: 0` **no surten efecto** en M2.5. `LLMResponse` expone `reasoning_tokens` por separado para que el *pipeline* y `cost_model.py` puedan desglosar el costo si lo desean.
- **Ablación Holm vs t-test ejecutable vía CLI.** `--correction holm|none|bonferroni` + `--pairwise-test ttest|wilcoxon` cubre la matriz principal; documentado en `reports/informe.md` §8.2 (corto plazo).

---

## 8. Tests

```bash
python -m pytest tests/ -v
```

Resultado esperado: **33/33 pasan**. Cubre Holm-Bonferroni, Wilcoxon, las tres correcciones (holm/none/bonferroni), modelo de costos, dominancia de Pareto, fuzzy scorer SHORT, *scoring* LONG, *pipeline* end-to-end con LLM *mock*.

Si un test falla, el mensaje indica exactamente qué invariante se rompió — útil para diagnosticar cambios accidentales en el *racing* o el *scoring*.

---

## 9. Smoke test de la API (día 1)

Antes de correr cualquier experimento, valida que el *string* del modelo es correcto:

```bash
python -c "
from src.llm_client import LLMClient
client = LLMClient()
resp = client.complete('Di hola en una sola palabra.', temperature=0.0, role='smoke')
print('model:', resp.model, 'tokens_in:', resp.tokens_in, 'tokens_out:', resp.tokens_out, 'latency_ms:', resp.latency_ms)
print('text:', resp.text)
"
```

Si el *output* muestra números razonables (`tokens_in > 0`, `latency_ms > 0`), el *endpoint* está operativo. Si no, revisa `URL_API_BASE` y `MODEL` en `.env`.

---

## 10. Referencias

1. Zehle, T., Schlager, M., Heiß, T., & Feurer, M. (2025). *CAPO: Cost-Aware Prompt Optimization.* arXiv:2504.16005.
2. Amanchukwu et al. (2025). *CROP: Cost-Regularized Optimization of Prompts.*
3. Guo, Q. et al. (2024). *EvoPrompt.* NeurIPS 2024.
4. Birattari, M. et al. (2002). *F-Race.* GECCO 2002.
5. Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scand. J. Stat.
6. Khattab, O. et al. (2024). *DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines.* ICLR 2024.
7. BerriAI. *LiteLLM.* https://github.com/BerriAI/litellm
8. MiniMax. *API documentation.* https://platform.MiniMax.com/

---
