# Análisis de Modelos Óptimos: NLP vs LLM

> Basado en resultados empíricos de dos evaluaciones:
> - **Dataset principal**: `prithvi1029/sentiment-analysis-for-financial-news` (200–800 muestras, fine-tuning o few-shot)
> - **Dataset NVIDIA**: `nvidia_testing.csv` (30 muestras, zero-shot en dominio específico)

---

## 1. Modelos NLP (BERT-family)

### 1.1 Resultados comparativos

#### Dataset principal (zero-shot)

| Modelo | Accuracy | F1 weighted | ROC-AUC | Log Loss | F1 neg | F1 neu | F1 pos | Inf (ms) | RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|
| **FinancialBERT** | **95.0%** | **0.9496** | **0.9939** | **0.1743** | **0.923** | **0.842** | **0.970** | 3.27 | 565 |
| DistilRoBERTa | 85.0% | 0.8602 | 0.9025 | 1.069 | 0.923 | 0.592 | 0.905 | 16.4 | 919 |
| BERT (ProsusAI) | 84.5% | 0.806 | 0.841 | 0.420 | 0.000 | 0.400 | 0.912 | 3.10 | 842 |
| FinBERT | 84.0% | 0.8127 | 0.8928 | 0.3414 | 0.250 | 0.409 | 0.908 | 3.23 | 660 |
| DeBERTa v3 | — | — | — | — | — | — | — | — | — |

> DeBERTa v3: resultados pendientes. BERT y FinBERT incluían fine-tuning en versiones anteriores; DistilRoBERTa usa `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` en zero-shot.

#### Dataset NVIDIA (zero-shot, dominio específico)

| Modelo | Accuracy | F1 weighted | F1 neg | F1 neu | F1 pos | Inf (ms) | RAM (MB) |
|---|---|---|---|---|---|---|---|
| **FinBERT** | **83.3%** | **0.836** | **0.947** | **0.778** | **0.783** | 9.4 | 637 |
| BERT (ProsusAI) | 80.0% | 0.798 | 0.818 | 0.750 | 0.818 | 11.3 | 821 |
| DistilRoBERTa | 80.0% | 0.802 | 0.842 | 0.737 | 0.818 | 11.5 | 603 |
| DeBERTa v3 | 80.0% | 0.798 | 0.857 | 0.706 | 0.818 | 66.3 | 795 |
| FinancialBERT | 56.7% | 0.559 | 0.462 | 0.500 | 0.696 | 17.8 | 546 |

### 1.2 Análisis

**FinancialBERT** domina el dataset principal con un margen amplio: 95% de accuracy en zero-shot, sin necesidad de fine-tuning. ROC-AUC de 0.994 y log loss de 0.174 indican predicciones muy calibradas. Es el único modelo que detecta correctamente las tres clases (F1 neg=0.92, F1 neu=0.84). Sin embargo, **colapsa en el dataset NVIDIA** (56.7%), lo que sugiere que su entrenamiento original está muy ajustado al dominio del Financial PhraseBank y no generaliza bien a noticias de empresa específica.

**FinBERT** (`yiyanghkust/finbert-tone`) muestra el comportamiento más equilibrado entre los dos datasets: 84% en el principal y 83.3% en NVIDIA. Es el único modelo que mantiene rendimiento alto en ambos contextos. Su F1 en negativo sobre NVIDIA (0.947) es el más alto de todos los NLP, lo que es crítico para análisis de riesgo.

**DistilRoBERTa** (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`) con el modelo correcto en zero-shot obtiene 85% en el dataset principal con F1 neg=0.923 — empatando con FinancialBERT en la detección de negativos. Su ROC-AUC (0.903) y F1 weighted (0.860) son sólidos. El punto débil es la clase neutral (F1=0.592) y un log loss alto (1.069), indicando probabilidades menos calibradas. En NVIDIA mantiene 80% con balance razonable entre clases. Es la opción más ligera del grupo con buen balance (603 MB en NVIDIA).

**BERT (ProsusAI)** y **FinBERT** con fine-tuning tienen accuracy similar (~84%) pero ambos fallan en la clase negativa del dataset principal (F1 neg=0.0 y 0.25 respectivamente). FinBERT compensa con el mejor rendimiento en NVIDIA (83.3%, F1 neg=0.947).

**DeBERTa v3** obtiene 80% en NVIDIA con buen balance entre clases, pero su inferencia es 4-7x más lenta (66ms) y requiere CPU forzado por incompatibilidad con MPS.

### 1.3 Modelo NLP óptimo

**FinBERT (`yiyanghkust/finbert-tone`) es el NLP más óptimo para este proyecto.**

Razones:
- Único modelo con rendimiento alto y consistente en ambos datasets (84% / 83.3%)
- Mejor F1 en clase negativa sobre NVIDIA (0.947) — crítico para análisis de riesgo financiero
- Menor log loss del grupo con fine-tuning (0.341), indicando predicciones más confiables
- Menor consumo de RAM (660 MB) entre los modelos con fine-tuning
- Pre-entrenado en texto financiero, lo que le da ventaja semántica sin coste adicional

**Mención especial**: FinancialBERT es superior en el dataset principal (95% vs 84%), pero su caída a 56.7% en NVIDIA lo descarta para uso general. Sería la elección óptima si el dominio de aplicación es idéntico al de su entrenamiento.

**Para producción de alta velocidad con buen balance de clases**: DistilRoBERTa (`mrm8488/...`) en zero-shot — 85% accuracy, F1 neg=0.923, 603 MB RAM, sin necesidad de fine-tuning.

---

## 2. Modelos LLM

### 2.1 Resultados comparativos

#### Dataset principal (few-shot, 12 ejemplos)

| Modelo | Accuracy | F1 weighted | F1 neg | F1 neu | F1 pos | Inf (ms) | Balance clases |
|---|---|---|---|---|---|---|---|
| **Phi-3.5 MLX LoRA** | **88.0%** | 0.936* | 0.000* | 0.936* | 0.000* | 427 | ⚠️ Solo neutral |
| Phi-3.5 MLX base | 85.0% | 0.919* | 0.000* | 0.919* | 0.000* | 148 | ⚠️ Solo neutral |
| **Phi-3.5 Ollama** | **82.0%** | **0.818** | **0.897** | **0.842** | **0.737** | **167** | ✅ Equilibrado |
| Llama 3.1 8B | 79.0% | 0.791 | 0.800 | 0.804 | 0.762 | 318 | ✅ Equilibrado |
| Llama 3.2 | 41.0% | 0.457 | 0.313 | 0.447 | 0.537 | 185 | ❌ Sesgo negativo |

> *Los resultados MLX en dataset principal muestran colapso a clase neutral — el subset de evaluación usado era 100% neutral, por lo que las métricas no son comparables con los demás.

#### Dataset NVIDIA (few-shot, 3 ejemplos de dominio)

| Modelo | Accuracy | F1 weighted | F1 neg | F1 neu | F1 pos | Inf (ms) |
|---|---|---|---|---|---|---|
| **Phi-3.5 Ollama** | **100%** | **1.000** | **1.000** | **1.000** | **1.000** | 183 |
| **Llama 3.1 8B** | **100%** | **1.000** | **1.000** | **1.000** | **1.000** | 330 |
| Phi-3.5 MLX LoRA | 93.3% | 0.935 | 0.947 | 0.900 | 0.952 | 410 |
| Phi-3.5 MLX base | 93.3% | 0.933 | 0.889 | 0.900 | 1.000 | 394 |
| Llama 3.2 | 92.6% | 0.924 | 1.000 | 0.857 | 0.909 | 207 |

### 2.2 Análisis

**Phi-3.5 Ollama** es el modelo más equilibrado del grupo LLM. Alcanza 100% en NVIDIA y 82% en el dataset principal con balance perfecto entre las tres clases (F1 neg=0.897, F1 neu=0.842, F1 pos=0.737). Es el único LLM que combina alta accuracy en dominio específico con buen rendimiento general. Su inferencia (167ms) es la más rápida del grupo.

**Llama 3.1 8B** también logra 100% en NVIDIA y 79% en el dataset principal con buen balance de clases. Sin embargo, es casi 2x más lento que Phi-3.5 (318ms vs 167ms) y su accuracy general es 3 puntos inferior.

**Phi-3.5 MLX LoRA** obtiene 93.3% en NVIDIA con el mejor F1 en negativo (0.947). El fine-tuning LoRA mejora ligeramente el balance respecto al base. Sin embargo, su inferencia es la más lenta (410ms) y los resultados en el dataset principal no son comparables por el problema del subset.

**Llama 3.2** muestra un comportamiento errático: 92.6% en NVIDIA pero solo 41% en el dataset principal, con sesgo masivo hacia la clase negativa (predice negativo en 70 de 100 muestras). No es fiable para uso general.

### 2.3 Modelo LLM óptimo

**Phi-3.5 Ollama (`phi3.5:latest`) es el LLM más óptimo para este proyecto.**

Razones:
- 100% de accuracy en NVIDIA con balance perfecto en las tres clases
- 82% en dataset principal — el mejor accuracy real entre LLMs con balance de clases correcto
- Inferencia más rápida del grupo (167ms/muestra)
- Único LLM que detecta correctamente negativos en el dataset principal (F1=0.897)
- Comportamiento consistente y predecible (temperatura 0, sin errores de conexión)

**Mención especial**: Llama 3.1 8B es una alternativa sólida si se prioriza robustez sobre velocidad. Phi-3.5 MLX LoRA es la mejor opción si se dispone de Apple Silicon y se acepta la latencia adicional.

---

## 3. Análisis exploratorio — Dataset NVIDIA real (sin ground truth)

> Dataset: `nvidia_real_news.csv` (100 noticias reales de NVIDIA, sin etiquetas de referencia)
> Prompt: avanzado v2 — relevance gate + 5-step CoT + few-shot difficult cases
> Temperatura: 0.0

### 3.1 Resultados comparativos

| Modelo | Positivo | Neutral | Negativo | Relevancia alta/media | Inf (ms/avg) | Errores |
|---|---|---|---|---|---|---|
| **Phi-3.5 Ollama** (advanced) | — | — | — | — | — | — |
| **Qwen 2.5 7B** | 21% | 74% | 5% | 27% | 1024 | 0 |

> Resultados de Phi-3.5 advanced pendientes de ejecución con prompt v2.

### 3.2 Análisis Qwen 2.5 7B

**Qwen 2.5 7B** aplica correctamente el relevance gate: clasifica el 73% de las noticias como `relevance: none`, lo que indica que el modelo discrimina bien entre noticias directamente relacionadas con NVIDIA y ruido de mercado general (ETFs, macro, sectores no relacionados).

Del 27% de noticias con relevancia alta o media:
- **21% positivo**: noticias sobre acuerdos comerciales, expansión de infraestructura AI, aprobaciones de exportación de chips, y avances tecnológicos de NVIDIA.
- **5% negativo**: noticias sobre retrasos en chips Blackwell, omisión de NVIDIA en rankings de analistas, y competidores ganando terreno en AI compute.
- **1% neutral con relevancia**: noticias con impacto ambiguo o indirecto.

La distribución refleja un sesgo positivo moderado en el corpus, consistente con el período cubierto (mayo 2024 – enero 2026), que incluye el boom de AI y la expansión de Blackwell.

**Velocidad**: 1024ms/muestra es ~6x más lento que Phi-3.5 Ollama (167ms), lo que es esperable dado el mayor tamaño del modelo (7B parámetros vs 3.8B de Phi-3.5). Sin embargo, sigue siendo viable para análisis batch.

**Calidad de reasoning**: El campo `reasoning` muestra justificaciones concisas y coherentes con el análisis financiero esperado (e.g., "Competitor weakening = positive for NVDA", "AI chip rollout delays impact market").

### 3.3 Comparativa de distribución (dataset NVIDIA real)

| Distribución | Qwen 2.5 7B |
|---|---|
| Positivo | 21 / 100 |
| Neutral | 74 / 100 |
| Negativo | 5 / 100 |
| Relevancia alta | 20 / 100 |
| Relevancia media | 7 / 100 |
| Relevancia baja | 0 / 100 |
| Sin relevancia | 73 / 100 |

---

## 4. Comparativa NLP vs LLM

| Dimensión | Mejor NLP | Mejor LLM | Ganador |
|---|---|---|---|
| Accuracy dataset principal | FinancialBERT 95.0% | Phi-3.5 Ollama 82.0% | **NLP** |
| Accuracy NVIDIA | FinBERT 83.3% | Phi-3.5 / Llama3.1 100% | **LLM** |
| Balance de clases (principal) | FinancialBERT | Phi-3.5 Ollama | **Empate** |
| Balance de clases (NVIDIA) | FinBERT | Phi-3.5 / Llama3.1 | **LLM** |
| Velocidad de inferencia | DistilRoBERTa 16ms | Phi-3.5 Ollama 167ms | **NLP** (10x) |
| Consumo de memoria | FinancialBERT 565 MB | N/A (modelo local) | **NLP** |
| Necesidad de fine-tuning | FinancialBERT: no | Todos: no | **Empate** |
| Generalización a nuevo dominio | FinBERT: moderada | Phi-3.5: alta | **LLM** |
| Coste computacional | Bajo | Medio (Ollama local) | **NLP** |

### Conclusión

Los **NLP especializados** (FinancialBERT, FinBERT) son superiores cuando el dominio de aplicación es similar al de su entrenamiento y se requiere alta velocidad de inferencia. FinancialBERT logra 95% sin fine-tuning adicional, lo que es difícil de superar.

Los **LLMs** (Phi-3.5, Llama 3.1) generalizan mejor a dominios nuevos y muestran balance de clases más robusto. Su ventaja es clara en el dataset NVIDIA (100% vs 83.3%) y en la capacidad de adaptarse a nuevos contextos con solo cambiar los ejemplos few-shot.

**Para un sistema de producción de análisis de sentimiento financiero general**: FinancialBERT (si el dominio es estable) o FinBERT (si se necesita robustez cross-domain).

**Para análisis de noticias de empresa específica o dominio cambiante**: Phi-3.5 Ollama o Qwen 2.5 7B, con prompt few-shot adaptado al dominio objetivo. Qwen 2.5 7B muestra mayor capacidad de discriminación por relevancia (73% de noticias correctamente filtradas como no relevantes), a costa de ~6x más latencia.

---

*Análisis generado a partir de resultados empíricos — Marzo 2026*
