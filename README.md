# Análisis Comparativo de Modelos NLP para Sentimiento Financiero

Proyecto de TFG que compara modelos Transformer (BERT-family) y LLMs (Ollama + MLX) para análisis de sentimiento en noticias financieras. Evalúa precisión, eficiencia y capacidad de generalización sobre dos datasets: uno público de noticias financieras generales y uno propio de noticias de NVIDIA.

---

## Modelos evaluados

### BERT-family (fine-tuning supervisado)
| Modelo | HuggingFace ID |
|---|---|
| BERT | `ProsusAI/finbert` |
| FinBERT | `yiyanghkust/finbert-tone` |
| DistilRoBERTa | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` |
| DeBERTa v3 | `microsoft/deberta-v3-small` |
| FinancialBERT | `ahmedrachid/FinancialBERT-Sentiment-Analysis` |

### LLMs vía Ollama (few-shot / zero-shot)
| Modelo | Variante |
|---|---|
| Llama 3.1 | `llama3.1:8b` |
| Llama 3.2 | `llama3.2:latest` |
| Phi-3.5 | `phi3.5:latest` |

### Phi-3.5 vía MLX (Apple Silicon)
| Variante | Descripción |
|---|---|
| Base | Phi-3.5 sin fine-tuning |
| LoRA | Phi-3.5 fine-tuneado con LoRA sobre el dataset de entrenamiento |

---

## Datasets

**Dataset principal** — `prithvi1029/sentiment-analysis-for-financial-news` (HuggingFace)
- 4.846 titulares financieros en inglés
- Clases: `negative`, `neutral`, `positive`
- Usado para fine-tuning y evaluación de modelos BERT-family y MLX

**Dataset NVIDIA** — `data/raw/nvidia_testing.csv`
- 30 noticias sobre NVIDIA con etiqueta manual
- Usado para evaluar todos los modelos en zero-shot sobre un dominio específico

**Dataset NVIDIA real** — `data/raw/nvidia_real_news.csv`
- ~100 noticias reales de NVIDIA sin ground truth
- Usado para análisis exploratorio con prompt avanzado (chain-of-thought + relevance gate)

---

## Resultados principales

### Dataset principal (fine-tuning)
| Modelo | Accuracy | F1 weighted |
|---|---|---|
| FinancialBERT | 95.0% | — |
| DistilRoBERTa | 85.5% | — |
| Phi-3.5 MLX LoRA | 88.0% | — |
| Phi-3.5 MLX base | 85.0% | — |
| FinBERT | 84.0% | 0.813 |
| BERT (ProsusAI) | 84.5% | — |
| Phi-3.5 Ollama | 82.0% | — |
| Llama 3.1 | 79.0% | — |
| DeBERTa v3 | ~70% | — |
| Llama 3.2 | 41.0% | — |

### Dataset NVIDIA (zero-shot)
| Modelo | Accuracy |
|---|---|
| Phi-3.5 Ollama | 100% |
| Llama 3.1 | 100% |
| Llama 3.2 | 92.6% |
| Phi-3.5 MLX base | 93.3% |
| Phi-3.5 MLX LoRA | 93.3% |
| FinBERT | 83.3% |
| BERT (ProsusAI) | 80.0% |
| DistilRoBERTa | 80.0% |
| DeBERTa v3 | 80.0% |
| FinancialBERT | 56.7% |

---

## Estructura del proyecto

```
.
├── data/
│   ├── mlx/                    # Splits train/valid/test para MLX
│   └── raw/                    # CSVs de NVIDIA
├── models/
│   └── lora_checkpoints/       # Adaptadores LoRA entrenados
├── results/
│   ├── bert/{bert,finbert,distilroberta,deberta,financialbert}/
│   ├── mlx/{phi3.5-base,phi3.5-lora}/
│   └── ollama/{llama3.1,llama3.2,phi3.5,phi3.5-advanced}/
├── scripts/
│   ├── bert/                   # Script de entrenamiento BERT unificado
│   ├── mlx/                    # Entrenamiento y evaluación MLX
│   └── ollama/                 # Tests few-shot con Ollama
├── tests/
│   ├── BERT/
│   ├── FinBERT/
│   ├── DistilRoBERTa/
│   ├── DeBERTa v3/
│   ├── FinancialBERT/
│   ├── llama3.1/
│   ├── llama3.2/
│   ├── phi3.5/
│   ├── phi3.5-mlx/
│   └── phi3.5-advanced/        # Prompt avanzado con CoT + relevance gate
├── run_all_tests.py             # Ejecuta todos los tests (dataset principal)
├── run_all_NVIDIA_tests.py      # Ejecuta todos los tests (dataset NVIDIA)
└── requirements.txt
```

Cada carpeta en `tests/` contiene dos scripts:
- `test_<modelo>.py` — fine-tuning + evaluación sobre dataset principal
- `test_<modelo>_nvidia.py` — zero-shot sobre dataset NVIDIA

Cada ejecución genera en `results/`:
- `<modelo>_predictions.csv` — predicciones por muestra
- `<modelo>_metadata.json` — métricas, tiempos, matriz de confusión

---

## Instalación

Requiere Python 3.11 y un entorno virtual:

```bash
python3.11 -m venv ~/envs/sentimiento
source ~/envs/sentimiento/bin/activate
pip install -r requirements.txt
```

Para los modelos MLX (solo Apple Silicon):
```bash
pip install mlx mlx-lm
```

Para los modelos Ollama:
```bash
# Instalar Ollama desde https://ollama.com
ollama pull llama3.1
ollama pull llama3.2
ollama pull phi3.5
```

---

## Ejecución

```bash
# Todos los tests sobre dataset principal
python run_all_tests.py

# Todos los tests sobre dataset NVIDIA
python run_all_NVIDIA_tests.py

# Test individual (desde su directorio)
cd tests/FinBERT
python test_finbert.py
```

Los scripts se auto-relanzan con el venv correcto si se ejecutan con otro Python.

### Notas de hardware

- Los modelos BERT-family usan **MPS** (Apple Silicon) automáticamente
- DeBERTa v3 fuerza **CPU** por incompatibilidad con MPS (gradientes NaN)
- Los modelos MLX requieren Apple Silicon
- Ollama corre en CPU/GPU según disponibilidad

---

## Metodología

### BERT-family
Fine-tuning de 3 épocas sobre 800 muestras de entrenamiento, evaluación sobre 200. Métricas: accuracy, F1 weighted, ROC-AUC, log loss, matriz de confusión.

### LLMs (Ollama)
Few-shot con 3 ejemplos del dataset de entrenamiento. Temperatura 0 para reproducibilidad. Parsing de respuesta con fallback robusto.

### MLX
- **Base**: inferencia directa sin fine-tuning
- **LoRA**: fine-tuning con `mlx-lm` sobre los splits en `data/mlx/`, adaptadores guardados en `models/lora_checkpoints/`

### Prompt avanzado (phi3.5-advanced)
Chain-of-thought de 5 pasos con relevance gate explícito para noticias NVIDIA reales. Devuelve `sentiment`, `intensity`, `relevance` y `reasoning` por noticia.

---

## Dependencias principales

```
transformers    — modelos BERT-family
datasets        — carga de datasets HuggingFace
torch           — backend PyTorch (MPS/CPU)
scikit-learn    — métricas de evaluación
scipy           — softmax y utilidades
mlx / mlx-lm   — inferencia y fine-tuning en Apple Silicon
ollama          — cliente Python para Ollama
psutil          — monitoreo de memoria
```
