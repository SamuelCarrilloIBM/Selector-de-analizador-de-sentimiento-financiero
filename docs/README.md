# 🤖 Análisis Comparativo de Modelos NLP para Sentimiento Financiero

Proyecto de comparación de tres modelos Transformer (BERT, DistilBERT y FinBERT) para análisis de sentimiento en textos financieros.

## 📋 Descripción

Este proyecto evalúa y compara el rendimiento de tres modelos de procesamiento de lenguaje natural:
- **BERT** (bert-base-uncased) - Modelo generalista
- **DistilBERT** (distilbert-base-uncased) - Versión optimizada de BERT
- **FinBERT** (yiyanghkust/finbert-tone) - Modelo especializado en textos financieros

El análisis incluye métricas de precisión (Accuracy, F1-Score, ROC-AUC) y eficiencia (tiempo de entrenamiento, inferencia y uso de memoria).

## 📁 Estructura del Proyecto

```
testing NLP/
├── main.py                    # Script principal de entrenamiento y evaluación
├── requirements.txt           # Dependencias del proyecto
├── ANALISIS_RESULTADOS.md     # Análisis detallado de resultados
├── README.md                  # Este archivo
├── output_results.txt         # Resultados de la ejecución
└── results_*/                 # Modelos entrenados y checkpoints
```

## 🔧 Requisitos del Sistema

### Requisitos Mínimos
- **RAM**: 8 GB mínimo (16 GB recomendado)
- **Almacenamiento**: 5 GB de espacio libre
- **Procesador**: CPU multi-core (GPU/MPS opcional pero recomendado)

### Software
- **Python**: 3.8 o superior (3.9-3.11 recomendado)
- **pip**: Gestor de paquetes de Python

---

## 🚀 Instalación y Ejecución

### 📱 Para macOS (Apple Silicon / Intel)

#### 1. Verificar Python
```bash
python3 --version
```
Si no tienes Python instalado, descárgalo desde [python.org](https://www.python.org/downloads/) o usa Homebrew:
```bash
brew install python@3.11
```

#### 2. Crear entorno virtual (recomendado)
```bash
# Navegar al directorio del proyecto
cd "/Users/samuel/Desktop/testing NLP"

# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

#### 3. Instalar dependencias
```bash
# Actualizar pip
pip install --upgrade pip

# Instalar librerías necesarias
pip install -r requirements.txt
```

**Nota para Apple Silicon (M1/M2/M3):** PyTorch detectará automáticamente MPS (Metal Performance Shaders) para aceleración GPU.

#### 4. Ejecutar el análisis
```bash
python main.py
```

#### 5. Desactivar entorno virtual (cuando termines)
```bash
deactivate
```

---

### 🪟 Para Windows

#### 1. Verificar Python
Abre **Command Prompt** o **PowerShell** y ejecuta:
```cmd
python --version
```
Si no tienes Python, descárgalo desde [python.org](https://www.python.org/downloads/) e instálalo marcando "Add Python to PATH".

#### 2. Crear entorno virtual (recomendado)
```cmd
# Navegar al directorio del proyecto
cd "C:\Users\TuUsuario\Desktop\testing NLP"

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate
```

#### 3. Instalar dependencias
```cmd
# Actualizar pip
python -m pip install --upgrade pip

# Instalar librerías necesarias
pip install -r requirements.txt
```

**Nota para Windows con GPU NVIDIA:** Si tienes una tarjeta gráfica NVIDIA, instala PyTorch con soporte CUDA:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Ejecutar el análisis
```cmd
python main.py
```

#### 5. Desactivar entorno virtual (cuando termines)
```cmd
deactivate
```

---

## ⚙️ Configuración del Proyecto

### Parámetros Principales (main.py)

Puedes modificar estos parámetros en `main.py`:

```python
# Tamaño del dataset
dataset_small = dataset["train"].select(range(1000))  # Cambia 1000 por el número deseado

# División entrenamiento/test
dataset_split = dataset_small.train_test_split(test_size=0.2, seed=42)  # 80/20

# Parámetros de entrenamiento
training_args = TrainingArguments(
    learning_rate=2e-5,              # Tasa de aprendizaje
    per_device_train_batch_size=16,  # Tamaño del batch
    num_train_epochs=3,              # Número de épocas
    weight_decay=0.01,               # Regularización
)
```

### Modelos Evaluados

Para cambiar los modelos evaluados, modifica las llamadas en la función `main()`:

```python
# Modelos disponibles:
bert_metrics = train_and_evaluate("bert-base-uncased", train_dataset, test_dataset)
distilbert_metrics = train_and_evaluate("distilbert-base-uncased", train_dataset, test_dataset)
finbert_metrics = train_and_evaluate("yiyanghkust/finbert-tone", train_dataset, test_dataset)

# Otros modelos financieros que puedes probar:
# "ProsusAI/finbert"
# "ahmedrachid/FinancialBERT-Sentiment-Analysis"
```

---

## 📊 Resultados Esperados

Al ejecutar el script, obtendrás:

### Salida en Consola
- Progreso del entrenamiento para cada modelo
- Métricas por época
- Tabla comparativa final con:
  - Accuracy, F1-Score, Log Loss, ROC-AUC
  - Tiempos de entrenamiento e inferencia
  - Uso de memoria
  - F1-Score por clase (negativo, neutral, positivo)

### Archivos Generados
- `results_[modelo]/`: Checkpoints y modelos entrenados
- `output_results.txt`: Log completo de la ejecución
- `ANALISIS_RESULTADOS.md`: Análisis detallado de resultados

---

## 🐛 Solución de Problemas

### Error: "No module named 'transformers'"
**Solución:** Asegúrate de haber activado el entorno virtual y ejecutado `pip install -r requirements.txt`

### Error: "CUDA out of memory" o "MPS out of memory"
**Solución:** Reduce el batch size en `main.py`:
```python
per_device_train_batch_size=8,  # En lugar de 16
per_device_eval_batch_size=8,
```

### Error: "SSL Certificate verify failed"
**Solución (macOS):**
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Solución (Windows):**
```cmd
pip install --upgrade certifi
```

### Ejecución muy lenta
**Soluciones:**
1. Reduce el tamaño del dataset: `select(range(500))` en lugar de 1000
2. Reduce las épocas: `num_train_epochs=2`
3. Usa DistilBERT únicamente (es 2x más rápido)

### Error al descargar modelos
**Solución:** Verifica tu conexión a internet. Los modelos se descargan automáticamente desde Hugging Face (3-5 GB total).

---

## 📈 Interpretación de Resultados

### Métricas Principales

- **Accuracy**: Porcentaje de predicciones correctas (0-1)
- **F1-Score**: Balance entre precisión y recall (0-1, mayor es mejor)
- **Log Loss**: Confianza en predicciones (menor es mejor)
- **ROC-AUC**: Capacidad de discriminación entre clases (0-1, mayor es mejor)

### Comparación de Modelos

Según los resultados obtenidos:

| Métrica | BERT | DistilBERT | FinBERT | Ganador |
|---------|------|------------|---------|---------|
| Accuracy | 0.82 | 0.835 | **0.84** | ✅ FinBERT |
| F1-Score | 0.768 | 0.779 | **0.813** | ✅ FinBERT |
| Log Loss | 0.420 | 0.417 | **0.341** | ✅ FinBERT |
| Velocidad | Media | **Rápida** | Media | ✅ DistilBERT |
| Memoria | **877 MB** | 1012 MB | 1210 MB | ✅ BERT |

**Recomendación:** FinBERT es el mejor modelo para análisis financiero profesional, mientras que DistilBERT es ideal para aplicaciones que requieren velocidad.

---

## 📚 Dataset Utilizado

**Financial PhraseBank**
- Fuente: Hugging Face (`prithvi1029/sentiment-analysis-for-financial-news`)
- Contenido: 4,846 titulares financieros en inglés
- Clases: Negative (negativo), Neutral, Positive (positivo)
- Anotación: Realizada por expertos financieros

---

## 🤝 Contribuciones

Este proyecto es parte de un Trabajo Final de Grado (TFG). Para sugerencias o mejoras:

1. Identifica el problema o mejora
2. Documenta los cambios propuestos
3. Valida que las pruebas pasen correctamente

---

## 📄 Licencia

Este proyecto utiliza modelos y datos de código abierto:
- Modelos de Hugging Face (licencias Apache 2.0 / MIT)
- Dataset Financial PhraseBank (uso académico)

---

## 📞 Contacto y Soporte

Para preguntas sobre el proyecto o problemas técnicos:

- **Documentación de Transformers**: https://huggingface.co/docs/transformers
- **Documentación de PyTorch**: https://pytorch.org/docs/stable/index.html
- **Dataset**: https://huggingface.co/datasets/prithvi1029/sentiment-analysis-for-financial-news

---

## 🎯 Próximos Pasos

Después de ejecutar el análisis:

1. ✅ Revisa `ANALISIS_RESULTADOS.md` para interpretación detallada
2. ✅ Examina los modelos guardados en `results_*/`
3. ✅ Experimenta con diferentes hiperparámetros
4. ✅ Prueba con otros modelos financieros
5. ✅ Expande el dataset para mejorar resultados

---

## ⚡ Ejecución Rápida (Sin entorno virtual)

Si solo quieres probar rápidamente:

**macOS/Linux:**
```bash
cd "/Users/samuel/Desktop/testing NLP"
pip3 install -r requirements.txt
python3 main.py
```

**Windows:**
```cmd
cd "C:\Users\TuUsuario\Desktop\testing NLP"
pip install -r requirements.txt
python main.py
```

**⚠️ Nota:** Se recomienda usar entorno virtual para evitar conflictos de dependencias.

---

**¡Feliz análisis de sentimientos financieros! 📊💰**
