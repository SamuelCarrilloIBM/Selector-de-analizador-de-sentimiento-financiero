from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)
import time
import psutil
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Importar ollama (opcional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama no está instalado. Se omitirá del análisis.")

# --- Función de métricas mejorada ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Métricas básicas
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    # Precisión, Recall y F1 por clase
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(labels, predictions)
    
    # Log Loss (requiere probabilidades)
    from scipy.special import softmax
    probs = softmax(logits, axis=-1)
    logloss = log_loss(labels, probs)
    
    # ROC-AUC (multiclase one-vs-rest)
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_neg": float(precision_per_class[0]),
        "precision_neu": float(precision_per_class[1]),
        "precision_pos": float(precision_per_class[2]),
        "recall_neg": float(recall_per_class[0]),
        "recall_neu": float(recall_per_class[1]),
        "recall_pos": float(recall_per_class[2]),
        "f1_neg": float(f1_per_class[0]),
        "f1_neu": float(f1_per_class[1]),
        "f1_pos": float(f1_per_class[2]),
        "log_loss": logloss,
        "roc_auc": roc_auc,
    }

OLLAMA_MODEL = 'llama3.2'  # modelo de texto puro, sin vision
OLLAMA_TEMPERATURE = 0.0

# Ejemplos few-shot extraídos del dominio financiero para guiar al modelo
FEW_SHOT_EXAMPLES = [
    ("Company reports record profits beating analyst expectations", "positive"),
    ("Stock plunges after disappointing quarterly earnings miss", "negative"),
    ("Federal Reserve holds interest rates steady at current levels", "neutral"),
    ("Firm announces major layoffs amid restructuring plan", "negative"),
    ("Merger deal approved by shareholders in landmark vote", "positive"),
    ("Oil prices remain unchanged following OPEC meeting", "neutral"),
]

def _build_few_shot_prompt(text: str) -> str:
    """Construye un prompt few-shot con ejemplos financieros reales."""
    examples_block = "\n".join(
        f'Texto: "{t}"\nSentimiento: {s}' for t, s in FEW_SHOT_EXAMPLES
    )
    return (
        "Eres un experto en análisis de sentimiento financiero. "
        "Clasifica el sentimiento del texto como EXACTAMENTE una de estas palabras: positive, negative, neutral.\n\n"
        "Ejemplos:\n"
        f"{examples_block}\n\n"
        f'Texto: "{text}"\n'
        "Sentimiento:"
    )

def _parse_sentiment(raw: str) -> int:
    """Mapea la respuesta del modelo a etiqueta numérica (0=neg, 1=neu, 2=pos)."""
    s = raw.strip().lower().split()[0] if raw.strip() else ""
    if "positive" in s:
        return 2
    if "negative" in s:
        return 0
    return 1

def _check_ollama_model() -> bool:
    """Verifica que el modelo esté disponible en Ollama."""
    if not OLLAMA_AVAILABLE:
        print("⚠️  Ollama no está instalado.")
        return False
    try:
        models_response = ollama.list()
        available = [m.get('model', m.get('name', '')) for m in models_response.get('models', [])]
        if not any(OLLAMA_MODEL in m for m in available):
            print(f"⚠️  Modelo {OLLAMA_MODEL} no encontrado. Ejecuta: ollama pull {OLLAMA_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"⚠️  No se pudo conectar con Ollama: {e}")
        return False

# --- Prueba rápida: 10 noticias con reporte de fiabilidad ---
def test_ollama_10_samples(dataset):
    """
    Prueba Ollama con 10 noticias del dataset usando few-shot prompting.
    Muestra predicción vs etiqueta real y calcula fiabilidad (accuracy).
    """
    print(f"\n{'='*60}")
    print("PRUEBA OLLAMA: 10 NOTICIAS (FEW-SHOT, TEMPERATURA 0.0)")
    print(f"{'='*60}")

    if not _check_ollama_model():
        return

    label_names = {0: "negative", 1: "neutral", 2: "positive"}

    # Seleccionar 10 muestras distribuidas (no solo las primeras)
    indices = list(range(0, len(dataset), max(1, len(dataset) // 10)))[:10]
    samples = [dataset[i] for i in indices]

    correct = 0
    print(f"\n{'#':<4} {'Noticia':<55} {'Real':<10} {'Predicho':<10} {'OK'}")
    print("-" * 90)

    for idx, sample in enumerate(samples, 1):
        text = sample['news_headline']
        true_label = sample['labels']
        prompt = _build_few_shot_prompt(text)

        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={'temperature': OLLAMA_TEMPERATURE, 'num_predict': 5}
            )
            raw = response['response']
        except Exception as e:
            print(f"Error en muestra {idx}: {e}")
            raw = "neutral"

        pred_label = _parse_sentiment(raw)
        hit = pred_label == true_label
        if hit:
            correct += 1

        truncated = text[:52] + "..." if len(text) > 52 else text
        mark = "✓" if hit else "✗"
        print(f"{idx:<4} {truncated:<55} {label_names[true_label]:<10} {label_names[pred_label]:<10} {mark}")

    accuracy = correct / len(samples)
    print("-" * 90)
    print(f"\nResultados: {correct}/{len(samples)} correctas")
    print(f"Fiabilidad (accuracy): {accuracy:.0%}")

    if accuracy >= 0.8:
        nivel = "ALTA ✓"
    elif accuracy >= 0.6:
        nivel = "MEDIA ⚠"
    else:
        nivel = "BAJA ✗"
    print(f"Nivel de fiabilidad: {nivel}")
    print(f"Temperatura usada: {OLLAMA_TEMPERATURE} | Modo: few-shot ({len(FEW_SHOT_EXAMPLES)} ejemplos)")

# --- Función para evaluar Ollama (sin entrenamiento) ---
def evaluate_ollama(test_dataset):
    """
    Evalúa Ollama con few-shot prompting y temperatura 0.0 sobre todo el test set.
    """
    print(f"\n{'='*60}")
    print(f"Procesando modelo: Ollama {OLLAMA_MODEL} (few-shot)")
    print(f"{'='*60}")

    if not _check_ollama_model():
        return None

    try:
        texts = [item['news_headline'] for item in test_dataset]
        true_labels = [item['labels'] for item in test_dataset]

        print(f"Analizando {len(texts)} textos con few-shot prompting...")
        predictions = []
        start_time = time.time()

        for i, text in enumerate(texts):
            try:
                prompt = _build_few_shot_prompt(text)
                response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    options={'temperature': OLLAMA_TEMPERATURE, 'num_predict': 5}
                )
                predictions.append(_parse_sentiment(response['response']))
            except Exception as e:
                print(f"Error en texto {i}: {e}")
                predictions.append(1)

            if (i + 1) % 20 == 0:
                print(f"Procesados {i + 1}/{len(texts)} textos...")

        inference_time = time.time() - start_time

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted", zero_division=0
        )
        acc = accuracy_score(true_labels, predictions)

        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )

        n_samples = len(predictions)
        probs = np.zeros((n_samples, 3))
        for i, pred in enumerate(predictions):
            probs[i, pred] = 0.9
            for j in range(3):
                if j != pred:
                    probs[i, j] = 0.05

        try:
            logloss = log_loss(true_labels, probs)
            roc_auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='weighted')
        except Exception:
            logloss = 0.0
            roc_auc = 0.0

        avg_inference_time = (inference_time * 1000) / len(texts)
        print(f"\nTiempo total: {inference_time:.2f}s | Promedio por muestra: {avg_inference_time:.2f}ms")

        return {
            'eval_accuracy': acc,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            'eval_precision_neg': float(precision_per_class[0]),
            'eval_precision_neu': float(precision_per_class[1]),
            'eval_precision_pos': float(precision_per_class[2]),
            'eval_recall_neg': float(recall_per_class[0]),
            'eval_recall_neu': float(recall_per_class[1]),
            'eval_recall_pos': float(recall_per_class[2]),
            'eval_f1_neg': float(f1_per_class[0]),
            'eval_f1_neu': float(f1_per_class[1]),
            'eval_f1_pos': float(f1_per_class[2]),
            'eval_log_loss': logloss,
            'eval_roc_auc': roc_auc,
            'train_time_seconds': 0.0,
            'eval_time_seconds': inference_time,
            'inference_time_ms_mean': avg_inference_time,
            'inference_time_ms_std': 0.0,
            'memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            'memory_increase_mb': 0.0
        }

    except Exception as e:
        print(f"\n⚠️  Error al evaluar Ollama: {e}")
        print("1. Ollama está instalado (https://ollama.ai)")
        print("2. El servicio está corriendo")
        print(f"3. Has descargado el modelo: ollama pull {OLLAMA_MODEL}")
        return None


# --- Función para evaluar modelo ONNX (sin entrenamiento) ---
def evaluate_onnx_model(model_path, test_dataset):
    """
    Evalúa un modelo ONNX pre-exportado.
    No requiere entrenamiento, solo inferencia.
    """
    print(f"\n{'='*60}")
    print(f"Procesando modelo ONNX: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"⚠️  Directorio {model_path} no encontrado.")
        print(f"   Ejecuta primero: python export_distilroberta_onnx.py")
        return None
    
    try:
        # Cargar modelo ONNX
        print("🚀 Cargando modelo ONNX optimizado...")
        load_start = time.time()
        model = ORTModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        load_time = time.time() - load_start
        
        # Tokenizar dataset
        def tokenize(batch):
            return tokenizer(
                batch["news_headline"],
                padding=True,
                truncation=True,
                max_length=128
            )
        
        tokenized_test = test_dataset.map(tokenize, batched=True)
        
        # Configurar Trainer para evaluación
        training_args = TrainingArguments(
            output_dir=f"./results_{model_path.replace('/', '_')}",
            per_device_eval_batch_size=16,
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Evaluación
        print(f"Iniciando evaluación...")
        eval_start = time.time()
        metrics = trainer.evaluate()
        eval_time = time.time() - eval_start
        
        # Medición de tiempo de inferencia
        print(f"Calculando tiempo de inferencia...")
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        num_iterations = 5
        inference_times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = trainer.predict(tokenized_test)
            total_time = (time.time() - start) * 1000
            inference_times.append(total_time / len(tokenized_test))
        
        mem_after = process.memory_info().rss / 1024 / 1024
        
        # Agregar métricas de rendimiento
        metrics['train_time_seconds'] = 0.0  # No hay entrenamiento
        metrics['eval_time_seconds'] = eval_time
        metrics['load_time_seconds'] = load_time
        metrics['inference_time_ms_mean'] = np.mean(inference_times)
        metrics['inference_time_ms_std'] = np.std(inference_times)
        metrics['memory_mb'] = mem_after
        metrics['memory_increase_mb'] = mem_after - mem_before
        
        print(f"\nTiempo de carga: {load_time:.2f}s")
        print(f"Tiempo de evaluación: {eval_time:.2f}s")
        print(f"Tiempo de inferencia (media): {np.mean(inference_times):.2f}ms ± {np.std(inference_times):.2f}ms")
        print(f"Uso de memoria: {mem_after:.2f} MB")
        
        return metrics
        
    except Exception as e:
        print(f"\n⚠️  Error al evaluar modelo ONNX: {e}")
        print("Asegúrate de haber ejecutado: python export_distilroberta_onnx.py")
        return None


# --- Función de entrenamiento y evaluación con métricas de rendimiento ---
def train_and_evaluate(model_name, train_dataset, test_dataset):
    print(f"\n{'='*60}")
    print(f"Procesando modelo: {model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["news_headline"],
            padding=True,
            truncation=True,
            max_length=128
        )

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # NEG, NEU, POS
    )

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.replace('/', '_')}",
        eval_strategy="epoch",
        save_strategy="no",  # No guardar checkpoints para ahorrar espacio
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Entrenamiento
    print(f"Iniciando entrenamiento...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    # Evaluación
    print(f"Iniciando evaluación...")
    eval_start = time.time()
    metrics = trainer.evaluate()
    eval_time = time.time() - eval_start
    
    # Medición de tiempo de inferencia y uso de memoria
    print(f"Calculando tiempo de inferencia...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Medir tiempo de inferencia en todo el conjunto de test
    num_iterations = 5
    inference_times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = trainer.predict(tokenized_test)
        total_time = (time.time() - start) * 1000  # ms
        # Tiempo promedio por muestra
        inference_times.append(total_time / len(tokenized_test))
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Agregar métricas de rendimiento
    metrics['train_time_seconds'] = train_time
    metrics['eval_time_seconds'] = eval_time
    metrics['inference_time_ms_mean'] = np.mean(inference_times)
    metrics['inference_time_ms_std'] = np.std(inference_times)
    metrics['memory_mb'] = mem_after
    metrics['memory_increase_mb'] = mem_after - mem_before
    
    print(f"\nTiempo de entrenamiento: {train_time:.2f}s")
    print(f"Tiempo de evaluación: {eval_time:.2f}s")
    print(f"Tiempo de inferencia (media): {np.mean(inference_times):.2f}ms ± {np.std(inference_times):.2f}ms")
    print(f"Uso de memoria: {mem_after:.2f} MB")
    
    return metrics

# --- Función principal ---
def main():
    # Cargar dataset
    dataset = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
    print("Filas del dataset original:", dataset["train"].num_rows)

    # Limitar a 1000 muestras
    dataset_small = dataset["train"].select(range(1000))
    print("Filas del dataset limitado a 1000:", len(dataset_small))

    # División entrenamiento/prueba 80/20
    dataset_split = dataset_small.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split["train"].rename_column("sentiment", "labels")
    test_dataset = dataset_split["test"].rename_column("sentiment", "labels")

    # Mapeo de etiquetas a números
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    train_dataset = train_dataset.map(
        lambda batch: {"labels": [label_map[l] for l in batch["labels"]]},
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda batch: {"labels": [label_map[l] for l in batch["labels"]]},
        batched=True
    )

    print("Columnas del dataset tras renombrar:", train_dataset.column_names)

    # Evaluar BERT
    print("\n🔹 Evaluando BERT...")
    bert_metrics = train_and_evaluate("bert-base-uncased", train_dataset, test_dataset)
    
    # Evaluar DistilBERT
    print("\n🔹 Evaluando DistilBERT...")
    distilbert_metrics = train_and_evaluate("distilbert-base-uncased", train_dataset, test_dataset)
    
    # Evaluar FinBERT
    print("\n🔹 Evaluando FinBERT...")
    finbert_metrics = train_and_evaluate("yiyanghkust/finbert-tone", train_dataset, test_dataset)
    

    # Prueba rápida de Ollama con 10 noticias
    if OLLAMA_AVAILABLE:
        test_ollama_10_samples(test_dataset)

    # Evaluar Ollama completo (opcional)
    ollama_metrics = None
    if OLLAMA_AVAILABLE:
        print("\n🔹 Evaluando Ollama Llama 3.2-Vision (few-shot, temperatura 0.0)...")
        ollama_metrics = evaluate_ollama(test_dataset)

    # Resultados Comparativos
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO DE RESULTADOS")
    print("="*80)
    
    models_results = {
        "BERT": bert_metrics,
        "DistilBERT": distilbert_metrics,
        "FinBERT": finbert_metrics
    }
    
    # Añadir DistilRoBERTa ONNX si está disponible
    if distilroberta_onnx_metrics is not None:
        models_results["DistilRoBERTa ONNX"] = distilroberta_onnx_metrics
    
    # Añadir Ollama si está disponible
    if ollama_metrics is not None:
        models_results["Ollama 3.2-Vision"] = ollama_metrics
    
    # Tabla comparativa
    print("\n📊 TABLA COMPARATIVA DE MÉTRICAS PRINCIPALES")
    print("-" * 80)
    print(f"{'Modelo':<15} {'Accuracy':<10} {'F1-Score':<10} {'Log Loss':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<15} {metrics['eval_accuracy']:<10.4f} {metrics['eval_f1']:<10.4f} "
              f"{metrics['eval_log_loss']:<10.4f} {metrics['eval_roc_auc']:<10.4f}")
    
    print("\n📊 MÉTRICAS DE RENDIMIENTO")
    print("-" * 80)
    print(f"{'Modelo':<15} {'Tiempo Entrena (s)':<20} {'Inferencia (ms)':<25} {'Memoria (MB)':<15}")
    print("-" * 80)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<15} {metrics['train_time_seconds']:<20.2f} "
              f"{metrics['inference_time_ms_mean']:.2f} ± {metrics['inference_time_ms_std']:.2f}{'':<10} "
              f"{metrics['memory_mb']:<15.2f}")
    
    print("\n📊 MÉTRICAS POR CLASE (F1-Score)")
    print("-" * 80)
    print(f"{'Modelo':<15} {'Negativo':<12} {'Neutral':<12} {'Positivo':<12}")
    print("-" * 80)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<15} {metrics['eval_f1_neg']:<12.4f} {metrics['eval_f1_neu']:<12.4f} "
              f"{metrics['eval_f1_pos']:<12.4f}")
    
    print("\n" + "="*80)
    print("ANÁLISIS FINALIZADO")
    print("="*80)

if __name__ == "__main__":
    main()
