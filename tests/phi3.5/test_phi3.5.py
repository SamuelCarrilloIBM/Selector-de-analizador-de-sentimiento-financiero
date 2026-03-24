"""
Test individual: phi3.5:latest via Ollama — few-shot
Genera: predictions_csv + metadata_json
"""
import os, time, csv, json, sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
from collections import Counter
import ollama
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random

MODEL       = "phi3.5:latest"
TEMPERATURE = 0.0
N_TEST      = 100
N_FEW_SHOT_PER_CLASS = 4
LABEL_MAP   = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
RESULTS_DIR = "../../results/ollama/phi3.5"

print("Cargando dataset...")
dataset = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
rows = dataset["train"].to_list()
random.seed(42)
random.shuffle(rows)

few_shot_pool = {0: [], 1: [], 2: []}
test_pool = []
for row in rows:
    label = LABEL_MAP[row["sentiment"]]
    if len(few_shot_pool[label]) < N_FEW_SHOT_PER_CLASS:
        few_shot_pool[label].append((row["news_headline"], row["sentiment"]))
    elif len(test_pool) < N_TEST:
        test_pool.append((row["news_headline"], label))

FEW_SHOT_EXAMPLES = few_shot_pool[0] + few_shot_pool[1] + few_shot_pool[2]

def build_prompt(text):
    examples = "\n".join(f'Texto: "{t}"\nSENTIMIENTO: {s}' for t, s in FEW_SHOT_EXAMPLES)
    return (
        "Eres un analista financiero experto en clasificación de sentimiento de noticias bursátiles.\n\n"
        "DEFINICIONES:\n"
        "- positive: la noticia implica crecimiento, ganancias, acuerdos favorables o mejora para la empresa\n"
        "- negative: la noticia implica pérdidas, caídas, riesgos, recortes o deterioro para la empresa\n"
        "- neutral: la noticia es un hecho informativo sin impacto claro positivo ni negativo\n\n"
        "REGLAS:\n"
        "- Responde ÚNICAMENTE con una de estas palabras exactas: positive, negative, neutral\n"
        "- No añadas explicaciones, puntuación ni texto adicional\n\n"
        f"EJEMPLOS:\n{examples}\n\n"
        f'Texto: "{text}"\nSENTIMIENTO:'
    )

def parse(raw):
    s = raw.strip().lower()
    if "sentimiento:" in s: s = s.split("sentimiento:")[-1].strip()
    if "positive" in s: return 2
    if "negative" in s: return 0
    if "neutral"  in s: return 1
    first = s.split()[0] if s.split() else ""
    if "pos" in first: return 2
    if "neg" in first: return 0
    return 1

def export_results(true_labels, pred_labels, headlines, inf_times_ms, errors):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f"{RESULTS_DIR}/{MODEL.replace(':', '_')}_predictions.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto", "tiempo_ms"])
        writer.writeheader()
        for i, (h, t, p, ms) in enumerate(zip(headlines, true_labels, pred_labels, inf_times_ms), 1):
            writer.writerow({"id": i, "noticia": h, "etiqueta_real": LABEL_NAMES[t],
                             "etiqueta_predicha": LABEL_NAMES[p],
                             "correcto": "si" if t == p else "no", "tiempo_ms": ms})
    print(f"  CSV predicciones: {csv_path}")

    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1,2], zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2]).tolist()
    pred_dist = Counter(LABEL_NAMES[p] for p in pred_labels)
    true_dist = Counter(LABEL_NAMES[t] for t in true_labels)

    json_path = f"{RESULTS_DIR}/{MODEL.replace(':', '_')}_metadata.json"
    meta = {
        "model": MODEL,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": "prithvi1029/sentiment-analysis-for-financial-news",
        "n_test": len(true_labels),
        "approach": f"few-shot ({len(FEW_SHOT_EXAMPLES)} ejemplos, {N_FEW_SHOT_PER_CLASS} por clase)",
        "temperature": TEMPERATURE,
        "performance": {
            "accuracy": round(accuracy, 4),
            "accuracy_pct": f"{accuracy:.1%}",
            "f1_weighted": round(float(f1_w), 4),
            "precision_weighted": round(float(p_w), 4),
            "recall_weighted": round(float(r_w), 4),
        },
        "per_class": {
            "negative": {"f1": round(float(f1_cls[0]), 4), "precision": round(float(p_cls[0]), 4), "recall": round(float(r_cls[0]), 4)},
            "neutral":  {"f1": round(float(f1_cls[1]), 4), "precision": round(float(p_cls[1]), 4), "recall": round(float(r_cls[1]), 4)},
            "positive": {"f1": round(float(f1_cls[2]), 4), "precision": round(float(p_cls[2]), 4), "recall": round(float(r_cls[2]), 4)},
        },
        "confusion_matrix": {"labels": ["negative", "neutral", "positive"], "matrix": cm},
        "distribution": {"predicted": dict(pred_dist), "true": dict(true_dist)},
        "efficiency": {
            "inference_ms_mean": round(float(np.mean(inf_times_ms)), 2),
            "inference_ms_std": round(float(np.std(inf_times_ms)), 2),
            "connection_errors": errors,
        },
        "reliability": "ALTA" if accuracy >= 0.8 else "MEDIA" if accuracy >= 0.6 else "BAJA",
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  JSON metadata:    {json_path}")

try:
    ollama.list()
except Exception as e:
    print(f"⚠️  No se pudo conectar con Ollama: {e}")
    sys.exit(1)

print(f"\n{'='*65}\nMODELO: {MODEL} ({N_TEST} muestras)\n{'='*65}")
true_labels, pred_labels, headlines, inf_times_ms = [], [], [], []
errors = 0

for idx, (text, true_label) in enumerate(test_pool, 1):
    try:
        t0 = time.time()
        response = ollama.generate(model=MODEL, prompt=build_prompt(text),
                                   options={'temperature': TEMPERATURE, 'num_predict': 20, 'stop': ['\n', '.']})
        elapsed_ms = (time.time() - t0) * 1000
        pred_label = parse(response['response'])
    except Exception as e:
        print(f"  Error muestra {idx}: {e}")
        pred_label, elapsed_ms, errors = 1, 0.0, errors + 1

    true_labels.append(true_label)
    pred_labels.append(pred_label)
    headlines.append(text)
    inf_times_ms.append(round(elapsed_ms, 1))
    if idx % 10 == 0:
        correct = sum(t == p for t, p in zip(true_labels, pred_labels))
        print(f"  {idx}/{N_TEST}... ({correct} correctas)")

accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / N_TEST
nivel = "ALTA ✓" if accuracy >= 0.8 else "MEDIA ⚠" if accuracy >= 0.6 else "BAJA ✗"
print(f"\nAccuracy: {accuracy:.1%} | {nivel} | {np.mean(inf_times_ms):.0f}ms/muestra")

print("\n📁 Exportando resultados...")
export_results(true_labels, pred_labels, headlines, inf_times_ms, errors)
