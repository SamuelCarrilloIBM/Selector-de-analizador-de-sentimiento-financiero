"""
Test NVIDIA: phi3.5:latest via Ollama — few-shot sobre nvidia_testing.csv
Genera: predictions_csv + metadata_json
"""
import os, time, csv, json, sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from collections import Counter
import ollama
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

MODEL       = "phi3.5:latest"
TEMPERATURE = 0.0
CSV_PATH    = "../../data/raw/nvidia_testing.csv"
RESULTS_DIR = "../../results/ollama/phi3.5"
LABEL_MAP   = {"positive": 2, "negative": 0, "neutral": 1}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

def load_nvidia_csv(path):
    samples = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label_str = row["Label"].strip().lower()
            if label_str not in LABEL_MAP:
                continue
            samples.append({"headline": row["Headline"].strip(), "label": LABEL_MAP[label_str]})
    return samples

samples = load_nvidia_csv(CSV_PATH)
print(f"Dataset: {CSV_PATH} — {len(samples)} noticias NVIDIA")

FEW_SHOT = samples[:3]
test_pool = samples[3:]

def build_prompt(text):
    examples = "\n".join(
        f'Noticia: "{s["headline"]}"\nSENTIMIENTO: {LABEL_NAMES[s["label"]]}'
        for s in FEW_SHOT
    )
    return (
        "Eres un analista financiero experto en NVIDIA y semiconductores.\n\n"
        "Clasifica el sentimiento de noticias sobre NVIDIA.\n"
        "- positive: crecimiento, ganancias, demanda, acuerdos favorables\n"
        "- negative: pérdidas, caídas, riesgos, regulación adversa\n"
        "- neutral: hecho informativo sin impacto claro\n\n"
        "Responde ÚNICAMENTE con una palabra: positive, negative o neutral\n\n"
        f"EJEMPLOS:\n{examples}\n\n"
        f'Noticia: "{text}"\nSENTIMIENTO:'
    )

def parse(raw):
    s = raw.strip().lower()
    if "positive" in s: return 2
    if "negative" in s: return 0
    if "neutral"  in s: return 1
    return 1

try:
    ollama.list()
except Exception as e:
    print(f"⚠️  No se pudo conectar con Ollama: {e}")
    sys.exit(1)

print(f"\n{'='*65}\nMODELO: {MODEL} — NVIDIA ({len(test_pool)} muestras)\n{'='*65}")
true_labels, pred_labels, headlines, inf_times_ms = [], [], [], []
errors = 0

for idx, s in enumerate(test_pool, 1):
    try:
        t0 = time.time()
        response = ollama.generate(model=MODEL, prompt=build_prompt(s["headline"]),
                                   options={'temperature': TEMPERATURE, 'num_predict': 20, 'stop': ['\n', '.']})
        elapsed_ms = (time.time() - t0) * 1000
        pred_label = parse(response['response'])
    except Exception as e:
        print(f"  Error muestra {idx}: {e}")
        pred_label, elapsed_ms, errors = 1, 0.0, errors + 1

    true_labels.append(s["label"])
    pred_labels.append(pred_label)
    headlines.append(s["headline"])
    inf_times_ms.append(round(elapsed_ms, 1))

accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
nivel = "ALTA ✓" if accuracy >= 0.8 else "MEDIA ⚠" if accuracy >= 0.6 else "BAJA ✗"
print(f"\nAccuracy: {accuracy:.1%} | {nivel} | {np.mean(inf_times_ms):.0f}ms/muestra")

os.makedirs(RESULTS_DIR, exist_ok=True)
p_w, r_w, f1_w, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)
p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1,2], zero_division=0)
cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2]).tolist()
pred_dist = Counter(LABEL_NAMES[p] for p in pred_labels)
true_dist = Counter(LABEL_NAMES[t] for t in true_labels)

csv_out = f"{RESULTS_DIR}/{MODEL.replace(':', '_')}_nvidia_predictions.csv"
with open(csv_out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto", "tiempo_ms"])
    writer.writeheader()
    for i, (h, t, p, ms) in enumerate(zip(headlines, true_labels, pred_labels, inf_times_ms), 1):
        writer.writerow({"id": i, "noticia": h, "etiqueta_real": LABEL_NAMES[t],
                         "etiqueta_predicha": LABEL_NAMES[p],
                         "correcto": "si" if t == p else "no", "tiempo_ms": ms})
print(f"  CSV: {csv_out}")

json_out = f"{RESULTS_DIR}/{MODEL.replace(':', '_')}_nvidia_metadata.json"
meta = {
    "model": MODEL, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": CSV_PATH, "n_test": len(true_labels),
    "approach": f"few-shot ({len(FEW_SHOT)} ejemplos NVIDIA)",
    "temperature": TEMPERATURE,
    "performance": {
        "accuracy": round(accuracy, 4), "accuracy_pct": f"{accuracy:.1%}",
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
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  JSON: {json_out}")
