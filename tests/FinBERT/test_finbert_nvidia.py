"""
Test NVIDIA: yiyanghkust/finbert-tone — zero-shot sobre nvidia_testing.csv
Genera: predictions_csv + metadata_json
"""
import os, time, csv, json
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import torch, psutil
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

MODEL_NAME  = "yiyanghkust/finbert-tone"
CSV_PATH    = "../../data/raw/nvidia_testing.csv"
RESULTS_DIR = "../../results/bert/finbert"
# Mapa para convertir etiquetas del CSV a índices internos
LABEL_MAP   = {"positive": 2, "negative": 0, "neutral": 1}
LABEL_NAMES = ["negative", "neutral", "positive"]
# finbert-tone tiene id2label: {0: "Neutral", 1: "Positive", 2: "Negative"}
# Necesitamos mapear los índices del modelo a nuestros índices internos
FINBERT_IDX_TO_INTERNAL = {0: 1, 1: 2, 2: 0}  # Neutral→1, Positive→2, Negative→0


def load_nvidia_csv(path):
    samples = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label_str = row["Label"].strip().lower()
            if label_str not in LABEL_MAP:
                continue
            samples.append({"headline": row["Headline"].strip(), "label": LABEL_MAP[label_str]})
    return samples


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

samples = load_nvidia_csv(CSV_PATH)
print(f"Dataset: {CSV_PATH} — {len(samples)} noticias NVIDIA")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# No usar ignore_mismatched_sizes — finbert-tone tiene 3 labels y queremos la cabeza preentrenada
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)

true_labels, pred_labels, headlines, inf_times_ms = [], [], [], []

print(f"\n{'='*60}\nMODELO: {MODEL_NAME} (zero-shot NVIDIA)\n{'='*60}")
for s in samples:
    inputs = tokenizer(s["headline"], return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        logits = model(**inputs).logits
    elapsed_ms = (time.time() - t0) * 1000
    # Convertir índice del modelo (Positive=0, Negative=1, Neutral=2) a índice interno
    raw_pred = int(torch.argmax(logits, dim=-1).item())
    pred = FINBERT_IDX_TO_INTERNAL[raw_pred]
    true_labels.append(s["label"])
    pred_labels.append(pred)
    headlines.append(s["headline"])
    inf_times_ms.append(round(elapsed_ms, 1))

accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
p_w, r_w, f1_w, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)
p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1,2], zero_division=0)
cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2]).tolist()

print(f"Accuracy: {accuracy:.1%} | F1: {f1_w:.4f} | {np.mean(inf_times_ms):.1f}ms/muestra")

os.makedirs(RESULTS_DIR, exist_ok=True)
csv_out = f"{RESULTS_DIR}/finbert_nvidia_predictions.csv"
with open(csv_out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto", "tiempo_ms"])
    writer.writeheader()
    for i, (h, t, p, ms) in enumerate(zip(headlines, true_labels, pred_labels, inf_times_ms), 1):
        writer.writerow({"id": i, "noticia": h, "etiqueta_real": LABEL_NAMES[t],
                         "etiqueta_predicha": LABEL_NAMES[p],
                         "correcto": "si" if t == p else "no", "tiempo_ms": ms})
print(f"  CSV: {csv_out}")

proc = psutil.Process(os.getpid())
mem_mb = proc.memory_info().rss / 1024 / 1024
json_out = f"{RESULTS_DIR}/finbert_nvidia_metadata.json"
meta = {
    "model": MODEL_NAME, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": CSV_PATH, "n_test": len(true_labels),
    "approach": "zero-shot (pre-trained on financial sentiment, NVIDIA dataset)",
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
    "confusion_matrix": {"labels": LABEL_NAMES, "matrix": cm},
    "efficiency": {
        "inference_ms_mean": round(float(np.mean(inf_times_ms)), 2),
        "inference_ms_std": round(float(np.std(inf_times_ms)), 2),
        "memory_mb": round(mem_mb, 1),
    },
    "reliability": "ALTA" if accuracy >= 0.8 else "MEDIA" if accuracy >= 0.6 else "BAJA",
}
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  JSON: {json_out}")
