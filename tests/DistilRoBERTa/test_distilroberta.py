"""
Test individual: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
Zero-shot sobre dataset de sentimiento financiero (prithvi1029).
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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, log_loss)
from scipy.special import softmax

MODEL_NAME  = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
RESULTS_DIR = "../../results/bert/distilroberta"
LABEL_MAP   = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

print("Cargando dataset...")
ds = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
ds_small = ds["train"].select(range(1000))
split = ds_small.train_test_split(test_size=0.2, seed=42)
test_ds = split["test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)

true_labels, pred_labels, headlines, inf_times_ms, all_probs = [], [], [], [], []

print(f"\n{'='*60}\nMODELO: {MODEL_NAME}\n{'='*60}")
for row in test_ds:
    label_str = row["sentiment"].strip().lower()
    if label_str not in LABEL_MAP:
        continue
    true_labels.append(LABEL_MAP[label_str])
    headlines.append(row["news_headline"])

    inputs = tokenizer(row["news_headline"], return_tensors="pt",
                       truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        logits = model(**inputs).logits
    elapsed_ms = (time.time() - t0) * 1000

    probs = softmax(logits.cpu().numpy(), axis=-1)[0]
    pred = int(np.argmax(probs))
    pred_labels.append(pred)
    all_probs.append(probs)
    inf_times_ms.append(round(elapsed_ms, 1))

all_probs = np.array(all_probs)
accuracy = accuracy_score(true_labels, pred_labels)
p_w, r_w, f1_w, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)
p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None,
                                                           labels=[0, 1, 2], zero_division=0)
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2]).tolist()
roc = roc_auc_score(true_labels, all_probs, multi_class='ovr', average='weighted')
ll  = log_loss(true_labels, all_probs)

print(f"Accuracy: {accuracy:.1%} | F1: {f1_w:.4f} | ROC-AUC: {roc:.4f} | LogLoss: {ll:.4f}")
print(f"Inferencia: {np.mean(inf_times_ms):.2f}ms/muestra")

proc = psutil.Process(os.getpid())
mem_mb = proc.memory_info().rss / 1024 / 1024

os.makedirs(RESULTS_DIR, exist_ok=True)
csv_out = f"{RESULTS_DIR}/distilroberta_predictions.csv"
with open(csv_out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real",
                                           "etiqueta_predicha", "correcto", "tiempo_ms"])
    writer.writeheader()
    for i, (h, t, p, ms) in enumerate(zip(headlines, true_labels, pred_labels, inf_times_ms), 1):
        writer.writerow({"id": i, "noticia": h, "etiqueta_real": LABEL_NAMES[t],
                         "etiqueta_predicha": LABEL_NAMES[p],
                         "correcto": "si" if t == p else "no", "tiempo_ms": ms})
print(f"  CSV: {csv_out}")

json_out = f"{RESULTS_DIR}/distilroberta_metadata.json"
meta = {
    "model": MODEL_NAME,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": "prithvi1029/sentiment-analysis-for-financial-news",
    "n_test": len(true_labels),
    "approach": "zero-shot (fine-tuned on financial news sentiment)",
    "performance": {
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy:.1%}",
        "f1_weighted": round(float(f1_w), 4),
        "precision_weighted": round(float(p_w), 4),
        "recall_weighted": round(float(r_w), 4),
        "roc_auc": round(float(roc), 4),
        "log_loss": round(float(ll), 4),
    },
    "per_class": {
        "negative": {"f1": round(float(f1_cls[0]), 4),
                     "precision": round(float(p_cls[0]), 4),
                     "recall": round(float(r_cls[0]), 4)},
        "neutral":  {"f1": round(float(f1_cls[1]), 4),
                     "precision": round(float(p_cls[1]), 4),
                     "recall": round(float(r_cls[1]), 4)},
        "positive": {"f1": round(float(f1_cls[2]), 4),
                     "precision": round(float(p_cls[2]), 4),
                     "recall": round(float(r_cls[2]), 4)},
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
