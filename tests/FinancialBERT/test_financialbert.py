"""
Test individual: ahmedrachid/FinancialBERT-Sentiment-Analysis
Zero-shot evaluation — ya entrenado en dominio financiero.
Genera: predictions_csv + metadata_json
"""
import os, time, csv, json
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import psutil
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, log_loss)
from scipy.special import softmax

MODEL_NAME  = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
RESULTS_DIR = "../../results/bert/financialbert"
LABEL_MAP   = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p, "recall": r, "f1": f1,
        "f1_neg": float(f1_cls[0]), "f1_neu": float(f1_cls[1]), "f1_pos": float(f1_cls[2]),
        "precision_neg": float(p_cls[0]), "precision_neu": float(p_cls[1]), "precision_pos": float(p_cls[2]),
        "recall_neg": float(r_cls[0]), "recall_neu": float(r_cls[1]), "recall_pos": float(r_cls[2]),
        "log_loss": log_loss(labels, probs),
        "roc_auc": roc_auc_score(labels, probs, multi_class='ovr', average='weighted'),
    }

def export_results(model_name, headlines, true_labels, pred_labels, metrics,
                   train_time, inf_times, mem_mb, results_dir, suffix):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = f"{results_dir}/{suffix}_predictions.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto"])
        writer.writeheader()
        for i, (h, t, p) in enumerate(zip(headlines, true_labels, pred_labels), 1):
            writer.writerow({"id": i, "noticia": h, "etiqueta_real": LABEL_NAMES[t],
                             "etiqueta_predicha": LABEL_NAMES[p], "correcto": "si" if t == p else "no"})
    print(f"  CSV predicciones: {csv_path}")

    cm = confusion_matrix(true_labels, pred_labels).tolist()
    json_path = f"{results_dir}/{suffix}_metadata.json"
    meta = {
        "model": model_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": "prithvi1029/sentiment-analysis-for-financial-news",
        "n_test": len(true_labels),
        "approach": "zero-shot (pre-trained on financial sentiment, no additional fine-tuning)",
        "performance": {
            "accuracy": round(metrics['eval_accuracy'], 4),
            "accuracy_pct": f"{metrics['eval_accuracy']:.1%}",
            "f1_weighted": round(metrics['eval_f1'], 4),
            "precision_weighted": round(metrics['eval_precision'], 4),
            "recall_weighted": round(metrics['eval_recall'], 4),
            "roc_auc": round(metrics['eval_roc_auc'], 4),
            "log_loss": round(metrics['eval_log_loss'], 4),
        },
        "per_class": {
            "negative": {"f1": round(metrics['eval_f1_neg'], 4),
                         "precision": round(metrics['eval_precision_neg'], 4),
                         "recall": round(metrics['eval_recall_neg'], 4)},
            "neutral":  {"f1": round(metrics['eval_f1_neu'], 4),
                         "precision": round(metrics['eval_precision_neu'], 4),
                         "recall": round(metrics['eval_recall_neu'], 4)},
            "positive": {"f1": round(metrics['eval_f1_pos'], 4),
                         "precision": round(metrics['eval_precision_pos'], 4),
                         "recall": round(metrics['eval_recall_pos'], 4)},
        },
        "confusion_matrix": {"labels": LABEL_NAMES, "matrix": cm},
        "efficiency": {
            "train_time_s": 0.0,
            "inference_ms_mean": round(float(np.mean(inf_times)), 2),
            "inference_ms_std": round(float(np.std(inf_times)), 2),
            "memory_mb": round(mem_mb, 1),
        },
        "reliability": "ALTA" if metrics['eval_accuracy'] >= 0.8 else
                       "MEDIA" if metrics['eval_accuracy'] >= 0.6 else "BAJA",
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  JSON metadata:    {json_path}")

print("Cargando dataset...")
ds = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
ds_small = ds["train"].select(range(1000))
split = ds_small.train_test_split(test_size=0.2, seed=42)
test_ds = split["test"].rename_column("sentiment", "labels")
test_ds = test_ds.map(lambda b: {"labels": [LABEL_MAP[l] for l in b["labels"]]}, batched=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(batch["news_headline"], padding=True, truncation=True, max_length=128)
test_tok = test_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
args = TrainingArguments(
    output_dir=f"{RESULTS_DIR}/checkpoints_financialbert",
    per_device_eval_batch_size=16, report_to="none",
    dataloader_num_workers=0,
)
trainer = Trainer(model=model, args=args, eval_dataset=test_tok,
                  compute_metrics=compute_metrics)

print(f"\n{'='*60}\nMODELO: {MODEL_NAME} (zero-shot)\n{'='*60}")

proc = psutil.Process(os.getpid())
metrics = trainer.evaluate()
mem_mb = proc.memory_info().rss / 1024 / 1024

inf_times = []
for _ in range(5):
    t0 = time.time()
    trainer.predict(test_tok)
    inf_times.append((time.time() - t0) * 1000 / len(test_tok))

preds_output = trainer.predict(test_tok)
pred_labels = np.argmax(preds_output.predictions, axis=-1).tolist()
true_labels = list(test_ds["labels"])
headlines   = list(test_ds["news_headline"])

print(f"\nAccuracy: {metrics['eval_accuracy']:.1%} | F1: {metrics['eval_f1']:.4f} | ROC-AUC: {metrics['eval_roc_auc']:.4f}")
print(f"Inferencia: {np.mean(inf_times):.2f}ms/muestra | RAM: {mem_mb:.0f}MB")

print("\n📁 Exportando resultados...")
export_results(MODEL_NAME, headlines, true_labels, pred_labels,
               metrics, 0.0, inf_times, mem_mb, RESULTS_DIR, "financialbert")
