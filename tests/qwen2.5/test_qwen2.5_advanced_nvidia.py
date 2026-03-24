"""
Test qwen2.5:7b con prompt avanzado de análisis financiero NVIDIA.
Dataset: nvidia_real_news.csv (title + summary, sin ground truth)
Genera: predictions_csv + metadata_json

Usa el singleton QwenSentimentAnalyzer (utils/qwen_sentiment.py):
el modelo se carga y el system prompt se cachea una sola vez.
"""
import os, csv, json, sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from collections import Counter

# Añadir raíz del proyecto al path para importar utils/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.qwen_sentiment import get_analyzer

import ollama

CSV_PATH    = "../../data/raw/nvidia_real_news.csv"
RESULTS_DIR = "../../results/ollama/qwen2.5"


def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                "date":    row.get("date", "").strip(),
                "title":   row.get("title", "").strip(),
                "summary": row.get("summary", "").strip(),
            })
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

try:
    ollama.list()
except Exception as e:
    print(f"⚠️  No se pudo conectar con Ollama: {e}")
    sys.exit(1)

rows = load_csv(CSV_PATH)
print(f"Dataset: {CSV_PATH} — {len(rows)} noticias")

# Primera llamada → inicializa modelo + warmup (una sola vez)
analyzer = get_analyzer()

print(f"\n{'='*65}\nMODELO: {analyzer.__class__.__module__} — prompt avanzado NVIDIA\n{'='*65}")

results_data = []
inf_times_ms = []
errors = 0

for idx, row in enumerate(rows, 1):
    try:
        parsed = analyzer.analyze(row["title"], row["summary"])
    except Exception as e:
        print(f"  Error muestra {idx}: {e}")
        parsed = {"sentiment": "neutral", "intensity": "weak",
                  "relevance": "none", "reasoning": f"ERROR: {e}", "tiempo_ms": 0.0}
        errors += 1

    results_data.append({**row, **parsed})
    inf_times_ms.append(parsed["tiempo_ms"])

    if idx % 10 == 0 or idx == len(rows):
        dist = Counter(r["sentiment"] for r in results_data)
        print(f"  [{idx}/{len(rows)}] pos={dist['positive']} neu={dist['neutral']} neg={dist['negative']} | {np.mean(inf_times_ms):.0f}ms/avg")

# ─── Export ───────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

csv_out = f"{RESULTS_DIR}/qwen2.5_advanced_nvidia_real_predictions.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "date", "title", "summary", "sentiment",
                                           "intensity", "relevance", "reasoning", "tiempo_ms"])
    writer.writeheader()
    for i, r in enumerate(results_data, 1):
        writer.writerow({
            "id": i, "date": r["date"], "title": r["title"], "summary": r["summary"],
            "sentiment": r["sentiment"], "intensity": r["intensity"],
            "relevance": r["relevance"], "reasoning": r["reasoning"],
            "tiempo_ms": r["tiempo_ms"],
        })

dist_sentiment = Counter(r["sentiment"] for r in results_data)
dist_intensity = Counter(r["intensity"] for r in results_data)
dist_relevance = Counter(r["relevance"] for r in results_data)

json_out = f"{RESULTS_DIR}/qwen2.5_advanced_nvidia_real_metadata.json"
meta = {
    "model": "qwen2.5:7b",
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": CSV_PATH,
    "n_samples": len(results_data),
    "approach": "advanced financial prompt — chain-of-thought with relevance gate",
    "temperature": 0.0,
    "prompt_version": "v2 — relevance gate + 5-step CoT + few-shot difficult cases",
    "distribution": {
        "sentiment": dict(dist_sentiment),
        "intensity": dict(dist_intensity),
        "relevance": dict(dist_relevance),
    },
    "efficiency": {
        "inference_ms_mean": round(float(np.mean(inf_times_ms)), 2),
        "inference_ms_std":  round(float(np.std(inf_times_ms)), 2),
        "connection_errors": errors,
    },
}
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nDistribución: {dict(dist_sentiment)}")
print(f"Relevancia:   {dict(dist_relevance)}")
print(f"\n{'='*65}")
print(f"RESULTADOS GUARDADOS EN:")
print(f"  CSV  → {os.path.abspath(csv_out)}")
print(f"  JSON → {os.path.abspath(json_out)}")
print(f"{'='*65}")
