"""
Comparativa NVIDIA: phi3.5 MLX base vs phi3.5 MLX fine-tuneado (LoRA)
Genera: predictions_csv + metadata_json para cada modelo
Requiere haber ejecutado train_mlx.py previamente.

Uso:
    python3 test_mlx_nvidia.py
    python3 test_mlx_nvidia.py --adapter-path ../../models/lora_checkpoints --csv ../../data/raw/nvidia_testing.csv
"""
import os, csv, json, sys, time, argparse
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--adapter-path", type=str, default="../../models/lora_checkpoints")
parser.add_argument("--csv",          type=str, default="../../data/raw/nvidia_testing.csv")
args = parser.parse_args()

MLX_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LABEL_MAP   = {"positive": 2, "negative": 0, "neutral": 1}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
RESULTS_DIR_BASE = "../../results/mlx/phi3.5-base"
RESULTS_DIR_LORA = "../../results/mlx/phi3.5-lora"

def load_nvidia_csv(path):
    samples = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label_str = row["Label"].strip().lower()
            if label_str not in LABEL_MAP:
                continue
            samples.append({
                "date":     row["Date"].strip(),
                "headline": row["Headline"].strip(),
                "label":    LABEL_MAP[label_str],
                "score":    float(row["Sentiment_Score"]),
            })
    return samples

samples = load_nvidia_csv(args.csv)
print(f"Dataset: {args.csv} — {len(samples)} noticias NVIDIA")

def parse(raw):
    s = raw.strip().lower()
    if "positive" in s: return 2
    if "negative" in s: return 0
    if "neutral"  in s: return 1
    first = s.split()[0] if s.split() else ""
    if "pos" in first: return 2
    if "neg" in first: return 0
    return 1

def build_mlx_prompt(text):
    return (
        "<|system|>\n"
        "Eres un analista financiero experto. Clasifica el sentimiento de noticias bursátiles.\n"
        "- positive: crecimiento, ganancias, acuerdos favorables, subida de precio\n"
        "- negative: pérdidas, caídas, riesgos, recortes, deterioro\n"
        "- neutral: hecho informativo sin impacto claro\n"
        "Responde ÚNICAMENTE con una palabra: positive, negative o neutral.<|end|>\n"
        f"<|user|>\nNoticia financiera: {text}<|end|>\n"
        "<|assistant|>\n"
    )

def export_results(model_id, true_labels, pred_labels, headlines, inf_times_ms, errors, approach, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    safe = model_id.replace(':', '_').replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    csv_path = f"{results_dir}/nvidia_{safe}_predictions.csv"
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

    json_path = f"{results_dir}/nvidia_{safe}_metadata.json"
    meta = {
        "model": model_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": args.csv,
        "n_test": len(true_labels),
        "approach": approach,
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

def load_mlx_model(adapter_path=None):
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("⚠️  mlx_lm no instalado. Ejecuta: pip install mlx-lm")
        sys.exit(1)
    if adapter_path:
        from pathlib import Path
        if not Path(adapter_path).exists():
            print(f"⚠️  Adaptador no encontrado en '{adapter_path}'. Ejecuta primero train_mlx.py")
            sys.exit(1)
        print(f"\nCargando phi3.5 fine-tuneado desde '{adapter_path}'...")
        model, tokenizer = load(MLX_BASE_MODEL, adapter_path=adapter_path)
    else:
        print(f"\nCargando phi3.5 base MLX (sin fine-tuning)...")
        model, tokenizer = load(MLX_BASE_MODEL)
    print("✓ Modelo cargado")
    return model, tokenizer, generate

def run_mlx(model, tokenizer, generate_fn, model_id):
    print(f"\n{'='*65}\nMODELO: {model_id}\n{'='*65}")
    true_labels, pred_labels, headlines, inf_times_ms = [], [], [], []
    errors = 0

    for idx, s in enumerate(samples, 1):
        try:
            t0 = time.time()
            response = generate_fn(model, tokenizer, prompt=build_mlx_prompt(s["headline"]), max_tokens=10, verbose=False)
            elapsed_ms = (time.time() - t0) * 1000
            pred_label = parse(response)
        except Exception as e:
            print(f"  Error muestra {idx}: {e}")
            pred_label, elapsed_ms, errors = 1, 0.0, errors + 1

        true_labels.append(s["label"])
        pred_labels.append(pred_label)
        headlines.append(s["headline"])
        inf_times_ms.append(round(elapsed_ms, 1))

    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(samples)
    nivel = "ALTA ✓" if accuracy >= 0.8 else "MEDIA ⚠" if accuracy >= 0.6 else "BAJA ✗"
    print(f"Accuracy: {accuracy:.1%} ({sum(t==p for t,p in zip(true_labels,pred_labels))}/{len(samples)}) | {nivel} | {np.mean(inf_times_ms):.0f}ms/muestra")
    return true_labels, pred_labels, headlines, inf_times_ms, errors

if __name__ == "__main__":
    # 1. Base (sin adaptador)
    model_base, tok_base, gen_base = load_mlx_model(adapter_path=None)
    t, p, h, ms, err = run_mlx(model_base, tok_base, gen_base, "phi3.5 MLX base (sin fine-tuning)")
    print("\n📁 Exportando resultados base...")
    export_results("phi3.5_MLX_base", t, p, h, ms, err, "MLX base, sin fine-tuning", RESULTS_DIR_BASE)

    del model_base, tok_base

    # 2. Fine-tuneado (con adaptador LoRA)
    model_ft, tok_ft, gen_ft = load_mlx_model(adapter_path=args.adapter_path)
    t2, p2, h2, ms2, err2 = run_mlx(model_ft, tok_ft, gen_ft, f"phi3.5 MLX LoRA ({args.adapter_path})")
    print("\n📁 Exportando resultados fine-tuneado...")
    export_results("phi3.5_MLX_LoRA", t2, p2, h2, ms2, err2,
                   f"MLX LoRA fine-tuned, adapter: {args.adapter_path}", RESULTS_DIR_LORA)

    print("\n✓ Exportación completada")
