"""
Evalúa phi3.5 MLX base (sin fine-tuning) vs phi3.5 MLX fine-tuneado (LoRA)
sobre el dataset real de noticias NVIDIA (nvidia_testing.csv).
Requiere haber ejecutado train_mlx.py previamente.

Uso:
    python3 test_mlx_nvidia.py
    python3 test_mlx_nvidia.py --adapter-path lora_checkpoints --csv nvidia_testing.csv
"""
import os
import csv
import sys
import time
import argparse
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument("--adapter-path", type=str, default="../../models/lora_checkpoints")
parser.add_argument("--csv",          type=str, default="../../data/raw/nvidia_testing.csv")
args = parser.parse_args()

MLX_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LABEL_MAP   = {"positive": 2, "negative": 0, "neutral": 1}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# --- Cargar CSV ---
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

# --- Parse respuesta ---
def parse(raw):
    s = raw.strip().lower()
    if "positive" in s: return 2
    if "negative" in s: return 0
    if "neutral"  in s: return 1
    first = s.split()[0] if s.split() else ""
    if "pos" in first: return 2
    if "neg" in first: return 0
    return 1

# --- Prompt MLX (mismo para ambos modelos, comparación justa) ---
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

# --- Cargar modelo MLX (con o sin adaptador) ---
def load_mlx_model(adapter_path=None):
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("⚠️  mlx_lm no instalado. Ejecuta: pip install mlx-lm")
        sys.exit(1)
    if adapter_path:
        from pathlib import Path
        if not Path(adapter_path).exists():
            print(f"⚠️  Adaptador no encontrado en '{adapter_path}'.")
            print("    Ejecuta primero: python3 train_mlx.py")
            sys.exit(1)
        print(f"\nCargando phi3.5 fine-tuneado desde '{adapter_path}'...")
        model, tokenizer = load(MLX_BASE_MODEL, adapter_path=adapter_path)
    else:
        print(f"\nCargando phi3.5 base MLX (sin fine-tuning)...")
        model, tokenizer = load(MLX_BASE_MODEL)
    print("✓ Modelo cargado")
    return model, tokenizer, generate

# --- Evaluar un modelo MLX ---
def run_mlx(model, tokenizer, generate_fn, model_name):
    print(f"\n{'='*65}")
    print(f"MODELO: {model_name}")
    print(f"{'='*65}")
    correct, total_time, errors, rows_out = 0, 0.0, 0, []

    for idx, s in enumerate(samples, 1):
        try:
            t0 = time.time()
            response = generate_fn(model, tokenizer, prompt=build_mlx_prompt(s["headline"]), max_tokens=10, verbose=False)
            elapsed = time.time() - t0
            total_time += elapsed
            pred_label = parse(response)
        except Exception as e:
            print(f"  Error muestra {idx}: {e}")
            pred_label, elapsed, errors = 1, 0.0, errors + 1

        hit = pred_label == s["label"]
        if hit: correct += 1
        rows_out.append({
            "id": idx, "fecha": s["date"], "noticia": s["headline"],
            "etiqueta_real": LABEL_NAMES[s["label"]],
            "etiqueta_predicha": LABEL_NAMES[pred_label],
            "correcto": "si" if hit else "no",
            "score_real": s["score"], "tiempo_ms": round(elapsed * 1000, 1)
        })

    accuracy = correct / len(samples)
    avg_ms = (total_time / len(samples)) * 1000
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(samples)}) | {avg_ms:.0f}ms/muestra")
    return {"model": model_name, "accuracy": accuracy,
            "avg_ms": avg_ms, "correct": correct, "errors": errors, "rows": rows_out}

# --- Exportar CSV ---
def export_csv(result):
    safe = result['model'].replace(':', '_').replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    filename = f"../../results/mlx/nvidia_resultados_{safe}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "fecha", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto", "score_real", "tiempo_ms"])
        writer.writeheader()
        writer.writerows(result['rows'])
    print(f"  CSV guardado: {filename}")

# --- Exportar Markdown ---
def export_markdown(results):
    filename = "../../results/mlx/nvidia_resultados_resumen.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sorted_r = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Comparativa NVIDIA: phi3.5 MLX base vs phi3.5 MLX fine-tuneado (LoRA)\n\n")
        f.write(f"**Fecha:** {ts}  \n**Dataset:** `{args.csv}` ({len(samples)} noticias NVIDIA)  \n")
        f.write(f"**Adaptador LoRA:** `{args.adapter_path}/`  \n\n")

        f.write("## Resumen comparativo\n\n")
        f.write("| Modelo | Correctas | Accuracy | Fiabilidad | Velocidad media |\n")
        f.write("|--------|-----------|----------|------------|-----------------|\n")
        for r in sorted_r:
            nivel = "ALTA" if r['accuracy'] >= 0.8 else "MEDIA" if r['accuracy'] >= 0.6 else "BAJA"
            f.write(f"| {r['model']} | {r['correct']}/{len(samples)} | {r['accuracy']:.1%} | {nivel} | {r['avg_ms']:.0f} ms/muestra |\n")

        f.write("\n## Detalle por modelo\n\n")
        for r in sorted_r:
            nivel = "ALTA ✓" if r['accuracy'] >= 0.8 else "MEDIA ⚠️" if r['accuracy'] >= 0.6 else "BAJA ✗"
            f.write(f"### {r['model']}\n\n")
            f.write(f"- **Accuracy:** {r['accuracy']:.1%} ({r['correct']}/{len(samples)})\n")
            f.write(f"- **Fiabilidad:** {nivel}\n")
            f.write(f"- **Velocidad media:** {r['avg_ms']:.0f} ms/muestra\n")
            fp  = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'positive')
            fn  = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'negative')
            fnu = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'neutral')
            f.write(f"\n**Errores por clase real:**\n\n| Clase | Errores |\n|-------|---------|\n")
            f.write(f"| positive | {fp} |\n| negative | {fn} |\n| neutral | {fnu} |\n\n")
            wrong = [row for row in r['rows'] if row['correcto'] == 'no']
            if wrong:
                f.write("**Predicciones incorrectas:**\n\n")
                f.write("| Fecha | Noticia | Real | Predicha |\n|-------|---------|------|----------|\n")
                for row in wrong:
                    headline = row['noticia'][:70] + "..." if len(row['noticia']) > 70 else row['noticia']
                    f.write(f"| {row['fecha']} | {headline} | {row['etiqueta_real']} | {row['etiqueta_predicha']} |\n")
                f.write("\n")

    print(f"  Markdown guardado: {filename}")

# --- Main ---
if __name__ == "__main__":
    # 1. Cargar phi3.5 base (sin adaptador) y evaluar
    model_base, tok_base, gen_base = load_mlx_model(adapter_path=None)
    result_base = run_mlx(model_base, tok_base, gen_base, "phi3.5 MLX (base, sin fine-tuning)")

    # Liberar memoria antes de cargar el siguiente modelo
    del model_base, tok_base

    # 2. Cargar phi3.5 fine-tuneado y evaluar
    model_ft, tok_ft, gen_ft = load_mlx_model(adapter_path=args.adapter_path)
    result_ft = run_mlx(model_ft, tok_ft, gen_ft, f"phi3.5 MLX LoRA ({args.adapter_path})")

    results = [result_base, result_ft]

    # 3. Resumen
    print(f"\n{'='*65}")
    print("RESUMEN COMPARATIVO — NVIDIA")
    print(f"{'='*65}")
    print(f"{'Modelo':<40} {'Accuracy':<10} {'Velocidad'}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        acc_str = f"{r['accuracy']:.1%}"
        print(f"{r['model']:<40} {acc_str:<10} {r['avg_ms']:.0f}ms/muestra")

    # 4. Exportar
    print("\n📁 Exportando resultados...")
    for r in results:
        export_csv(r)
    export_markdown(results)
    print("✓ Exportación completada")
