"""
Comparativa: phi3.5 base (Ollama) vs phi3.5 fine-tuneado con MLX LoRA
Requiere haber ejecutado train_mlx.py previamente.

Uso:
    python3 test_mlx_vs_ollama.py
    python3 test_mlx_vs_ollama.py --adapter-path lora_checkpoints
"""
import os
import time
import csv
import sys
import argparse
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ollama
from datasets import load_dataset
import random

# --- Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument("--adapter-path", type=str, default="../../models/lora_checkpoints", help="Ruta al adaptador LoRA generado por train_mlx.py")
args = parser.parse_args()

# --- Config ---
OLLAMA_MODEL    = "phi3.5:latest"
MLX_BASE_MODEL  = "microsoft/Phi-3.5-mini-instruct"
OLLAMA_TEMPERATURE = 0.0
N_TEST = 100
N_FEW_SHOT_PER_CLASS = 4
LABEL_MAP   = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# --- Cargar dataset y separar few-shot / test (misma seed que train_mlx.py) ---
print("Cargando dataset...")
dataset = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
rows = dataset["train"].to_list()
random.seed(42)
random.shuffle(rows)

# Reservar los mismos ejemplos que usó train_mlx.py para train/valid
# para garantizar que el test no se solapa con el entrenamiento
N_TRAIN_PER_CLASS = 500
N_VALID_PER_CLASS = 50
skip_pool = {0: 0, 1: 0, 2: 0}
few_shot_pool = {0: [], 1: [], 2: []}
test_pool = []

for row in rows:
    label = LABEL_MAP[row["sentiment"]]
    text, sentiment = row["news_headline"], row["sentiment"]
    # saltar los que fueron a train/valid
    if skip_pool[label] < N_TRAIN_PER_CLASS + N_VALID_PER_CLASS:
        skip_pool[label] += 1
    elif len(few_shot_pool[label]) < N_FEW_SHOT_PER_CLASS:
        few_shot_pool[label].append((text, sentiment))
    elif len(test_pool) < N_TEST:
        test_pool.append((text, label))

FEW_SHOT_EXAMPLES = few_shot_pool[0] + few_shot_pool[1] + few_shot_pool[2]
print(f"Few-shot: {len(FEW_SHOT_EXAMPLES)} | Test: {len(test_pool)} muestras")

# --- Prompt Ollama (few-shot) ---
def build_ollama_prompt(text):
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
    if "sentimiento:" in s:
        s = s.split("sentimiento:")[-1].strip()
    if "positive" in s: return 2
    if "negative" in s: return 0
    if "neutral"  in s: return 1
    first = s.split()[0] if s.split() else ""
    if "pos" in first: return 2
    if "neg" in first: return 0
    return 1

# --- Evaluar Ollama ---
def run_ollama():
    print(f"\n{'='*65}")
    print(f"MODELO: {OLLAMA_MODEL} base via Ollama ({len(test_pool)} muestras)")
    print(f"{'='*65}")
    correct, total_time, errors, rows_out = 0, 0.0, 0, []

    for idx, (text, true_label) in enumerate(test_pool, 1):
        try:
            t0 = time.time()
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=build_ollama_prompt(text),
                options={'temperature': OLLAMA_TEMPERATURE, 'num_predict': 20, 'stop': ['\n', '.']}
            )
            elapsed = time.time() - t0
            total_time += elapsed
            pred_label = parse(response['response'])
        except Exception as e:
            print(f"  Error muestra {idx}: {e}")
            pred_label, elapsed, errors = 1, 0.0, errors + 1

        hit = pred_label == true_label
        if hit: correct += 1
        rows_out.append({"id": idx, "noticia": text, "etiqueta_real": LABEL_NAMES[true_label],
                         "etiqueta_predicha": LABEL_NAMES[pred_label],
                         "correcto": "si" if hit else "no", "tiempo_ms": round(elapsed * 1000, 1)})
        if idx % 10 == 0:
            print(f"  Procesadas {idx}/{len(test_pool)}... ({correct} correctas)")

    accuracy = correct / len(test_pool)
    avg_ms = (total_time / len(test_pool)) * 1000
    print(f"\nAccuracy: {accuracy:.1%} | {avg_ms:.0f}ms/muestra")
    return {"model": f"{OLLAMA_MODEL} (base, Ollama)", "accuracy": accuracy,
            "avg_ms": avg_ms, "correct": correct, "errors": errors, "rows": rows_out}

# --- Cargar modelo MLX fine-tuneado ---
def load_mlx_model():
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("⚠️  mlx_lm no instalado. Ejecuta: pip install mlx-lm")
        sys.exit(1)

    from pathlib import Path
    if not Path(args.adapter_path).exists():
        print(f"⚠️  Adaptador no encontrado en '{args.adapter_path}'.")
        print("    Ejecuta primero: python3 train_mlx.py")
        sys.exit(1)

    print(f"\nCargando phi3.5 fine-tuneado desde '{args.adapter_path}'...")
    model, tokenizer = load(MLX_BASE_MODEL, adapter_path=args.adapter_path)
    print("✓ Modelo cargado")
    return model, tokenizer, generate

def predict_mlx(model, tokenizer, generate_fn, text):
    prompt = (
        "<|system|>\n"
        "Eres un analista financiero experto. Clasifica el sentimiento de noticias bursátiles.\n"
        "- positive: crecimiento, ganancias, acuerdos favorables, subida de precio\n"
        "- negative: pérdidas, caídas, riesgos, recortes, deterioro\n"
        "- neutral: hecho informativo sin impacto claro\n"
        "Responde ÚNICAMENTE con una palabra: positive, negative o neutral.<|end|>\n"
        f"<|user|>\nNoticia financiera: {text}<|end|>\n"
        "<|assistant|>\n"
    )
    response = generate_fn(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    return parse(response)

# --- Evaluar MLX ---
def run_mlx(model, tokenizer, generate_fn):
    print(f"\n{'='*65}")
    print(f"MODELO: phi3.5 fine-tuneado MLX LoRA ({len(test_pool)} muestras)")
    print(f"{'='*65}")
    correct, total_time, errors, rows_out = 0, 0.0, 0, []

    for idx, (text, true_label) in enumerate(test_pool, 1):
        try:
            t0 = time.time()
            pred_label = predict_mlx(model, tokenizer, generate_fn, text)
            elapsed = time.time() - t0
            total_time += elapsed
        except Exception as e:
            print(f"  Error muestra {idx}: {e}")
            pred_label, elapsed, errors = 1, 0.0, errors + 1

        hit = pred_label == true_label
        if hit: correct += 1
        rows_out.append({"id": idx, "noticia": text, "etiqueta_real": LABEL_NAMES[true_label],
                         "etiqueta_predicha": LABEL_NAMES[pred_label],
                         "correcto": "si" if hit else "no", "tiempo_ms": round(elapsed * 1000, 1)})
        if idx % 10 == 0:
            print(f"  Procesadas {idx}/{len(test_pool)}... ({correct} correctas)")

    accuracy = correct / len(test_pool)
    avg_ms = (total_time / len(test_pool)) * 1000
    print(f"\nAccuracy: {accuracy:.1%} | {avg_ms:.0f}ms/muestra")
    return {"model": f"phi3.5 MLX LoRA ({args.adapter_path})", "accuracy": accuracy,
            "avg_ms": avg_ms, "correct": correct, "errors": errors, "rows": rows_out}

# --- Exportar CSV ---
def export_csv(result):
    safe_name = result['model'].replace(':', '_').replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    filename = f"../../results/mlx/mlx_resultados_{safe_name}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto", "tiempo_ms"])
        writer.writeheader()
        writer.writerows(result['rows'])
    print(f"  CSV guardado: {filename}")

# --- Exportar Markdown ---
def export_markdown(results):
    filename = "../../results/mlx/mlx_resultados_resumen.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sorted_r = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Comparativa: phi3.5 base vs phi3.5 fine-tuneado (MLX LoRA)\n\n")
        f.write(f"**Fecha:** {ts}  \n")
        f.write(f"**Dataset:** prithvi1029/sentiment-analysis-for-financial-news  \n")
        f.write(f"**Muestras test:** {N_TEST}  \n")
        f.write(f"**Adaptador LoRA:** `{args.adapter_path}/`  \n\n")

        f.write("## Resumen comparativo\n\n")
        f.write("| Modelo | Correctas | Accuracy | Fiabilidad | Velocidad media |\n")
        f.write("|--------|-----------|----------|------------|-----------------|\n")
        for r in sorted_r:
            nivel = "ALTA" if r['accuracy'] >= 0.8 else "MEDIA" if r['accuracy'] >= 0.6 else "BAJA"
            f.write(f"| {r['model']} | {r['correct']}/{N_TEST} | {r['accuracy']:.1%} | {nivel} | {r['avg_ms']:.0f} ms/muestra |\n")

        f.write("\n## Detalle por modelo\n\n")
        for r in sorted_r:
            nivel = "ALTA ✓" if r['accuracy'] >= 0.8 else "MEDIA ⚠️" if r['accuracy'] >= 0.6 else "BAJA ✗"
            f.write(f"### {r['model']}\n\n")
            f.write(f"- **Accuracy:** {r['accuracy']:.1%}\n")
            f.write(f"- **Fiabilidad:** {nivel}\n")
            f.write(f"- **Correctas:** {r['correct']} de {N_TEST}\n")
            f.write(f"- **Velocidad media:** {r['avg_ms']:.0f} ms/muestra\n")
            if r['errors']:
                f.write(f"- **Errores:** {r['errors']}\n")
            fp  = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'positive')
            fn  = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'negative')
            fnu = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'neutral')
            f.write(f"\n**Errores por clase real:**\n\n| Clase | Errores |\n|-------|---------|\n")
            f.write(f"| positive | {fp} |\n| negative | {fn} |\n| neutral | {fnu} |\n\n")

    print(f"  Markdown guardado: {filename}")

# --- Main ---
if __name__ == "__main__":
    # 1. Verificar Ollama
    try:
        ollama.list()
    except Exception as e:
        print(f"⚠️  No se pudo conectar con Ollama: {e}")
        sys.exit(1)

    # 2. Evaluar phi3.5 base
    result_ollama = run_ollama()

    # 3. Cargar y evaluar phi3.5 fine-tuneado
    mlx_model, mlx_tokenizer, mlx_generate = load_mlx_model()
    result_mlx = run_mlx(mlx_model, mlx_tokenizer, mlx_generate)

    results = [result_ollama, result_mlx]

    # 4. Resumen
    print(f"\n{'='*65}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*65}")
    print(f"{'Modelo':<40} {'Accuracy':<10} {'Velocidad'}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        acc_str = f"{r['accuracy']:.1%}"
        print(f"{r['model']:<40} {acc_str:<10} {r['avg_ms']:.0f}ms/muestra")

    # 5. Exportar
    print("\n📁 Exportando resultados...")
    for r in results:
        export_csv(r)
    export_markdown(results)
    print("✓ Exportación completada")
