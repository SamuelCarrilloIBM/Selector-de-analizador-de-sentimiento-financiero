"""
Compara 3 modelos Ollama en análisis de sentimiento financiero.
- Few-shot examples extraídos del dataset real (sin solaparse con test)
- 100 muestras de test del dataset
- Temperatura 0.0, prompt refinado con definiciones y formato estricto
- Exporta CSV por modelo y resumen Markdown final
"""
import os
import time
import csv
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ollama
from datasets import load_dataset
import random

MODELS = ['llama3.2:latest', 'llama3.1:8b', 'phi3.5:latest']
OLLAMA_TEMPERATURE = 0.0
N_TEST = 100
N_FEW_SHOT_PER_CLASS = 4  # 4 ejemplos por clase = 12 en total
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# --- Cargar dataset ---
print("Cargando dataset...")
dataset = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
rows = dataset["train"].to_list()
random.seed(42)
random.shuffle(rows)

# Separar few-shot (primeros N por clase) y test (siguientes 100)
few_shot_pool = {0: [], 1: [], 2: []}
test_pool = []

for row in rows:
    label = LABEL_MAP[row["sentiment"]]
    if len(few_shot_pool[label]) < N_FEW_SHOT_PER_CLASS:
        few_shot_pool[label].append((row["news_headline"], row["sentiment"]))
    elif len(test_pool) < N_TEST:
        test_pool.append((row["news_headline"], label))

FEW_SHOT_EXAMPLES = few_shot_pool[0] + few_shot_pool[1] + few_shot_pool[2]
print(f"Few-shot: {len(FEW_SHOT_EXAMPLES)} ejemplos ({N_FEW_SHOT_PER_CLASS} por clase)")
print(f"Test: {len(test_pool)} muestras")

# --- Prompt ---
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
        "- No añadas explicaciones, puntuación ni texto adicional\n"
        "- Tu respuesta debe ser la primera y única palabra\n\n"
        f"EJEMPLOS:\n{examples}\n\n"
        f'Texto: "{text}"\nSENTIMIENTO:'
    )

def parse(raw):
    s = raw.strip().lower()
    if "sentimiento:" in s:
        s = s.split("sentimiento:")[-1].strip()
    if "positive" in s: return 2
    if "negative" in s: return 0
    if "neutral" in s: return 1
    first = s.split()[0] if s.split() else ""
    if "pos" in first: return 2
    if "neg" in first: return 0
    return 1

# --- Evaluar modelo ---
def run_model(model_name):
    print(f"\n{'='*65}")
    print(f"MODELO: {model_name}  ({len(test_pool)} muestras)")
    print(f"{'='*65}")

    correct = 0
    total_time = 0.0
    errors = 0
    rows_out = []  # para CSV

    for idx, (text, true_label) in enumerate(test_pool, 1):
        try:
            t0 = time.time()
            response = ollama.generate(
                model=model_name,
                prompt=build_prompt(text),
                options={'temperature': OLLAMA_TEMPERATURE, 'num_predict': 20, 'stop': ['\n', '.']}
            )
            elapsed = time.time() - t0
            total_time += elapsed
            pred_label = parse(response['response'])
        except Exception as e:
            print(f"  Error muestra {idx}: {e}")
            pred_label = 1
            elapsed = 0.0
            errors += 1

        hit = pred_label == true_label
        if hit:
            correct += 1

        rows_out.append({
            "id": idx,
            "noticia": text,
            "etiqueta_real": LABEL_NAMES[true_label],
            "etiqueta_predicha": LABEL_NAMES[pred_label],
            "correcto": "si" if hit else "no",
            "tiempo_ms": round(elapsed * 1000, 1)
        })

        if idx % 10 == 0:
            print(f"  Procesadas {idx}/{len(test_pool)}... ({correct} correctas hasta ahora)")

    accuracy = correct / len(test_pool)
    avg_ms = (total_time / len(test_pool)) * 1000
    nivel = "ALTA ✓" if accuracy >= 0.8 else "MEDIA ⚠" if accuracy >= 0.6 else "BAJA ✗"

    print(f"\nCorrects: {correct}/{len(test_pool)} | Accuracy: {accuracy:.1%} | {nivel} | {avg_ms:.0f}ms/muestra")
    if errors:
        print(f"Errores de conexión: {errors}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "avg_ms": avg_ms,
        "correct": correct,
        "errors": errors,
        "rows": rows_out
    }

# --- Verificar modelos disponibles ---
try:
    models_response = ollama.list()
    available = [m.get('model', m.get('name', '')) for m in models_response.get('models', [])]
except Exception as e:
    print(f"⚠️  No se pudo conectar con Ollama: {e}")
    exit(1)

# --- Correr comparativa ---
results = []
for model in MODELS:
    if not any(model.split(':')[0] in m for m in available):
        print(f"⚠️  {model} no encontrado, saltando...")
        continue
    results.append(run_model(model))

# --- Resumen final ---
print(f"\n{'='*65}")
print("RESUMEN COMPARATIVO — 100 muestras del dataset real")
print(f"{'='*65}")
print(f"{'Modelo':<22} {'Accuracy':<10} {'Fiabilidad':<12} {'Velocidad'}")
print("-" * 65)
for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
    nivel = "ALTA ✓" if r['accuracy'] >= 0.8 else "MEDIA ⚠" if r['accuracy'] >= 0.6 else "BAJA ✗"
    acc_str = f"{r['accuracy']:.1%}"
    print(f"{r['model']:<22} {acc_str:<10} {nivel:<12} {r['avg_ms']:.0f}ms/muestra")

# --- Exportar CSV por modelo ---
def export_csv(result):
    safe_name = result['model'].replace(':', '_').replace('/', '_')
    filename = f"ollama_resultados_{safe_name}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "noticia", "etiqueta_real", "etiqueta_predicha", "correcto", "tiempo_ms"])
        writer.writeheader()
        writer.writerows(result['rows'])
    print(f"  CSV guardado: {filename}")
    return filename

# --- Exportar Markdown resumen ---
def export_markdown(results, n_test):
    filename = "ollama_resultados_resumen.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Resultados Comparativa Ollama — Análisis de Sentimiento Financiero\n\n")
        f.write(f"**Fecha:** {ts}  \n")
        f.write(f"**Dataset:** prithvi1029/sentiment-analysis-for-financial-news  \n")
        f.write(f"**Muestras evaluadas:** {n_test}  \n")
        f.write(f"**Temperatura:** {OLLAMA_TEMPERATURE}  \n")
        f.write(f"**Estrategia:** Few-shot ({len(FEW_SHOT_EXAMPLES)} ejemplos del dataset, {N_FEW_SHOT_PER_CLASS} por clase)  \n\n")

        f.write("## Resumen comparativo\n\n")
        f.write("| Modelo | Correctas | Accuracy | Fiabilidad | Velocidad media |\n")
        f.write("|--------|-----------|----------|------------|-----------------|\n")
        for r in sorted_results:
            nivel = "ALTA" if r['accuracy'] >= 0.8 else "MEDIA" if r['accuracy'] >= 0.6 else "BAJA"
            f.write(f"| {r['model']} | {r['correct']}/{n_test} | {r['accuracy']:.1%} | {nivel} | {r['avg_ms']:.0f} ms/muestra |\n")

        f.write("\n## Detalle por modelo\n\n")
        for r in sorted_results:
            nivel = "ALTA ✓" if r['accuracy'] >= 0.8 else "MEDIA ⚠️" if r['accuracy'] >= 0.6 else "BAJA ✗"
            safe_name = r['model'].replace(':', '_').replace('/', '_')
            f.write(f"### {r['model']}\n\n")
            f.write(f"- **Accuracy:** {r['accuracy']:.1%}\n")
            f.write(f"- **Fiabilidad:** {nivel}\n")
            f.write(f"- **Correctas:** {r['correct']} de {n_test}\n")
            f.write(f"- **Velocidad media:** {r['avg_ms']:.0f} ms/muestra\n")
            if r['errors']:
                f.write(f"- **Errores de conexión:** {r['errors']}\n")
            f.write(f"- **CSV detalle:** `ollama_resultados_{safe_name}.csv`\n\n")

            # distribución de errores por clase
            fp = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'positive')
            fn = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'negative')
            fnu = sum(1 for row in r['rows'] if row['correcto'] == 'no' and row['etiqueta_real'] == 'neutral')
            f.write(f"**Errores por clase real:**\n\n")
            f.write(f"| Clase real | Errores |\n|------------|--------|\n")
            f.write(f"| positive | {fp} |\n| negative | {fn} |\n| neutral | {fnu} |\n\n")

        f.write("## Configuración del prompt\n\n")
        f.write("```\n")
        f.write(build_prompt("[texto de ejemplo]"))
        f.write("\n```\n")

    print(f"  Markdown guardado: {filename}")

print("\n📁 Exportando resultados...")
for r in results:
    export_csv(r)
export_markdown(results, N_TEST)
print("✓ Exportación completada")
