"""
Fine-tuning de Phi-3.5 con MLX LoRA para análisis de sentimiento financiero.
Ejecutar una vez; el adaptador queda guardado en lora_checkpoints/ para uso posterior.

Uso:
    python train_mlx.py
    python train_mlx.py --iters 300 --train-per-class 80
"""
import os
import json
import subprocess
import sys
import argparse
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datasets import load_dataset

# --- Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument("--iters",            type=int, default=500,  help="Iteraciones de entrenamiento")
parser.add_argument("--train-per-class",  type=int, default=500,  help="Ejemplos de entrenamiento por clase (usa todos si hay suficientes)")
parser.add_argument("--valid-per-class",  type=int, default=50,   help="Ejemplos de validación por clase")
parser.add_argument("--batch-size",       type=int, default=4,    help="Batch size")
parser.add_argument("--learning-rate",    type=float, default=2e-4)
parser.add_argument("--adapter-path",     type=str, default="../../models/lora_checkpoints")
parser.add_argument("--data-dir",         type=str, default="../../data/mlx")
args = parser.parse_args()

MLX_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# Prompt template Phi-3.5 instruct con contexto financiero enriquecido
SYSTEM_PROMPT = (
    "Eres un analista financiero experto. Clasifica el sentimiento de noticias bursátiles.\n"
    "- positive: crecimiento, ganancias, acuerdos favorables, subida de precio, expansión\n"
    "- negative: pérdidas, caídas, riesgos, recortes, quiebra, bajada de precio, deterioro\n"
    "- neutral: hecho informativo sin impacto claro positivo ni negativo\n"
    "Responde ÚNICAMENTE con una palabra: positive, negative o neutral."
)

PHI_TEMPLATE = (
    "<|system|>\n{system}<|end|>\n"
    "<|user|>\nNoticia financiera: {headline}<|end|>\n"
    "<|assistant|>\n{sentiment}<|end|>"
)

def format_example(headline, sentiment):
    return {"text": PHI_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        headline=headline,
        sentiment=sentiment
    )}

# --- Cargar y dividir dataset ---
print("Cargando dataset...")
dataset = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
rows = dataset["train"].to_list()
random.seed(42)
random.shuffle(rows)

train_pool = {0: [], 1: [], 2: []}
valid_pool = {0: [], 1: [], 2: []}

for row in rows:
    label = LABEL_MAP[row["sentiment"]]
    text, sentiment = row["news_headline"], row["sentiment"]
    if len(train_pool[label]) < args.train_per_class:
        train_pool[label].append((text, sentiment))
    elif len(valid_pool[label]) < args.valid_per_class:
        valid_pool[label].append((text, sentiment))

train_data = train_pool[0] + train_pool[1] + train_pool[2]
valid_data = valid_pool[0] + valid_pool[1] + valid_pool[2]
random.shuffle(train_data)

print(f"Train: {len(train_data)} ejemplos ({args.train_per_class} por clase)")
print(f"Valid: {len(valid_data)} ejemplos ({args.valid_per_class} por clase)")

# --- Guardar JSONL ---
Path(args.data_dir).mkdir(exist_ok=True)
for split, pairs in [("train", train_data), ("valid", valid_data)]:
    path = f"{args.data_dir}/{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for text, sentiment in pairs:
            f.write(json.dumps(format_example(text, sentiment), ensure_ascii=False) + "\n")
    print(f"  {path} guardado ({len(pairs)} ejemplos)")

# Necesario aunque no se use para test
test_path = f"{args.data_dir}/test.jsonl"
if not Path(test_path).exists():
    with open(test_path, "w", encoding="utf-8") as f:
        for text, sentiment in valid_data[:5]:
            f.write(json.dumps(format_example(text, sentiment), ensure_ascii=False) + "\n")

# --- Fine-tuning ---
Path(args.adapter_path).mkdir(exist_ok=True)

cmd = [
    sys.executable, "-m", "mlx_lm", "lora",
    "--model",         MLX_BASE_MODEL,
    "--train",
    "--data",          args.data_dir,
    "--batch-size",    str(args.batch_size),
    "--iters",         str(args.iters),
    "--learning-rate", str(args.learning_rate),
    "--adapter-path",  args.adapter_path,
    "--save-every",    str(max(1, args.iters // 4)),
]

print(f"\n{'='*65}")
print(f"FINE-TUNING: {MLX_BASE_MODEL}")
print(f"  iters={args.iters} | batch={args.batch_size} | lr={args.learning_rate}")
print(f"  adaptador → {args.adapter_path}/")
print(f"{'='*65}\n")

result = subprocess.run(cmd)

if result.returncode != 0:
    print("\n⚠️  Fine-tuning falló.")
    print("Asegúrate de tener mlx-lm instalado: pip install mlx-lm")
    sys.exit(1)

print(f"\n✓ Adaptador LoRA guardado en: {args.adapter_path}/")
print(f"  Para usar el modelo fine-tuneado en tu script de evaluación,")
print(f"  cárgalo con: mlx_lm.load('{MLX_BASE_MODEL}', adapter_path='{args.adapter_path}')")
